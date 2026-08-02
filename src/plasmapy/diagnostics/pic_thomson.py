r"""
Synthetic Thomson scattering spectra from particle-in-cell (PIC) simulation output.

This module turns the phase-space output of a PIC code (OSIRIS, WarpX, ...) into
synthetic electron-plasma-wave (EPW) and ion-acoustic-wave (IAW) Thomson scattering
spectra, using the arbitrary-VDF forward model in
`~plasmapy.diagnostics.thomson.arbitrary_forwardmodel`.

Only the *readers* are code-specific. Every reader produces a `PICPhaseSpace`: a
reduced one-velocity-dimension phase space :math:`f(t, v, x)` in SI units. All of the
conditioning and forward-modelling below that boundary is code-agnostic, so
supporting an additional PIC code means writing one reader.

.. note::

   This module depends on ``thomson.arbitrary_forwardmodel``, which exists only in
   the Schaeffer-Lab fork of PlasmaPy. It is not upstream-mergeable as written.

Conditioning pipeline
---------------------

Raw PIC phase space is noisy, is defined on a bounded velocity grid, and carries an
arbitrary normalisation, while the forward model expects a smooth probability density
with :math:`\int f \, dv = 1`. `condition_phase_space` applies, in order:

1. `smooth_vdf` -- repeated boxcar average along the velocity axis, to suppress
   macroparticle shot noise.
2. `taper_vdf_edges` -- half-cosine rolloff of the distribution tails, replacing the
   discontinuity where the PIC noise floor meets the edge of the velocity grid.
3. `normalize_vdf` -- rescale each ``(time, position)`` slice to unit integral.
4. A small positive floor, to keep the forward model's divisions finite.

Tapering before normalising (the reverse of the order used by the ``osiris2thomson``
pipeline this module is derived from) leaves the resulting distribution shapes
unchanged -- the taper threshold is relative to the peak -- but guarantees the unit
normalisation the forward model actually assumes.
"""

__all__ = [
    "PICPhaseSpace",
    "ThomsonSpectrogram",
    "condition_phase_space",
    "from_arrays",
    "normalize_vdf",
    "number_density",
    "rescale_velocity_axis",
    "smooth_vdf",
    "species_presence_mask",
    "spectra_from_phase_spaces",
    "taper_vdf_edges",
]

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d
from tqdm import tqdm

# Imported from the submodule rather than the package so that this module can be
# imported from `plasmapy.diagnostics.__init__` without depending on the order in
# which that file imports its submodules.
from plasmapy.diagnostics.thomson import arbitrary_forwardmodel
from plasmapy.particles import Particle

#: Axis of a ``(n_time, n_v, n_x)`` phase-space array that holds velocity.
VELOCITY_AXIS = 1

#: Default positive floor applied after normalisation. Small enough not to perturb
#: the distribution, large enough to keep the forward model's divisions finite.
DEFAULT_FLOOR = 1e-30


def _si_values(quantity, unit: u.UnitBase) -> np.ndarray:
    """
    Return *quantity* as a bare float array in *unit*.

    Accepts either a `~astropy.units.Quantity` -- which is converted -- or a plain
    array, which is assumed to already be in *unit*.
    """
    if isinstance(quantity, u.Quantity):
        return np.asarray(quantity.to(unit).value, dtype=np.float64)
    return np.asarray(quantity, dtype=np.float64)


@dataclass(frozen=True)
class PICPhaseSpace:
    r"""
    A reduced one-velocity-dimension phase space for a single particle species.

    This is the contract between the code-specific readers and the code-agnostic
    forward-modelling pipeline. Construct one with `from_arrays` (which validates
    shapes and units) rather than calling this class directly.

    Parameters
    ----------
    f : `~numpy.ndarray`
        Phase-space density, shape ``(n_time, n_v, n_x)``, in arbitrary units. It
        need not be normalised, but it must be *proportional* to the true
        phase-space density -- when histogramming raw macroparticles, weights must
        be included.
    v : `~numpy.ndarray`
        Lab-frame velocity along the diagnostic axis, shape ``(n_v,)``, in m/s.
        Strictly increasing. Species need not share a velocity grid.
    x : `~numpy.ndarray`
        Position along the diagnostic axis, shape ``(n_x,)``, in m.
    t : `~numpy.ndarray`
        Times of the phase-space dumps, shape ``(n_time,)``, in s.
    label : `str`
        A `~plasmapy.particles.particle_class.ParticleLike` identifier for the
        species, for example ``"e-"``, ``"p+"``, or ``"Al 13+"``.
    is_electron : `bool`
        Whether this species is an electron population.
    meta : `dict`
        Provenance: PIC code, source paths, and any normalisations applied.

    Notes
    -----
    Velocities must be lab-frame and relativistically correct. Codes that store
    proper velocity :math:`u = \gamma v / c` (OSIRIS does) must convert with
    :math:`v = u c / \sqrt{1 + u^2}` in the reader.
    """

    f: np.ndarray
    v: np.ndarray
    x: np.ndarray
    t: np.ndarray
    label: str
    is_electron: bool
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of the phase-space array, ``(n_time, n_v, n_x)``."""
        return self.f.shape

    @property
    def particle(self) -> Particle:
        """The species as a `~plasmapy.particles.particle_class.Particle`."""
        return Particle(self.label)

    def slice_position(self, position: float) -> tuple[int, np.ndarray]:
        """
        Return the phase space at the grid point nearest *position*.

        Parameters
        ----------
        position : `float` or `~astropy.units.Quantity`
            Position along `x`, in m if given as a bare float.

        Returns
        -------
        index : `int`
            Index of the selected grid point.
        f_slice : `~numpy.ndarray`
            Phase space at that point, shape ``(n_time, n_v)``.
        """
        position = float(_si_values(position, u.m))
        if position < self.x.min() or position > self.x.max():
            raise ValueError(
                f"position {position} m is outside the spatial axis "
                f"[{self.x.min()}, {self.x.max()}] m of species {self.label!r}."
            )
        index = int(np.argmin(np.abs(self.x - position)))
        return index, self.f[:, :, index]

    def __repr__(self) -> str:
        """Summarise the species and the phase-space dimensions."""
        n_time, n_v, n_x = self.f.shape
        return (
            f"PICPhaseSpace(label={self.label!r}, is_electron={self.is_electron}, "
            f"n_time={n_time}, n_v={n_v}, n_x={n_x})"
        )


def from_arrays(
    f,
    v,
    x,
    t,
    label: str,
    *,
    is_electron: bool | None = None,
    meta: dict[str, Any] | None = None,
) -> PICPhaseSpace:
    """
    Build a validated `PICPhaseSpace` from arrays.

    This is the generic entry point: any PIC code, or an analytic distribution, can
    be fed to the pipeline by producing the four arrays below.

    Parameters
    ----------
    f : array_like
        Phase-space density, shape ``(n_time, n_v, n_x)``, arbitrary units.
    v : array_like or `~astropy.units.Quantity`
        Lab-frame velocity axis, shape ``(n_v,)``. Bare arrays are assumed to be
        in m/s.
    x : array_like or `~astropy.units.Quantity`
        Spatial axis, shape ``(n_x,)``. Bare arrays are assumed to be in m.
    t : array_like or `~astropy.units.Quantity`
        Time axis, shape ``(n_time,)``. Bare arrays are assumed to be in s.
    label : `str`
        Species identifier, for example ``"e-"`` or ``"Al 13+"``.
    is_electron : `bool`, optional
        Whether this is an electron population. Inferred from *label* when omitted.
    meta : `dict`, optional
        Provenance to attach to the result.

    Returns
    -------
    `PICPhaseSpace`

    Raises
    ------
    `ValueError`
        If the array shapes are inconsistent or the velocity axis is not strictly
        increasing.
    """
    f = np.asarray(f, dtype=np.float64)
    v = _si_values(v, u.m / u.s)
    x = _si_values(x, u.m)
    t = _si_values(t, u.s)

    if f.ndim != 3:
        raise ValueError(
            f"f must be 3-D with shape (n_time, n_v, n_x); got ndim={f.ndim}."
        )
    for name, axis, n_expected in (
        ("t", t, f.shape[0]),
        ("v", v, f.shape[1]),
        ("x", x, f.shape[2]),
    ):
        if axis.ndim != 1:
            raise ValueError(f"{name} must be 1-D; got ndim={axis.ndim}.")
        if axis.size != n_expected:
            raise ValueError(
                f"{name} has size {axis.size} but f implies {n_expected} "
                f"(f.shape={f.shape})."
            )
    if np.any(np.diff(v) <= 0):
        raise ValueError("v must be strictly increasing.")

    if is_electron is None:
        is_electron = Particle(label).is_category("electron")

    return PICPhaseSpace(
        f=f,
        v=v,
        x=x,
        t=t,
        label=label,
        is_electron=bool(is_electron),
        meta=dict(meta or {}),
    )


def number_density(f, v, axis: int = VELOCITY_AXIS) -> np.ndarray:
    r"""
    Zeroth velocity moment :math:`\int f \, dv`.

    Parameters
    ----------
    f : array_like
        Phase-space density.
    v : array_like
        Velocity axis, matching ``f.shape[axis]``.
    axis : `int`
        Axis of *f* holding velocity.

    Returns
    -------
    `~numpy.ndarray`
        *f* integrated over velocity, with *axis* removed. In the same arbitrary
        units as *f* times m/s.

    Notes
    -----
    Integration is trapezoidal rather than Simpson's rule, matching the moments the
    forward model takes internally, so that a distribution normalised here is seen
    as unit-normalised there.
    """
    return np.trapezoid(np.asarray(f, dtype=np.float64), np.asarray(v), axis=axis)


def smooth_vdf(
    f,
    window: int,
    iterations: int = 1,
    axis: int = VELOCITY_AXIS,
) -> np.ndarray:
    """
    Suppress macroparticle shot noise with a repeated boxcar average.

    Applies a moving average of width *window* along the velocity axis
    *iterations* times. Repeated boxcar passes approach a Gaussian kernel while
    staying cheap and strictly local.

    Parameters
    ----------
    f : array_like
        Phase-space density.
    window : `int`
        Width of the moving-average window, in velocity bins.
    iterations : `int`
        Number of passes. ``0`` returns the input unchanged (aside from cleaning
        non-finite values), which is often what is wanted for ion distributions --
        the IAW feature is narrow and easily washed out.
    axis : `int`
        Axis of *f* holding velocity.

    Returns
    -------
    `~numpy.ndarray`
        The smoothed distribution, same shape as *f*.

    Notes
    -----
    Non-finite values are replaced with zero. Negative values are *not* modified
    here; `normalize_vdf` clips them, with a warning. Taking the absolute value
    would reflect shot noise into fabricated signal.
    """
    if window < 1:
        raise ValueError(f"window must be at least 1; got {window}.")
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative; got {iterations}.")

    f = np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    for _ in range(iterations):
        f = uniform_filter1d(f, size=window, axis=axis, mode="nearest")
    return f


def normalize_vdf(f, v, axis: int = VELOCITY_AXIS) -> np.ndarray:
    r"""
    Rescale every slice of *f* to unit integral over velocity.

    The forward model treats ``efn`` and ``ifn`` as probability densities: it takes
    their moments directly and interpolates them to evaluate :math:`S(k, \omega)`.
    Each species must therefore be normalised **on its own velocity axis** -- using
    another species' axis rescales the distribution by the ratio of the two grid
    spacings and biases that species' susceptibility.

    Parameters
    ----------
    f : array_like
        Phase-space density.
    v : array_like
        Velocity axis *of this species*, matching ``f.shape[axis]``.
    axis : `int`
        Axis of *f* holding velocity.

    Returns
    -------
    `~numpy.ndarray`
        The normalised distribution, same shape as *f*.

    Warns
    -----
    `RuntimeWarning`
        If negative values were clipped. PIC shot noise and interpolation can leave
        small negative values; they are clipped to zero rather than reflected.
    """
    f = np.nan_to_num(np.asarray(f, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    n_negative = int(np.count_nonzero(f < 0))
    if n_negative:
        warnings.warn(
            f"normalize_vdf: clipped {n_negative} negative value(s) to zero.",
            RuntimeWarning,
            stacklevel=2,
        )
    f = np.clip(f, a_min=0.0, a_max=None)

    integral = np.expand_dims(number_density(f, v, axis=axis), axis=axis)
    # Slices that are entirely empty stay empty rather than dividing by zero.
    # A `where` mask is exact, unlike adding a small epsilon to the
    # denominator, which biases low-amplitude slices.
    return np.divide(f, integral, out=np.zeros_like(f), where=integral > 0)


def _index_space_variance(cols: np.ndarray) -> np.ndarray:
    """
    Variance of each column of *cols* over its row index.

    Used only as a cheap, grid-spacing-independent diagnostic of how much the
    taper has changed the width of a distribution.
    """
    position = np.arange(cols.shape[0])[:, None]
    total = cols.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (cols * position).sum(axis=0) / total
        return (cols * (position - mean) ** 2).sum(axis=0) / total


def taper_vdf_edges(
    f,
    threshold_frac: float = 0.005,
    axis: int = VELOCITY_AXIS,
    max_taper_bins: int | None = None,
    pedestal_warning: float | None = 0.05,
) -> np.ndarray:
    r"""
    Replace the sharp PIC noise floor at the velocity-grid edges with a smooth taper.

    For each one-dimensional slice along *axis*, locates the outermost bins whose
    value exceeds ``threshold_frac`` times the slice peak, and replaces everything
    beyond them with a half-cosine rolloff towards zero. Without this, the forward
    model sees a discontinuity where the distribution is truncated, which shows up
    as ringing in the computed susceptibility.

    Parameters
    ----------
    f : array_like
        Phase-space density.
    threshold_frac : `float`
        Fraction of the slice peak defining the edge of the signal. About 0.005
        suits typical PIC noise floors.
    axis : `int`
        Axis of *f* holding velocity.
    max_taper_bins : `int`, optional
        Longest rolloff to use, in velocity bins; beyond it the distribution is set
        to zero. The default, `None`, stretches the rolloff across the whole
        remaining tail, out to the grid boundary.
    pedestal_warning : `float`, optional
        Emit a `RuntimeWarning` if the taper widens any slice's second moment by
        more than this fraction. Set to `None` to silence the check.

    Returns
    -------
    `~numpy.ndarray`
        The tapered distribution, same shape as *f*.

    Warns
    -----
    `RuntimeWarning`
        If the taper has substantially widened the distribution -- see the warning
        about pedestals below.

    Notes
    -----
    Slices that are entirely non-positive, or whose signal already reaches both grid
    edges, are returned unchanged.

    .. warning::

       The default unbounded rolloff assumes the distribution roughly fills the
       velocity grid. When the signal occupies only a small part of the grid, the
       rolloff runs across a long stretch of empty axis, and the resulting pedestal
       sits at large :math:`|v|` where the :math:`v^2` weighting of the second
       moment amplifies it. For a Maxwellian at the default threshold, the recovered
       width grows by roughly 5% on a grid of half-width :math:`6\sigma`, 14% at
       :math:`8\sigma`, and 40% at :math:`12\sigma`. Since the forward model reads
       the thermal speed straight off the distribution, that propagates directly
       into :math:`\alpha` and the fitted temperature.

       Either size the velocity grid to the data, lower *threshold_frac*, or set
       *max_taper_bins* to bound the rolloff. The unbounded default is kept because
       it reproduces the ``osiris2thomson`` pipeline this module is derived from;
       *pedestal_warning* exists so the effect cannot pass unnoticed.

    In the degenerate case where the signal edge sits exactly one bin from the end of
    the rolloff, that bin takes the edge value rather than zero; this affects a
    single bin at the noise floor.
    """
    f = np.asarray(f, dtype=np.float64)
    moved = np.moveaxis(f, axis, 0)
    moved_shape = moved.shape
    n_v = moved_shape[0]
    cols = moved.reshape(n_v, -1)
    n_cols = cols.shape[1]

    limit = n_v if max_taper_bins is None else int(max_taper_bins)
    if limit < 0:
        raise ValueError(f"max_taper_bins must be non-negative; got {max_taper_bins}.")

    peak = cols.max(axis=0)
    mask = cols > threshold_frac * peak
    valid = mask.any(axis=0) & (peak > 0)

    # First and last bin above threshold, per column.
    i_left = np.argmax(mask, axis=0)
    i_right = n_v - 1 - np.argmax(mask[::-1, :], axis=0)

    pos = np.arange(n_v)[:, None]
    cidx = np.arange(n_cols)

    # Left tail: ramp 0 -> f[i_left] across [start_left, i_left), zero below.
    length_left = np.minimum(i_left, limit)
    start_left = i_left - length_left
    denom_left = np.where(length_left > 1, length_left - 1, 1)
    frac_left = np.clip((pos - start_left) / denom_left, 0.0, 1.0)
    left_values = cols[i_left, cidx] * np.sin(frac_left * (np.pi / 2))
    left_region = (pos >= start_left) & (pos < i_left) & valid

    # Right tail: ramp f[i_right] -> 0 across (i_right, end_right], zero above.
    length_right = np.minimum(n_v - 1 - i_right, limit)
    end_right = i_right + length_right
    denom_right = np.where(length_right > 1, length_right - 1, 1)
    frac_right = np.clip(1.0 - (pos - (i_right + 1)) / denom_right, 0.0, 1.0)
    right_values = cols[i_right, cidx] * np.sin(frac_right * (np.pi / 2))
    right_region = (pos > i_right) & (pos <= end_right) & valid

    out = np.where(left_region, left_values, cols)
    out = np.where(right_region, right_values, out)
    # Everything beyond a bounded rolloff is zeroed.
    out = np.where(((pos < start_left) | (pos > end_right)) & valid, 0.0, out)

    if pedestal_warning is not None and valid.any():
        before = _index_space_variance(np.clip(cols[:, valid], 0.0, None))
        after = _index_space_variance(np.clip(out[:, valid], 0.0, None))
        with np.errstate(invalid="ignore", divide="ignore"):
            growth = np.where(before > 0, after / before - 1.0, 0.0)
        worst = float(np.nanmax(growth)) if growth.size else 0.0
        if worst > pedestal_warning:
            warnings.warn(
                f"taper_vdf_edges: the taper widened the second moment of at "
                f"least one slice by {worst:.0%}. The distribution likely "
                f"occupies only a small part of the velocity grid, so the "
                f"rolloff is fabricating a pedestal at large |v|. Bound it "
                f"with max_taper_bins, lower threshold_frac, or narrow the "
                f"velocity grid.",
                RuntimeWarning,
                stacklevel=2,
            )

    return np.moveaxis(out.reshape(moved_shape), 0, axis)


def rescale_velocity_axis(
    f,
    v,
    scale: float,
    *,
    target_v=None,
    axis: int = VELOCITY_AXIS,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Compress a velocity axis by *scale* and resample onto a common grid.

    Divides the velocity axis by *scale* and interpolates *f* onto *target_v*,
    zero-padding outside the compressed domain. The distribution is multiplied by
    *scale* so that :math:`\int f \, dv` is preserved.

    The primary use is undoing a reduced ion-to-electron mass ratio. PIC runs that
    reduce :math:`m_i / m_e` by a factor :math:`R` stretch the velocity axis by
    :math:`\sqrt{R}` to keep Mach-number-like quantities invariant, so recovering
    physical velocities means passing ``scale = sqrt(R)`` for *every* species.

    Parameters
    ----------
    f : array_like
        Phase-space density.
    v : array_like
        Velocity axis, in m/s, matching ``f.shape[axis]``.
    scale : `float`
        Divisor for the velocity axis. Must be positive; ``1.0`` is a no-op
        resampling onto *target_v*.
    target_v : array_like, optional
        Velocity axis to resample onto, in m/s. Defaults to the original range with
        ``ceil(scale)`` times as many points, so the compressed distribution is not
        under-resolved.
    axis : `int`
        Axis of *f* holding velocity.
    fill_value : `float`
        Value assigned outside the compressed domain.

    Returns
    -------
    target_v : `~numpy.ndarray`
        The output velocity axis, in m/s.
    f_out : `~numpy.ndarray`
        *f* resampled onto *target_v*.
    """
    if scale <= 0:
        raise ValueError(f"scale must be positive; got {scale}.")

    f = np.asarray(f, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    v_scaled = v / scale

    if target_v is None:
        n_target = int(np.ceil(scale)) * v.size
        target_v = np.linspace(v[0], v[-1], n_target)
    else:
        target_v = np.asarray(target_v, dtype=np.float64)

    interpolate = interp1d(
        v_scaled,
        f,
        axis=axis,
        kind="linear",
        bounds_error=False,
        fill_value=fill_value,
    )
    return target_v, interpolate(target_v) * scale


def species_presence_mask(
    density,
    reference_density: float,
    threshold: float = 1e-2,
) -> np.ndarray:
    """
    Mark where a species is physically present.

    Below some density a species contributes nothing but noise, and including it
    fabricates a scattering contribution. Compute this from the *raw* density --
    before smoothing, tapering, or flooring -- so that the conditioning steps cannot
    manufacture presence.

    Parameters
    ----------
    density : array_like
        Species density, in any units consistent with *reference_density*.
    reference_density : `float`
        Density scale to compare against, typically the background plasma density.
    threshold : `float`
        Fraction of *reference_density* below which the species counts as absent.

    Returns
    -------
    `~numpy.ndarray`
        Boolean array, `True` where the species is present.
    """
    return np.asarray(density) > threshold * reference_density


def condition_phase_space(
    phase_space: PICPhaseSpace,
    *,
    smoothing_window: int = 40,
    smoothing_iterations: int = 0,
    taper_threshold: float | None = 0.005,
    max_taper_bins: int | None = None,
    pedestal_warning: float | None = 0.05,
    floor: float = DEFAULT_FLOOR,
    velocity_scale_factor: float | None = None,
    target_velocity_axis=None,
) -> PICPhaseSpace:
    r"""
    Prepare raw PIC phase space for the Thomson forward model.

    Applies, in order: `smooth_vdf`, `taper_vdf_edges`, `normalize_vdf`, a positive
    floor, and -- if requested -- `rescale_velocity_axis`. See the module docstring
    for why this order.

    Parameters
    ----------
    phase_space : `PICPhaseSpace`
        The raw phase space.
    smoothing_window : `int`
        Boxcar width in velocity bins, passed to `smooth_vdf`.
    smoothing_iterations : `int`
        Number of boxcar passes. Defaults to ``0`` -- no smoothing -- because the
        right amount differs between species: electron distributions usually need a
        few passes, while smoothing ions washes out the narrow IAW feature. Set it
        per species rather than relying on a default.
    taper_threshold : `float`, optional
        Passed to `taper_vdf_edges` as ``threshold_frac``. Set to `None` to skip
        tapering.
    max_taper_bins : `int`, optional
        Bounds the taper rolloff, passed to `taper_vdf_edges`. Worth setting when
        the distribution occupies only a small part of the velocity grid; see the
        notes on `taper_vdf_edges`.
    pedestal_warning : `float`, optional
        Passed to `taper_vdf_edges`; `None` silences its pedestal check.
    floor : `float`
        Positive floor applied after normalisation.
    velocity_scale_factor : `float`, optional
        Mass-ratio reduction factor :math:`R`. When given, all velocities are
        divided by :math:`\sqrt{R}` via `rescale_velocity_axis`. Omit -- the
        default -- to leave velocities as the simulation reports them.
    target_velocity_axis : array_like, optional
        Velocity grid to resample onto when *velocity_scale_factor* is given.

    Returns
    -------
    `PICPhaseSpace`
        A new phase space; the input is not modified.
    """
    f = smooth_vdf(
        phase_space.f,
        window=smoothing_window,
        iterations=smoothing_iterations,
    )
    if taper_threshold is not None:
        f = taper_vdf_edges(
            f,
            threshold_frac=taper_threshold,
            max_taper_bins=max_taper_bins,
            pedestal_warning=pedestal_warning,
        )
    f = normalize_vdf(f, phase_space.v)
    f = np.clip(f, a_min=floor, a_max=None)

    v = phase_space.v
    conditioning = {
        "smoothing_window": smoothing_window,
        "smoothing_iterations": smoothing_iterations,
        "taper_threshold": taper_threshold,
        "max_taper_bins": max_taper_bins,
        "floor": floor,
    }

    if velocity_scale_factor is not None:
        scale = float(np.sqrt(velocity_scale_factor))
        v, f = rescale_velocity_axis(f, v, scale=scale, target_v=target_velocity_axis)
        conditioning["velocity_scale_factor"] = velocity_scale_factor

    return replace(
        phase_space,
        f=f,
        v=v,
        meta={**phase_space.meta, "conditioning": conditioning},
    )


# ---------------------------------------------------------------------------
# Forward-model driver
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThomsonSpectrogram:
    r"""
    Synthetic Thomson spectra over time at one point in a PIC simulation.

    Returned by `spectra_from_phase_spaces`. All quantities are SI: wavelengths in
    m, times in s, densities in m\ :sup:`-3`.

    Parameters
    ----------
    epw, iaw : `~numpy.ndarray`
        Spectra over the electron-plasma-wave and ion-acoustic-wave windows, shape
        ``(n_time, n_wavelength)``. Timesteps with no plasma present are `~numpy.nan`
        throughout. ``iaw`` is `None` if no IAW window was requested.
    epw_wavelengths, iaw_wavelengths : `~numpy.ndarray`
        The wavelength axes, in m. ``iaw_wavelengths`` is `None` if not requested.
    t : `~numpy.ndarray`
        Times, shape ``(n_time,)``, in s.
    alpha_epw, alpha_iaw : `~numpy.ndarray`
        Scattering parameter reported by the forward model at each timestep.

        .. note::

           `~plasmapy.diagnostics.thomson.arbitrary_forwardmodel` defines this as
           :math:`\sqrt{2}\, \omega_{pe} / (k \sigma)`, with :math:`\sigma` the
           standard deviation of the electron distribution. That is
           :math:`\sqrt{2}` times the conventional :math:`1 / (k \lambda_{De})`
           reported by `~plasmapy.diagnostics.thomson.spectral_density`.

    electron_density : `~numpy.ndarray`
        Total electron density at the sampled point, shape ``(n_time,)``, in
        m\ :sup:`-3`.
    efract, ifract : `~numpy.ndarray`
        Fractional densities of each electron and ion population, shape
        ``(n_population, n_time)``. These are the raw fractions over *all*
        populations; the fractions actually handed to the forward model are these
        restricted to the present populations and renormalised to sum to one.
    electron_present, ion_present : `~numpy.ndarray`
        Boolean, shape ``(n_population, n_time)``: whether each population was
        included at each timestep.
    position : `float`
        Requested sampling position, in m.
    electron_labels, ion_labels : `list` of `str`
        Species identifiers, ordered as the ``efract`` / ``ifract`` rows.
    meta : `dict`
        Provenance: geometry, conditioning settings, and per-species metadata.

    Notes
    -----
    The forward model normalises each spectrum to unit area over its own window
    (``thomson.py``, ``Skw / trapezoid(Skw, wavelengths)``). Each row of ``epw`` and
    ``iaw`` therefore carries **shape only** -- there is no absolute intensity, no
    brightness history along the time axis, and no meaningful EPW-to-IAW ratio.
    """

    epw: np.ndarray
    epw_wavelengths: np.ndarray
    t: np.ndarray
    alpha_epw: np.ndarray
    electron_density: np.ndarray
    efract: np.ndarray
    ifract: np.ndarray
    electron_present: np.ndarray
    ion_present: np.ndarray
    position: float
    electron_labels: list[str]
    ion_labels: list[str]
    iaw: np.ndarray | None = None
    iaw_wavelengths: np.ndarray | None = None
    alpha_iaw: np.ndarray | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_time(self) -> int:
        """Number of timesteps."""
        return self.t.size

    def __repr__(self) -> str:
        """Summarise the species, sampling point, and spectrogram dimensions."""
        species = ", ".join([*self.electron_labels, *self.ion_labels])
        iaw = "none" if self.iaw is None else f"{self.iaw.shape[1]} bins"
        return (
            f"ThomsonSpectrogram(species=[{species}], "
            f"n_time={self.n_time}, epw={self.epw.shape[1]} bins, iaw={iaw})"
        )


def _as_phase_space_list(phase_spaces, name: str) -> list[PICPhaseSpace]:
    """Normalise a single `PICPhaseSpace` or an iterable of them into a list."""
    if isinstance(phase_spaces, PICPhaseSpace):
        return [phase_spaces]
    phase_spaces = list(phase_spaces)
    if not phase_spaces:
        raise ValueError(f"{name} must contain at least one PICPhaseSpace.")
    for phase_space in phase_spaces:
        if not isinstance(phase_space, PICPhaseSpace):
            raise TypeError(
                f"{name} must contain PICPhaseSpace objects; got "
                f"{type(phase_space).__name__}."
            )
    return phase_spaces


def _check_shared_time_axis(phase_spaces: list[PICPhaseSpace]) -> np.ndarray:
    """Return the common time axis, raising if the species disagree."""
    reference = phase_spaces[0]
    for phase_space in phase_spaces[1:]:
        if phase_space.t.size != reference.t.size:
            raise ValueError(
                f"species {phase_space.label!r} has {phase_space.t.size} timesteps "
                f"but {reference.label!r} has {reference.t.size}; all species must "
                "be sampled at the same times."
            )
        if not np.allclose(phase_space.t, reference.t, rtol=1e-6, atol=0):
            raise ValueError(
                f"species {phase_space.label!r} and {reference.label!r} have "
                "different time axes."
            )
    return reference.t


def _fractions(densities: np.ndarray) -> np.ndarray:
    """Row-wise fractions of a ``(n_population, n_time)`` density array."""
    total = densities.sum(axis=0)
    return np.divide(densities, total, out=np.zeros_like(densities), where=total > 0)


def _notches_to_quantity(notches):
    """Coerce notch specifications into an ``(n, 2)`` Quantity in nm."""
    if notches is None:
        return None
    notches = u.Quantity(notches, u.nm)
    if notches.ndim == 1:
        notches = notches.reshape(1, -1)
    if notches.ndim != 2 or notches.shape[1] != 2:
        raise ValueError(
            "notches must be a pair, or a sequence of pairs, of wavelengths; got "
            f"shape {tuple(notches.shape)}."
        )
    return notches


def spectra_from_phase_spaces(  # noqa: C901, PLR0915
    electrons,
    ions,
    *,
    position,
    reference_density,
    probe_wavelength,
    epw_wavelengths,
    iaw_wavelengths=None,
    epw_notches=None,
    iaw_notches=None,
    probe_vec=(1.0, 0.0, 0.0),
    scatter_vec=(0.0, 1.0, 0.0),
    electron_conditioning: dict[str, Any] | None = None,
    ion_conditioning: dict[str, Any] | None = None,
    velocity_scale_factor: float | None = None,
    presence_threshold: float = 1e-2,
    scattered_power: bool = True,
    progress: bool = True,
) -> ThomsonSpectrogram:
    r"""
    Forward-model Thomson spectra from PIC phase space, timestep by timestep.

    This is the code-agnostic core of the pipeline: it takes `PICPhaseSpace` objects
    from any reader, conditions them, reduces them to a single spatial point, and
    evaluates `~plasmapy.diagnostics.thomson.arbitrary_forwardmodel` over an EPW and
    an optional IAW wavelength window at every timestep.

    Parameters
    ----------
    electrons : `PICPhaseSpace` or `list` of `PICPhaseSpace`
        Electron population(s). Multiple populations -- for instance a piston and an
        ambient plasma -- are combined through the forward model's ``efract``.
    ions : `PICPhaseSpace` or `list` of `PICPhaseSpace`
        Ion population(s). Each must carry a `~plasmapy.particles` species label,
        since the forward model needs the charge and mass.
    position : `float` or `~astropy.units.Quantity`
        Where along the spatial axis to sample, in m if given as a bare float. Each
        species independently uses its own nearest grid point, so the species need
        not share a spatial grid.
    reference_density : `float` or `~astropy.units.Quantity`
        The physical density corresponding to a unit zeroth moment of the supplied
        phase space, in m\ :sup:`-3` if given as a bare float. For a PIC code that
        normalises density to a reference :math:`n_0`, this is :math:`n_0`. A reader
        that already produces ``f`` in SI should pass ``1 * u.m**-3``.
    probe_wavelength : `~astropy.units.Quantity`
        Probe laser wavelength.
    epw_wavelengths : `~astropy.units.Quantity`
        Wavelengths spanning the electron-plasma-wave feature.
    iaw_wavelengths : `~astropy.units.Quantity`, optional
        Wavelengths spanning the ion-acoustic-wave feature. Omit to skip the IAW
        calculation, which roughly halves the runtime.
    epw_notches, iaw_notches : `~astropy.units.Quantity`, optional
        Wavelength ranges to zero out, as a pair or a sequence of pairs -- a
        stray-light notch filter. Usually only the EPW window needs one, since the
        IAW window sits inside the notch.
    probe_vec, scatter_vec : array_like
        Unit vectors along the probe beam and towards the detector. The defaults
        give a 90 degree scattering geometry.
    electron_conditioning, ion_conditioning : `dict`, optional
        Keyword arguments for `condition_phase_space`, applied to every electron and
        every ion population respectively. The default, `None`, applies
        `condition_phase_space` with its own defaults, which do **no smoothing** --
        real PIC electron distributions usually want a few boxcar passes, and the
        right number is problem-specific, so set it here explicitly. Pass
        ``{"skip": True}`` if the phase spaces are already conditioned.
    velocity_scale_factor : `float`, optional
        Mass-ratio reduction factor :math:`R`, applied to every species. See
        `condition_phase_space`.
    presence_threshold : `float`
        A population contributes at a given timestep only if its fractional density
        exceeds this. Timesteps where the total electron density falls below
        ``presence_threshold * reference_density`` -- vacuum, essentially -- are
        recorded as `~numpy.nan` rather than given a fabricated spectrum.
    scattered_power : `bool`
        Convert :math:`S(k, \omega)` to scattered power per unit wavelength.
    progress : `bool`
        Show a progress bar over timesteps.

    Returns
    -------
    `ThomsonSpectrogram`

    Notes
    -----
    Population fractions are computed from the **raw** phase space, before
    smoothing, tapering, and flooring, so that conditioning cannot manufacture a
    population that is not physically there. When some populations are excluded at a
    timestep, the fractions of those remaining are renormalised to sum to one, as
    the forward model requires.

    Each returned spectrum is normalised to unit area over its own window; see the
    notes on `ThomsonSpectrogram`.

    Examples
    --------
    .. code-block:: python

       spectra = spectra_from_phase_spaces(
           electrons=electron_phase_space,
           ions=[aluminium_phase_space],
           position=80 * u.mm,
           reference_density=9e17 * u.cm**-3,
           probe_wavelength=532 * u.nm,
           epw_wavelengths=np.linspace(457, 607, 500) * u.nm,
           iaw_wavelengths=np.linspace(525, 539, 500) * u.nm,
           epw_notches=[525, 540] * u.nm,
           electron_conditioning={"smoothing_window": 40, "smoothing_iterations": 3},
           velocity_scale_factor=50,
       )
    """
    electrons = _as_phase_space_list(electrons, "electrons")
    ions = _as_phase_space_list(ions, "ions")
    all_species = [*electrons, *ions]
    t = _check_shared_time_axis(all_species)
    n_time = t.size

    for phase_space in electrons:
        if not phase_space.is_electron:
            warnings.warn(
                f"species {phase_space.label!r} was passed as an electron "
                "population but is not flagged as one.",
                RuntimeWarning,
                stacklevel=2,
            )

    position = float(_si_values(position, u.m))
    reference_density = float(_si_values(reference_density, u.m**-3))
    if reference_density <= 0:
        raise ValueError(
            f"reference_density must be positive; got {reference_density} m^-3."
        )

    # --- population densities and fractions, from the RAW phase space ---
    indices = {}
    densities = {}
    for phase_space in all_species:
        index, _ = phase_space.slice_position(position)
        indices[id(phase_space)] = index
        densities[id(phase_space)] = (
            number_density(phase_space.f, phase_space.v)[:, index] * reference_density
        )

    e_densities = np.stack([densities[id(ps)] for ps in electrons])
    i_densities = np.stack([densities[id(ps)] for ps in ions])
    electron_density = e_densities.sum(axis=0)
    efract = _fractions(e_densities)
    ifract = _fractions(i_densities)

    plasma_present = electron_density > presence_threshold * reference_density
    electron_present = (efract > presence_threshold) & plasma_present
    ion_present = (ifract > presence_threshold) & plasma_present

    # --- conditioning ---
    def _condition(phase_spaces, settings):
        settings = dict(settings or {})
        if settings.pop("skip", False):
            return list(phase_spaces)
        if "velocity_scale_factor" in settings:
            raise ValueError(
                "pass velocity_scale_factor to spectra_from_phase_spaces, not "
                "through the per-species conditioning settings; it must be the "
                "same for every species."
            )
        return [
            condition_phase_space(
                phase_space,
                velocity_scale_factor=velocity_scale_factor,
                **settings,
            )
            for phase_space in phase_spaces
        ]

    electrons_c = _condition(electrons, electron_conditioning)
    ions_c = _condition(ions, ion_conditioning)

    # --- wavelength windows ---
    epw_wavelengths = u.Quantity(epw_wavelengths, u.nm)
    epw_notches = _notches_to_quantity(epw_notches)
    want_iaw = iaw_wavelengths is not None
    if want_iaw:
        iaw_wavelengths = u.Quantity(iaw_wavelengths, u.nm)
        iaw_notches = _notches_to_quantity(iaw_notches)

    probe_vec = np.asarray(probe_vec, dtype=np.float64)
    scatter_vec = np.asarray(scatter_vec, dtype=np.float64)

    epw = np.full((n_time, epw_wavelengths.size), np.nan)
    alpha_epw = np.full(n_time, np.nan)
    # Always allocated, so the loop below stays simply typed; discarded at the
    # end when no IAW window was requested.
    iaw = np.full((n_time, iaw_wavelengths.size if want_iaw else 0), np.nan)
    alpha_iaw = np.full(n_time, np.nan)

    steps = range(n_time)
    if progress:
        steps = tqdm(steps, desc="Forward-modelling Thomson spectra")

    for step in steps:
        selected_e = np.nonzero(electron_present[:, step])[0]
        selected_i = np.nonzero(ion_present[:, step])[0]
        if selected_e.size == 0 or selected_i.size == 0:
            # Nothing to scatter off. Leave NaN rather than inventing a spectrum.
            continue

        # The forward model weights populations by these, so they must sum to one
        # over the populations actually passed.
        efract_step = efract[selected_e, step]
        efract_step = efract_step / efract_step.sum()
        ifract_step = ifract[selected_i, step]
        ifract_step = ifract_step / ifract_step.sum()

        e_axes = [electrons_c[k].v for k in selected_e] * u.m / u.s
        i_axes = [ions_c[k].v for k in selected_i] * u.m / u.s
        efn = (
            [electrons_c[k].f[step, :, indices[id(electrons[k])]] for k in selected_e]
            * u.s
            / u.m
        )
        ifn = (
            [ions_c[k].f[step, :, indices[id(ions[k])]] for k in selected_i] * u.s / u.m
        )
        ion_labels_step = [ions[k].label for k in selected_i]

        common = {
            "probe_wavelength": probe_wavelength,
            "e_velocity_axes": e_axes,
            "i_velocity_axes": i_axes,
            "efn": efn,
            "ifn": ifn,
            "efract": efract_step,
            "ifract": ifract_step,
            "n": electron_density[step] * u.m**-3,
            "ion_species": ion_labels_step,
            "probe_vec": probe_vec,
            "scatter_vec": scatter_vec,
            "scattered_power": scattered_power,
        }

        alpha, spectrum = arbitrary_forwardmodel(
            wavelengths=epw_wavelengths, notches=epw_notches, **common
        )
        epw[step] = np.asarray(spectrum, dtype=np.float64)
        alpha_epw[step] = float(np.asarray(alpha))

        if want_iaw:
            alpha, spectrum = arbitrary_forwardmodel(
                wavelengths=iaw_wavelengths, notches=iaw_notches, **common
            )
            iaw[step] = np.asarray(spectrum, dtype=np.float64)
            alpha_iaw[step] = float(np.asarray(alpha))

    return ThomsonSpectrogram(
        epw=epw,
        epw_wavelengths=epw_wavelengths.to_value(u.m),
        iaw=iaw if want_iaw else None,
        iaw_wavelengths=iaw_wavelengths.to_value(u.m) if want_iaw else None,
        t=t,
        alpha_epw=alpha_epw,
        alpha_iaw=alpha_iaw if want_iaw else None,
        electron_density=electron_density,
        efract=efract,
        ifract=ifract,
        electron_present=electron_present,
        ion_present=ion_present,
        position=position,
        electron_labels=[ps.label for ps in electrons],
        ion_labels=[ps.label for ps in ions],
        meta={
            "reference_density": reference_density,
            "probe_vec": probe_vec.tolist(),
            "scatter_vec": scatter_vec.tolist(),
            "presence_threshold": presence_threshold,
            "velocity_scale_factor": velocity_scale_factor,
            "scattered_power": scattered_power,
            "species_meta": {ps.label: ps.meta for ps in [*electrons_c, *ions_c]},
        },
    )
