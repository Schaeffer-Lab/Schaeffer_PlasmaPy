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
    "condition_phase_space",
    "from_arrays",
    "normalize_vdf",
    "number_density",
    "rescale_velocity_axis",
    "smooth_vdf",
    "species_presence_mask",
    "taper_vdf_edges",
]

import warnings
from dataclasses import dataclass, field, replace
from typing import Any

import astropy.units as u
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import uniform_filter1d

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
