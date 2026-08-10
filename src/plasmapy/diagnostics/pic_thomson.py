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
    "from_moments",
    "histogram2d_deck_block",
    "normalize_vdf",
    "number_density",
    "read_openpmd_phase_space",
    "read_osiris_phase_space",
    "read_warpx_hybrid_electrons",
    "read_warpx_phase_space",
    "rescale_velocity_axis",
    "smooth_vdf",
    "species_presence_mask",
    "spectra_from_phase_spaces",
    "taper_vdf_edges",
]

import json
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import astropy.constants as const
import astropy.units as u
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter, uniform_filter, uniform_filter1d
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

#: Fraction of the in-volume weight that may fall outside the histogram bounds
#: before a reader complains. A handful of runaway particles is normal; a
#: percent of the distribution is a clipped tail.
_CLIP_WARNING_FRACTION = 1e-3

#: How far a reader's zeroth moment may drift from the density the PIC code
#: itself deposited before the reader complains.
_DENSITY_CHECK_TOLERANCE = 0.05

#: How much of a reconstructed distribution may fall off its velocity grid
#: before `from_moments` complains.
_MOMENT_RECOVERY_TOLERANCE = 0.01


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

    def at_position(self, position: float) -> "PICPhaseSpace":
        """
        Return a copy reduced to the single grid point nearest *position*.

        Conditioning acts independently on each ``(time, position)`` slice, so
        reducing before conditioning gives identical results to reducing after,
        at a fraction of the memory. That matters: a velocity rescale oversamples
        the velocity axis, and carrying every unused spatial cell through it can
        turn a few gigabytes into a hundred.

        Parameters
        ----------
        position : `float` or `~astropy.units.Quantity`
            Position along `x`, in m if given as a bare float.

        Returns
        -------
        `PICPhaseSpace`
            A new phase space with ``n_x == 1``.
        """
        index, f_slice = self.slice_position(position)
        return replace(
            self,
            f=np.ascontiguousarray(f_slice[:, :, np.newaxis]),
            x=self.x[index : index + 1],
            meta={**self.meta, "sampled_index": index, "sampled_position": position},
        )

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


def _thermal_energy(temperature) -> np.ndarray:
    r"""
    Return :math:`k_B T` in joules, from a temperature in K, J, or eV.

    Bare arrays are read as kelvin, following the rest of PlasmaPy; the fluid
    temperatures that hybrid codes write are usually in eV, so pass those as a
    `~astropy.units.Quantity` and let the units carry the conversion.
    """
    if isinstance(temperature, u.Quantity):
        if temperature.unit.is_equivalent(u.K):
            return np.asarray((const.k_B * temperature).to(u.J).value, dtype=np.float64)
        if temperature.unit.is_equivalent(u.J):
            return np.asarray(temperature.to(u.J).value, dtype=np.float64)
        raise u.UnitConversionError(
            f"temperature must be in K or in energy units; got {temperature.unit}."
        )
    return const.k_B.si.value * np.asarray(temperature, dtype=np.float64)


def from_moments(
    density,
    temperature,
    drift=None,
    *,
    t,
    x,
    label: str,
    v=None,
    mass=None,
    is_electron: bool | None = None,
    n_velocity_bins: int = 512,
    velocity_headroom: float = 6.0,
    meta: dict[str, Any] | None = None,
) -> PICPhaseSpace:
    r"""
    Build a `PICPhaseSpace` from a drifting Maxwellian's moments.

    Not every code carries every species as macroparticles. A kinetic-ion /
    fluid-electron (hybrid) code has no electron macroparticles at all: its
    electrons are a quasineutral, inertialess fluid described entirely by a
    density, a temperature, and a drift. This reconstructs the distribution
    those moments imply,

    .. math::

       f(v) = n \left(\frac{m}{2 \pi k_B T}\right)^{1/2}
              \exp\!\left[-\frac{m (v - u)^2}{2 k_B T}\right],

    so that a fluid species can be fed to the same forward model as a kinetic
    one.

    Parameters
    ----------
    density : array_like or `~astropy.units.Quantity`
        Number density, shape ``(n_time, n_x)``, in m\ :sup:`-3` if bare.
    temperature : array_like or `~astropy.units.Quantity`
        Temperature, shape ``(n_time, n_x)``, in K if bare. Quantities may be in
        K or in energy units, so an eV field can be passed straight through.
    drift : array_like or `~astropy.units.Quantity`, optional
        Fluid velocity **along the diagnostic axis**, shape ``(n_time, n_x)``, in
        m/s if bare. Defaults to zero. Project a vector drift onto
        :math:`\hat{k}` before passing it.
    t, x : array_like or `~astropy.units.Quantity`
        Time and position axes, shapes ``(n_time,)`` and ``(n_x,)``.
    label : `str`
        Species identifier, for example ``"e-"``.
    v : array_like or `~astropy.units.Quantity`, optional
        Velocity axis. The default builds one spanning the drift plus
        *velocity_headroom* thermal speeds either side, which is what the
        moments themselves imply; pass one explicitly to share a grid between
        species.
    mass : `float` or `~astropy.units.Quantity`, optional
        Mass setting the thermal width, in kg if bare. Defaults to the physical
        mass of *label*. For a fluid electron population this is the **real**
        electron mass even though the solver treats the fluid as inertialess:
        the inertia that is dropped is the electrons' response to the fields,
        not the mass that sets their thermal spread at a given temperature.
    is_electron : `bool`, optional
        Whether this is an electron population. Inferred from *label* if omitted.
    n_velocity_bins : `int`
        Resolution of the default velocity axis.
    velocity_headroom : `float`
        How many thermal speeds past the extreme drift the default axis reaches.
    meta : `dict`, optional
        Extra provenance to attach.

    Returns
    -------
    `PICPhaseSpace`
        With ``f`` in SI, so `spectra_from_phase_spaces` needs no density scale.

    Raises
    ------
    `ValueError`
        If the moment arrays disagree in shape, or a temperature is not positive
        where the density is.

    Notes
    -----
    A Maxwellian is not an assumption imposed here so much as one inherited: a
    fluid closure carries a single scalar temperature and no higher moments, so
    there is nothing in the data from which to build any other shape. What that
    costs is a real question about the model, not about this function --
    reconstruct the moments of an equivalent kinetic run and compare the two
    spectra to answer it.

    Condition the result with ``taper_threshold=None``. The taper exists to
    replace the discontinuity where macroparticle shot noise meets the edge of
    the velocity grid; a reconstructed distribution has neither, so tapering it
    only fabricates a pedestal in the wings and widens the feature.

    Examples
    --------
    .. code-block:: python

       electrons = from_moments(
           density=n_e,  # (n_time, n_x), m^-3
           temperature=T_e * u.eV,
           drift=u_e,  # m/s along k-hat
           t=times,
           x=positions,
           label="e-",
       )
    """
    density = np.atleast_2d(_si_values(density, u.m**-3))
    thermal_energy = np.atleast_2d(_thermal_energy(temperature))
    if drift is None:
        drift = np.zeros_like(density)
    drift = np.atleast_2d(_si_values(drift, u.m / u.s))

    shapes = {
        "density": density.shape,
        "temperature": thermal_energy.shape,
        "drift": drift.shape,
    }
    if len(set(shapes.values())) != 1:
        raise ValueError(f"the moments must share a shape (n_time, n_x); got {shapes}.")
    if density.ndim != 2:
        raise ValueError(
            f"the moments must be 2-D with shape (n_time, n_x); got ndim="
            f"{density.ndim}."
        )

    present = density > 0
    if np.any(present & ~(thermal_energy > 0)):
        raise ValueError(
            "temperature must be positive wherever the density is: a Maxwellian "
            "has no zero-temperature limit on a finite velocity grid. Floor the "
            "temperature, or zero the density where the species is absent."
        )

    mass_si = (
        float(Particle(label).mass.to(u.kg).value)
        if mass is None
        else float(_si_values(mass, u.kg))
    )
    if mass_si <= 0:
        raise ValueError(f"mass must be positive; got {mass_si} kg.")

    # sqrt(kT/m), the width of the Maxwellian in this one velocity component.
    thermal_speed = np.sqrt(np.where(present, thermal_energy, 0.0) / mass_si)

    if v is None:
        if not np.any(present):
            raise ValueError(
                f"species {label!r} has zero density everywhere, so no velocity "
                "axis could be inferred; pass v explicitly."
            )
        reach = float(
            np.max(np.abs(drift[present]) + velocity_headroom * thermal_speed[present])
        )
        reach = min(reach, 0.999 * const.c.si.value)
        v_axis = np.linspace(-reach, reach, n_velocity_bins)
    else:
        v_axis = _si_values(v, u.m / u.s)

    # (n_time, n_v, n_x): the velocity axis is inserted between the two the
    # moments already have.
    offset = v_axis[np.newaxis, :, np.newaxis] - drift[:, np.newaxis, :]
    width = thermal_speed[:, np.newaxis, :]
    amplitude = np.divide(
        density[:, np.newaxis, :],
        width * np.sqrt(2.0 * np.pi),
        out=np.zeros_like(offset + density[:, np.newaxis, :]),
        where=width > 0,
    )
    exponent = np.divide(
        -0.5 * offset**2,
        np.where(width > 0, width, 1.0) ** 2,
        out=np.zeros_like(offset),
        where=width > 0,
    )
    f = amplitude * np.exp(exponent)

    # A Maxwellian integrates to the density analytically; on a finite grid it
    # only does so if the grid actually contains it. Say so when it does not,
    # rather than handing back a distribution that quietly holds less plasma
    # than the moments describe.
    recovered = number_density(f, v_axis)
    shortfall = np.divide(
        np.abs(recovered - density),
        density,
        out=np.zeros_like(density),
        where=present,
    )
    worst = float(np.max(shortfall)) if shortfall.size else 0.0
    if worst > _MOMENT_RECOVERY_TOLERANCE:
        warnings.warn(
            f"the velocity axis [{v_axis[0]:.3e}, {v_axis[-1]:.3e}] m/s does not "
            f"contain the reconstructed {label!r} distribution: its zeroth moment "
            f"is off by up to {100 * worst:.2f}% of the density it was built "
            "from. Widen v, or leave it unset so it is sized from the moments.",
            RuntimeWarning,
            stacklevel=2,
        )

    return from_arrays(
        f=f,
        v=v_axis,
        x=x,
        t=t,
        label=label,
        is_electron=is_electron,
        meta={
            "source": "moments",
            "distribution": "drifting Maxwellian",
            "mass": mass_si,
            # f is already a density in m^-3.
            "reference_density": 1.0,
            "moment_recovery_error": worst,
            **dict(meta or {}),
        },
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

    # Boxcar smoothing of a strictly non-negative distribution can leave
    # negatives of order 1e-20 against a 1e-5 peak, which are round-off, not
    # shot noise. Warn only about negatives large enough to be physical.
    scale = float(np.max(np.abs(f))) if f.size else 0.0
    n_negative = int(np.count_nonzero(f < -1e-12 * scale))
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
        Emit a `RuntimeWarning` if the taper widens the second moment of any
        appreciably populated slice by more than this fraction. Slices carrying
        less than 1% of the peak slice's weight are excluded, since an almost-empty
        cell has a near-zero width that any taper multiplies enormously. Set to
        `None` to silence the check.

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

    # Judge the pedestal only on columns that carry appreciable weight: an
    # almost-empty cell in vacuum has a near-zero variance to begin with, so any
    # taper multiplies it enormously and would drown the check in false alarms.
    weight = np.clip(cols, 0.0, None).sum(axis=0)
    significant = valid & (weight > 0.01 * weight.max()) if weight.size else valid

    if pedestal_warning is not None and significant.any():
        before = _index_space_variance(np.clip(cols[:, significant], 0.0, None))
        after = _index_space_variance(np.clip(out[:, significant], 0.0, None))
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


def _distinct(labels: list[str]) -> list[str]:
    """
    Number repeated labels, for legends.

    Two populations of the same species are perfectly legitimate -- an ambient
    and a piston plasma of the same element, say -- but plotting two lines with
    the same name is not.
    """
    if len(set(labels)) == len(labels):
        return list(labels)
    seen: dict[str, int] = {}
    out = []
    for label in labels:
        seen[label] = seen.get(label, 0) + 1
        out.append(f"{label} #{seen[label]}")
    return out


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

    def apply_instrument_response(
        self,
        *,
        time_fwhm=None,
        epw_wavelength_fwhm=None,
        iaw_wavelength_fwhm=None,
        kernel: str = "gaussian",
    ) -> "ThomsonSpectrogram":
        r"""
        Blur the spectrogram to the resolution of a real instrument.

        A streak camera integrates over a finite gate and a spectrometer over a
        finite slit width, so a synthetic spectrogram compared against measured
        data has to be degraded to match. Widths are given as the full width at
        half maximum of the instrument function, in physical units, and
        converted to bins from the spectrogram's own axes.

        Parameters
        ----------
        time_fwhm : `~astropy.units.Quantity`, optional
            Temporal resolution, in s if given as a bare float. A streak camera
            might be 100 ps.
        epw_wavelength_fwhm, iaw_wavelength_fwhm : `~astropy.units.Quantity`, optional
            Spectral resolution of each window, in m if given as a bare float.
            The two windows usually have very different dispersion, so they take
            separate values.
        kernel : ``'gaussian'`` or ``'boxcar'``
            Shape of the instrument function.

        Returns
        -------
        `ThomsonSpectrogram`
            A new spectrogram; the original is unchanged.

        Notes
        -----
        Timesteps with no plasma are `~numpy.nan`. They are left out of the
        average rather than spread over their neighbours, so a gap cannot
        destroy the data around it. A gap narrower than the instrument function
        is filled from the valid data either side, as a real instrument
        integrating over a finite gate would; a gap wider than the kernel's
        reach stays `~numpy.nan`.

        Each row of the input was normalised to unit area over its own window by
        the forward model. Smoothing in time mixes rows and does not preserve
        that, which is correct -- a real instrument integrates counts, it does
        not renormalise each gate.

        Examples
        --------
        .. code-block:: python

           degraded = spectra.apply_instrument_response(
               time_fwhm=100 * u.ps,
               epw_wavelength_fwhm=0.5 * u.nm,
               iaw_wavelength_fwhm=0.05 * u.nm,
           )
        """
        time_sigma = _sigma_in_bins(time_fwhm, self.t, u.s)

        epw = _smooth_preserving_gaps(
            self.epw,
            (
                time_sigma,
                _sigma_in_bins(epw_wavelength_fwhm, self.epw_wavelengths, u.m),
            ),
            kernel,
        )
        iaw = self.iaw
        if iaw is not None and self.iaw_wavelengths is not None:
            iaw = _smooth_preserving_gaps(
                iaw,
                (
                    time_sigma,
                    _sigma_in_bins(iaw_wavelength_fwhm, self.iaw_wavelengths, u.m),
                ),
                kernel,
            )

        return replace(
            self,
            epw=epw,
            iaw=iaw,
            meta={
                **self.meta,
                "instrument_response": {
                    "time_fwhm_s": None
                    if time_fwhm is None
                    else float(_si_values(time_fwhm, u.s)),
                    "epw_wavelength_fwhm_m": None
                    if epw_wavelength_fwhm is None
                    else float(_si_values(epw_wavelength_fwhm, u.m)),
                    "iaw_wavelength_fwhm_m": None
                    if iaw_wavelength_fwhm is None
                    else float(_si_values(iaw_wavelength_fwhm, u.m)),
                    "kernel": kernel,
                },
            },
        )

    def to_hdf5(self, path) -> None:
        r"""
        Write the spectrogram to a self-describing HDF5 file.

        Every dataset carries a ``UNITS`` attribute and everything is SI, so the
        file can be read without knowing how it was produced. `from_hdf5` reads
        it back.

        Parameters
        ----------
        path : path-like
            Destination file, overwritten if it exists.

        Notes
        -----
        The group names follow the ``osiris2thomson`` layout where the contents
        line up -- ``SPECTRA/EPW/epws``, ``SPECTRA/IAW/iaws``,
        ``SPECTRA/SCATTERING_PARAMETERS``, ``DENSITY/dens``, ``AXES`` -- but this
        is **not** a drop-in replacement for files that pipeline wrote. It has no
        ``TEMPERATURE``, ``FLOW_VELOCITY``, ``PERPENDICULAR_MAGNETIC_FIELD`` or
        ``VDF`` groups, since this module does not compute those, and its axes
        are in SI rather than in simulation units.
        """
        path = Path(path)
        with h5py.File(path, "w") as handle:
            spectra = handle.create_group("SPECTRA")

            epw = spectra.create_group("EPW")
            epw.create_dataset("epws", data=self.epw)
            axis = epw.create_dataset("wavelengths", data=self.epw_wavelengths)
            axis.attrs["UNITS"] = "m"
            if self.iaw is not None:
                iaw = spectra.create_group("IAW")
                iaw.create_dataset("iaws", data=self.iaw)
                axis = iaw.create_dataset("wavelengths", data=self.iaw_wavelengths)
                axis.attrs["UNITS"] = "m"

            alpha = spectra.create_group("SCATTERING_PARAMETERS")
            entry = alpha.create_dataset("alpha_epw", data=self.alpha_epw)
            entry.attrs["CONVENTION"] = (
                "sqrt(2) * wpe / (k * sigma), i.e. sqrt(2) times 1 / (k * lambda_De)"
            )
            if self.alpha_iaw is not None:
                alpha.create_dataset("alpha_iaw", data=self.alpha_iaw)

            density = handle.create_group("DENSITY")
            entry = density.create_dataset("dens", data=self.electron_density)
            entry.attrs["UNITS"] = "m^-3"

            populations = handle.create_group("POPULATIONS")
            populations.create_dataset("efract", data=self.efract)
            populations.create_dataset("ifract", data=self.ifract)
            populations.create_dataset("electron_present", data=self.electron_present)
            populations.create_dataset("ion_present", data=self.ion_present)
            populations.create_dataset(
                "electron_labels",
                data=np.array(self.electron_labels, dtype=h5py.string_dtype()),
            )
            populations.create_dataset(
                "ion_labels",
                data=np.array(self.ion_labels, dtype=h5py.string_dtype()),
            )

            axes = handle.create_group("AXES")
            times = axes.create_group("TIME_AXES")
            entry = times.create_dataset("time", data=self.t)
            entry.attrs["UNITS"] = "s"
            wavelengths = axes.create_group("WAVELENGTH_AXES")
            entry = wavelengths.create_dataset(
                "epw_wavelengths", data=self.epw_wavelengths
            )
            entry.attrs["UNITS"] = "m"
            if self.iaw_wavelengths is not None:
                entry = wavelengths.create_dataset(
                    "iaw_wavelengths", data=self.iaw_wavelengths
                )
                entry.attrs["UNITS"] = "m"

            handle.attrs["POSITION"] = self.position
            handle.attrs["POSITION_UNITS"] = "m"
            handle.attrs["NORMALISATION"] = (
                "each row is normalised to unit area over its own window, so the "
                "spectra carry shape only -- no absolute intensity and no "
                "meaningful EPW-to-IAW ratio"
            )
            handle.attrs["META"] = json.dumps(self.meta, default=str)

    @classmethod
    def from_hdf5(cls, path) -> "ThomsonSpectrogram":
        """
        Read a spectrogram written by `to_hdf5`.

        Parameters
        ----------
        path : path-like
            File to read.

        Returns
        -------
        `ThomsonSpectrogram`
        """

        def decode(values):
            return [
                value.decode() if isinstance(value, bytes) else str(value)
                for value in values
            ]

        with h5py.File(Path(path), "r") as handle:
            spectra = handle["SPECTRA"]
            populations = handle["POPULATIONS"]
            has_iaw = "IAW" in spectra
            return cls(
                epw=spectra["EPW/epws"][()],
                epw_wavelengths=spectra["EPW/wavelengths"][()],
                iaw=spectra["IAW/iaws"][()] if has_iaw else None,
                iaw_wavelengths=spectra["IAW/wavelengths"][()] if has_iaw else None,
                t=handle["AXES/TIME_AXES/time"][()],
                alpha_epw=spectra["SCATTERING_PARAMETERS/alpha_epw"][()],
                alpha_iaw=(
                    spectra["SCATTERING_PARAMETERS/alpha_iaw"][()] if has_iaw else None
                ),
                electron_density=handle["DENSITY/dens"][()],
                efract=populations["efract"][()],
                ifract=populations["ifract"][()],
                electron_present=populations["electron_present"][()],
                ion_present=populations["ion_present"][()],
                position=float(handle.attrs["POSITION"]),
                electron_labels=decode(populations["electron_labels"][()]),
                ion_labels=decode(populations["ion_labels"][()]),
                meta=json.loads(handle.attrs.get("META", "{}")),
            )

    def plot(self, *, figsize=None, save=None):
        """
        Draw the spectrogram alongside the quantities needed to read it.

        Parameters
        ----------
        figsize : pair of `float`, optional
            Figure size in inches.
        save : path-like, optional
            Write the figure here as well as returning it.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
        axes : `~numpy.ndarray` of `~matplotlib.axes.Axes`

        Notes
        -----
        The colour scale is clipped at the 99th percentile, because a single
        timestep near an electron-plasma-wave resonance can otherwise dominate
        the whole spectrogram.
        """
        windows = [("EPW", self.epw, self.epw_wavelengths)]
        if self.iaw is not None and self.iaw_wavelengths is not None:
            windows.append(("IAW", self.iaw, self.iaw_wavelengths))

        columns = max(len(windows), 2)
        figure, axes = plt.subplots(
            2, columns, figsize=figsize or (5.6 * columns, 8), squeeze=False
        )
        time_ns = self.t * 1e9

        for column, (name, data, axis) in enumerate(windows):
            finite = data[np.isfinite(data).all(axis=1)]
            image = axes[0][column].imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=[time_ns[0], time_ns[-1], axis[0] * 1e9, axis[-1] * 1e9],
                cmap="inferno",
                vmax=np.percentile(finite, 99) if finite.size else None,
            )
            figure.colorbar(image, ax=axes[0][column], label="area-normalised")
            axes[0][column].set_title(
                f"{name} spectrogram at x = {self.position * 1e3:.2f} mm", fontsize=10
            )
            axes[0][column].set_xlabel("time (ns)")
            axes[0][column].set_ylabel("wavelength (nm)")
        for column in range(len(windows), columns):
            axes[0][column].axis("off")

        axes[1][0].semilogy(time_ns, self.electron_density * 1e-6, lw=1.3)
        axes[1][0].set_xlabel("time (ns)")
        axes[1][0].set_ylabel(r"$n_e$ (cm$^{-3}$)")
        axes[1][0].set_title("electron density", fontsize=10)

        axes[1][1].plot(time_ns, self.alpha_epw, lw=1.3, label=r"$\alpha$")
        for row, label in enumerate(_distinct(self.ion_labels)):
            axes[1][1].plot(
                time_ns, self.ifract[row], lw=1.0, ls="--", label=f"ifract {label}"
            )
        if len(self.electron_labels) > 1:
            for row, label in enumerate(_distinct(self.electron_labels)):
                axes[1][1].plot(
                    time_ns, self.efract[row], lw=1.0, ls=":", label=f"efract {label}"
                )
        axes[1][1].set_xlabel("time (ns)")
        axes[1][1].set_title("scattering parameter and populations", fontsize=10)
        axes[1][1].legend(fontsize=7)
        for column in range(2, columns):
            axes[1][column].axis("off")

        figure.tight_layout()
        if save is not None:
            figure.savefig(save, dpi=140, bbox_inches="tight")
        return figure, axes

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


def _resolve_reference_density(phase_spaces: list[PICPhaseSpace], supplied) -> float:
    r"""
    Settle the density scale of the supplied phase spaces.

    An explicit *supplied* value always wins. Otherwise it comes from the
    ``reference_density`` every reader records in `PICPhaseSpace.meta`, which is
    :math:`n_0` for a code that normalises to a reference density and
    ``1`` m\ :sup:`-3` for a reader that already produces SI. Species must agree,
    since one scale multiplies all of them.
    """
    if supplied is None:
        recorded = [ps.meta.get("reference_density") for ps in phase_spaces]
        if any(value is None for value in recorded):
            missing = [
                ps.label
                for ps, value in zip(phase_spaces, recorded, strict=True)
                if value is None
            ]
            raise ValueError(
                "reference_density was not given and could not be taken from "
                f"the phase spaces: {missing} carry no 'reference_density' in "
                "their meta. Every reader in this module records one; pass it "
                "explicitly for phase spaces built by hand."
            )
        values = [float(_si_values(value, u.m**-3)) for value in recorded]
        if not np.allclose(values, values[0], rtol=1e-9):
            pairs = {
                ps.label: value for ps, value in zip(phase_spaces, values, strict=True)
            }
            raise ValueError(
                "the supplied phase spaces disagree about the density scale: "
                f"{pairs} m^-3. One scale multiplies all of them, so they must "
                "match; pass reference_density explicitly to override."
            )
        supplied = values[0]

    reference_density = float(_si_values(supplied, u.m**-3))
    if reference_density <= 0:
        raise ValueError(
            f"reference_density must be positive; got {reference_density} m^-3."
        )
    return reference_density


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
    reference_density=None,
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
    reference_density : `float` or `~astropy.units.Quantity`, optional
        The physical density corresponding to a unit zeroth moment of the supplied
        phase space, in m\ :sup:`-3` if given as a bare float. For a PIC code that
        normalises density to a reference :math:`n_0`, this is :math:`n_0`; for a
        reader that already produces ``f`` in SI it is ``1 * u.m**-3``. Every
        reader in this module records the right value in
        `PICPhaseSpace.meta`, so the default -- `None` -- takes it from there and
        raises if the species disagree. Pass it explicitly to override, or for
        phase spaces built by hand.
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
    reference_density = _resolve_reference_density(all_species, reference_density)

    # Reduce every species to the sampled point first. Conditioning acts
    # independently on each (time, position) slice, so this is equivalent to
    # conditioning the whole block and slicing afterwards -- but it avoids
    # carrying every unused spatial cell through a velocity rescale that
    # oversamples the velocity axis.
    sampled = {id(ps): ps.at_position(position) for ps in all_species}

    # --- population densities and fractions, from the RAW phase space ---
    densities = {
        key: number_density(ps.f, ps.v)[:, 0] * reference_density
        for key, ps in sampled.items()
    }

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

    electrons_c = _condition(
        [sampled[id(ps)] for ps in electrons], electron_conditioning
    )
    ions_c = _condition([sampled[id(ps)] for ps in ions], ion_conditioning)

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
        efn = [electrons_c[k].f[step, :, 0] for k in selected_e] * u.s / u.m
        ifn = [ions_c[k].f[step, :, 0] for k in selected_i] * u.s / u.m
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


# ---------------------------------------------------------------------------
# Readers -- the only code-specific part of the pipeline
# ---------------------------------------------------------------------------


def _attr_text(node, name: str) -> str:
    """Read an HDF5 attribute that OSIRIS writes as a byte string array."""
    value = node.attrs[name]
    value = np.atleast_1d(value)[0]
    return value.decode() if isinstance(value, bytes) else str(value)


def _plasma_frequency(reference_density: float) -> float:
    r"""Electron plasma frequency in rad/s for a density in m\ :sup:`-3`."""
    return float(
        np.sqrt(
            reference_density
            * const.e.si.value**2
            / (const.eps0.si.value * const.m_e.si.value)
        )
    )


def _osiris_axis_layout(handle, ndim: int) -> list[dict[str, Any]]:
    """
    Describe an OSIRIS file's axes in numpy order.

    OSIRIS writes ``AXIS/AXIS1`` as the fastest-varying (Fortran-order) axis, so
    numpy axis ``k`` corresponds to ``AXIS{ndim - k}``.
    """
    layout = []
    for k in range(ndim):
        node = handle["AXIS"][f"AXIS{ndim - k}"]
        layout.append(
            {
                "numpy_axis": k,
                "name": _attr_text(node, "NAME"),
                "min": float(node[0]),
                "max": float(node[1]),
            }
        )
    return layout


def _slab_indices(values: np.ndarray, centre: float, half_width) -> tuple[int, int]:
    """
    Index range of *values* lying within ``centre ± half_width``.

    Falls back to the single nearest cell when *half_width* is `None`, is
    non-positive, or is too narrow to contain any cell, so a slab always selects
    something rather than silently emptying the data.
    """
    if half_width is not None and half_width > 0:
        inside = np.nonzero(np.abs(values - centre) <= half_width)[0]
        if inside.size:
            return int(inside[0]), int(inside[-1]) + 1
    index = int(np.argmin(np.abs(values - centre)))
    return index, index + 1


def _transverse_selection(
    axes: list[dict[str, Any]],
    positions,
    reduction: str,
    half_width,
) -> dict[int, tuple[int, int]]:
    r"""
    Index range to keep along each transverse spatial axis.

    ``"slab"`` keeps a localized region about the requested position, which is
    what a Thomson scattering volume is. ``"chord"`` keeps the whole extent, for
    a measurement integrated along that direction.

    In both cases the caller *averages* over the selected cells rather than
    summing. Summing would turn a density into a line integral and quietly
    multiply the density handed to the forward model by the number of cells --
    which is what the ``osiris2thomson`` pipeline did for its ``p1x1x2``
    diagnostics.
    """
    if reduction not in {"slab", "chord"}:
        raise ValueError(
            f"transverse_reduction must be 'slab' or 'chord'; got {reduction!r}."
        )

    positions = dict(positions or {})
    unknown = set(positions) - {axis["name"] for axis in axes}
    if unknown:
        raise ValueError(
            f"transverse_position names {sorted(unknown)}, which are not "
            f"transverse axes; available: {[a['name'] for a in axes]}."
        )

    selection = {}
    for axis in axes:
        values = axis["values"]
        if reduction == "chord":
            selection[axis["numpy_axis"]] = (0, values.size)
            continue
        centre = positions.get(axis["name"])
        if centre is None:
            centre = 0.5 * (float(values[0]) + float(values[-1]))
        selection[axis["numpy_axis"]] = _slab_indices(values, float(centre), half_width)
    return selection


def _osiris_dump_index(filename: str) -> int:
    """Extract the six-digit dump index from an OSIRIS output filename."""
    return int(filename.rsplit("-", 1)[1].split(".", 1)[0])


def read_osiris_phase_space(  # noqa: C901, PLR0912, PLR0915
    path,
    field: str,
    species: str,
    *,
    reference_density,
    label: str | None = None,
    is_electron: bool | None = None,
    timesteps=None,
    spatial_axis: str | None = None,
    transverse_position=None,
    transverse_reduction: str = "slab",
    slab_halfwidth=None,
    position=None,
    charge_weighted: bool = True,
    relativistic_jacobian: bool = True,
    progress: bool = True,
) -> PICPhaseSpace:
    r"""
    Read an OSIRIS phase-space diagnostic into a `PICPhaseSpace`.

    Reads ``<path>/PHA/<field>/<species>/<field>-<species>-<dump>.h5`` for a series
    of dumps and reduces them to :math:`f(t, v, x)` in SI units. Uses `h5py`
    directly, so no OSIRIS-specific I/O package is needed.

    Parameters
    ----------
    path : path-like
        The OSIRIS output directory, conventionally named ``MS``.
    field : `str`
        Phase-space diagnostic name, for example ``"p1x1"`` or ``"p1x1x2"``.
    species : `str`
        Species name as given in the OSIRIS input deck.
    reference_density : `float` or `~astropy.units.Quantity`
        Density the simulation is normalised to, in m\ :sup:`-3` if given as a bare
        float. Sets the plasma frequency, and so the conversion of the OSIRIS length
        unit :math:`c / \omega_p` and time unit :math:`1 / \omega_p` into SI. Pass
        the same value to `spectra_from_phase_spaces`.
    label : `str`, optional
        `~plasmapy.particles` species identifier, for example ``"Al 13+"``. Defaults
        to ``"e-"`` when *is_electron* is `True`; otherwise required, since the
        forward model needs the ion charge and mass.
    is_electron : `bool`, optional
        Whether this is an electron population. Inferred from *label* if omitted.
    timesteps : iterable of `int`, optional
        Dump indices to read. The default reads every dump present, in order.
    spatial_axis : `str`, optional
        Which spatial axis to keep, for example ``"x1"``. The others are reduced --
        this is how a ``p1x1x2`` diagnostic is collapsed onto the shock normal.
        Defaults to the first spatial axis in the file.
    transverse_position : `dict`, optional
        ``{axis_name: coordinate}`` in m, saying where along each *other* spatial
        axis the probe sits. Defaults to the centre of each. Ignored when
        *transverse_reduction* is ``"chord"``.
    transverse_reduction : ``'slab'`` or ``'chord'``
        How to reduce the other spatial axes. ``"slab"`` -- the default -- keeps a
        localized region about *transverse_position*, which is what a scattering
        volume is. ``"chord"`` keeps the whole extent, for a measurement integrated
        along that direction. Both **average** over the cells kept; see the notes.
    slab_halfwidth : `float` or `~astropy.units.Quantity`, optional
        Half-thickness of the slab, in m. Defaults to the single nearest cell.
    position : `float` or `~astropy.units.Quantity`, optional
        Sample the diagnostic axis at this coordinate too, in m, so the reader
        returns the single point the probe looks at rather than a profile. Omit to
        keep the whole axis and choose the position later, which is what
        `spectra_from_phase_spaces` expects.
    charge_weighted : `bool`
        Whether the diagnostic holds charge density, as OSIRIS's ``charge``-weighted
        phase space does. When `True`, the sign is dropped and the result divided by
        the species' charge number, so the zeroth moment is a *number* density --
        which is what population fractions need when species have different charge
        states.
    relativistic_jacobian : `bool`
        Convert :math:`f(u)` to :math:`f(v)` with the Jacobian
        :math:`du/dv = \gamma^3 / c`. Leave this on; see the notes.
    progress : `bool`
        Show a progress bar while reading.

    Returns
    -------
    `PICPhaseSpace`

    Notes
    -----
    OSIRIS stores momentum as the proper velocity :math:`u = \gamma v / c`,
    normalised to each species' *own* mass despite the generic ``m_e c`` label its
    files carry, so :math:`v = u c / \sqrt{1 + u^2}` is correct for every species
    and no mass enters here.

    Reducing the transverse axes **averages** over the cells kept, rather than
    summing them. Summing would turn the density into a line integral and multiply
    the density handed to the forward model by the number of cells combined, which
    is what the ``osiris2thomson`` pipeline did for its 3-D phase spaces.

    OSIRIS phase-space diagnostics are projections fixed when the run was written,
    so the velocity component is whatever ``field`` holds -- ``p1x1x2`` gives
    :math:`v_1`. There is no way to reproject onto the scattering vector after the
    fact; that has to be arranged when choosing which diagnostic to dump.

    That map is nonlinear, so relabelling the axis is not enough: a distribution
    binned uniformly in :math:`u` is not uniformly binned in :math:`v`. With
    *relativistic_jacobian* on, ``f`` is multiplied by :math:`\gamma^3 / c`, which
    both gives a genuine velocity-space density and makes :math:`\int f \, dv` equal
    the density in units of *reference_density*. The correction is negligible for
    slow particles and reaches a factor of a few in the tails of a grid spanning
    :math:`|u| \sim 1`. The ``osiris2thomson`` pipeline this module derives from
    omitted it.

    Examples
    --------
    .. code-block:: python

       electrons = read_osiris_phase_space(
           "runs/omegashock/MS",
           "p1x1",
           "e",
           reference_density=9e17 * u.cm**-3,
           is_electron=True,
       )
    """
    path = Path(path)
    directory = path / "PHA" / field / species
    if not directory.is_dir():
        raise FileNotFoundError(f"no OSIRIS phase-space directory at {directory}")

    available = sorted(directory.glob(f"{field}-{species}-*.h5"))
    if not available:
        raise FileNotFoundError(f"no {field}-{species}-*.h5 files in {directory}")

    if timesteps is None:
        files = available
    else:
        by_index = {_osiris_dump_index(p.name): p for p in available}
        files = []
        for step in timesteps:
            if int(step) not in by_index:
                raise FileNotFoundError(
                    f"dump {int(step)} not found in {directory}; available dumps "
                    f"run from {min(by_index)} to {max(by_index)}."
                )
            files.append(by_index[int(step)])

    if label is None:
        if not is_electron:
            raise ValueError(
                f"a species label is required for {species!r} so the forward model "
                'knows its charge and mass; pass e.g. label="Al 13+", or '
                "is_electron=True for electrons."
            )
        label = "e-"

    reference_density = float(_si_values(reference_density, u.m**-3))
    if reference_density <= 0:
        raise ValueError(
            f"reference_density must be positive; got {reference_density} m^-3."
        )
    omega_p = _plasma_frequency(reference_density)
    skin_depth = const.c.si.value / omega_p

    # --- axis layout, from the first file ---
    with h5py.File(files[0], "r") as handle:
        dataset = handle[field]
        layout = _osiris_axis_layout(handle, dataset.ndim)
        shape = dataset.shape

    for axis in layout:
        scale = skin_depth if axis["name"].startswith("x") else 1.0
        axis["values"] = (
            np.linspace(axis["min"], axis["max"], shape[axis["numpy_axis"]]) * scale
        )

    momentum_axes = [a for a in layout if a["name"].startswith("p")]
    spatial_axes = [a for a in layout if a["name"].startswith("x")]
    if len(momentum_axes) != 1:
        raise ValueError(
            f"expected exactly one momentum axis in {field!r}; found "
            f"{[a['name'] for a in momentum_axes]}."
        )
    if not spatial_axes:
        raise ValueError(f"{field!r} has no spatial axis.")

    if spatial_axis is None:
        kept_space = spatial_axes[0]
    else:
        matches = [a for a in spatial_axes if a["name"] == spatial_axis]
        if not matches:
            raise ValueError(
                f"{field!r} has no spatial axis {spatial_axis!r}; available: "
                f"{[a['name'] for a in spatial_axes]}."
            )
        kept_space = matches[0]

    momentum = momentum_axes[0]
    transverse = [
        a for a in spatial_axes if a["numpy_axis"] != kept_space["numpy_axis"]
    ]
    selection = _transverse_selection(
        transverse, transverse_position, transverse_reduction, slab_halfwidth
    )

    u_axis = momentum["values"]
    x_axis = kept_space["values"]

    # An explicit sampling position reduces the diagnostic axis as well, so the
    # reader returns just the point the probe looks at.
    kept_range = None
    if position is not None:
        kept_range = _slab_indices(
            x_axis, float(_si_values(position, u.m)), slab_halfwidth
        )
        x_axis = np.atleast_1d(x_axis[kept_range[0] : kept_range[1]].mean())

    # Slicing preserves the axes, so the means below do not shift each other.
    index = [slice(None)] * len(layout)
    for numpy_axis, (low, high) in selection.items():
        index[numpy_axis] = slice(low, high)
    if kept_range is not None:
        index[kept_space["numpy_axis"]] = slice(*kept_range)
    index = tuple(index)
    collapse = tuple(selection)

    # --- read every dump ---
    blocks = []
    times = []
    iterator = tqdm(files, desc=f"Reading {species} phase space") if progress else files
    for file in iterator:
        with h5py.File(file, "r") as handle:
            data = np.asarray(handle[field][()], dtype=np.float64)
            times.append(float(np.atleast_1d(handle.attrs["TIME"])[0]))
        data = data[index]
        if kept_range is not None:
            # keepdims so the transverse collapse below still finds its axes.
            data = data.mean(axis=kept_space["numpy_axis"], keepdims=True)
        if collapse:
            # Average, not sum: summing turns a density into a line integral and
            # scales it by however many cells were combined.
            data = data.mean(axis=collapse)
        # Reduce to (momentum, space) regardless of the file's axis order.
        remaining = [
            a["numpy_axis"]
            for a in sorted(layout, key=lambda a: a["numpy_axis"])
            if a["numpy_axis"] in (momentum["numpy_axis"], kept_space["numpy_axis"])
        ]
        if remaining.index(momentum["numpy_axis"]) != 0:
            data = data.T
        blocks.append(data)

    f = np.stack(blocks)
    t = np.asarray(times, dtype=np.float64) / omega_p

    # --- OSIRIS conventions ---
    if charge_weighted:
        # Electron phase space is negative because it holds charge density.
        f = np.abs(f)
        charge_number = abs(Particle(label).charge_number)
        if charge_number != 1:
            f = f / charge_number

    gamma = np.sqrt(1.0 + u_axis**2)
    v_axis = u_axis * const.c.si.value / gamma
    if relativistic_jacobian:
        f = f * (gamma**3 / const.c.si.value)[np.newaxis, :, np.newaxis]

    return from_arrays(
        f=f,
        v=v_axis,
        x=x_axis,
        t=t,
        label=label,
        is_electron=is_electron,
        meta={
            "code": "OSIRIS",
            "path": str(directory),
            "field": field,
            "species": species,
            "dumps": [_osiris_dump_index(p.name) for p in files],
            "reference_density": reference_density,
            "plasma_frequency": omega_p,
            "skin_depth": skin_depth,
            "spatial_axis": kept_space["name"],
            "transverse_axes": [a["name"] for a in transverse],
            "transverse_reduction": transverse_reduction if transverse else None,
            "slab_halfwidth": slab_halfwidth,
            "sample_position": None if position is None else float(x_axis[0]),
            "charge_weighted": charge_weighted,
            "relativistic_jacobian": relativistic_jacobian,
        },
    )


def _load_yt():
    """Import yt, with an actionable message when it is not installed."""
    try:
        import yt  # noqa: PLC0415  (optional dependency, imported on demand)
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading WarpX output requires yt, which is not a required "
            "dependency of PlasmaPy. Install it with `pip install yt`, or "
            "`pip install plasmapy[pic]`."
        ) from error
    yt.set_log_level(50)
    return yt


def _warpx_plotfiles(path: Path, prefix: str, timesteps) -> list[Path]:
    """Sorted plotfile directories for a WarpX diagnostic."""
    available = sorted(p for p in path.glob(f"{prefix}*") if p.is_dir())
    if not available:
        raise FileNotFoundError(f"no {prefix}* plotfiles in {path}")
    if timesteps is None:
        return available
    try:
        return [available[int(step)] for step in timesteps]
    except IndexError as error:
        raise IndexError(
            f"requested a plotfile beyond the {len(available)} present in {path}."
        ) from error


def _warpx_fields(dataset, species: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """
    Which position and momentum components a plotfile actually carries.

    A 1-D WarpX run stores its single coordinate as ``particle_position_x`` even
    when the deck calls that direction ``z``, and a 2-D run stores ``x`` and
    ``y``; momenta keep their physical names. Detecting what is present avoids
    asking the caller to know that mapping.
    """
    available = {name for kind, name in dataset.field_list if kind == species}
    positions = tuple(
        f"particle_position_{c}" for c in "xyz" if f"particle_position_{c}" in available
    )
    momenta = tuple(
        f"particle_momentum_{c}" for c in "xyz" if f"particle_momentum_{c}" in available
    )
    if not positions:
        raise KeyError(
            f"{species!r} carries no particle_position_* fields; the plotfile has "
            f"particle types {dataset.particle_types}."
        )
    return positions, momenta


#: Which physical directions a WarpX run of each dimensionality spans. A 1-D run
#: is along ``z`` and a 2-D run spans ``x``-``z``, even though the plotfile stores
#: their coordinates as ``particle_position_x`` and ``particle_position_{x,y}``.
_WARPX_DECK_AXES = {1: ("z",), 2: ("x", "z"), 3: ("x", "y", "z")}


def _warpx_auto_momentum_field(species: str, n_spatial: int, axis: int) -> str:
    r"""
    The momentum component to project onto when the caller named none.

    Only unambiguous in one dimension, where the single resolved direction is the
    only thing :math:`\\hat{k}` can sensibly lie along. Momenta are stored for all
    three directions whatever the geometry, so in 2-D and 3-D a default would be a
    silently wrong answer rather than an error -- refuse instead.
    """
    if n_spatial == 1:
        return "particle_momentum_z"
    deck = _WARPX_DECK_AXES.get(n_spatial, ())
    likely = (
        f"particle_momentum_{deck[axis]}" if axis < len(deck) else "particle_momentum_z"
    )
    raise ValueError(
        f"{species!r} comes from a {n_spatial}-D run, where no single momentum "
        "component is the scattering direction in general: the Thomson k vector "
        "is set by the probe and detector geometry, not by the grid. Pass "
        "scatter_direction as the unit vector k in simulation coordinates "
        "(the physically correct choice), or momentum_field explicitly to "
        f"project onto one axis -- {likely!r} is the run's own axis {axis}."
    )


def _warpx_step_number(plotfile: Path, prefix: str) -> int | None:
    """The timestep a plotfile directory name encodes, or `None`."""
    tail = plotfile.name[len(prefix) :]
    return int(tail) if tail.isdigit() else None


def _warpx_matching_plotfile(reference, prefix: str, step: int | None) -> Path | None:
    """
    The plotfile of another diagnostic written at the same step.

    Particles and fields routinely live in separate diagnostics -- one written
    with ``write_species = 0``, the other with ``fields_to_plot = none`` -- so a
    cross-check between them has to match them up by step number rather than
    expecting both in one directory.
    """
    if step is None:
        return None
    candidate = Path(reference) / f"{prefix}{step:06d}"
    if candidate.is_dir():
        return candidate
    for other in sorted(Path(reference).glob(f"{prefix}*")):
        if other.is_dir() and _warpx_step_number(other, prefix) == step:
            return other
    return None


def _warpx_rho_field(dataset, species: str) -> tuple[str, str] | None:
    """The per-species charge-density field in a plotfile, or `None`."""
    wanted = f"rho_{species}"
    for kind, mesh_name in dataset.field_list:
        if mesh_name == wanted:
            return (kind, mesh_name)
    return None


def _warpx_deposited_weight(dataset, species: str, charge: float) -> float | None:
    r"""
    Total physical particles of *species*, from the charge the code deposited.

    Integrating ``rho_<species>`` over the domain and dividing by the charge
    gives the same number as summing macroparticle weights -- unless the
    particles being read are not the particles the code deposited. The common
    way for that to happen is ``<diag>.<species>.random_fraction``, which
    subsamples the particle output *without* reweighting, so every density built
    from it is low by that factor with nothing in the file to say so.

    Returns `None` when the plotfile carries no charge density for *species*.
    """
    field = _warpx_rho_field(dataset, species)
    if field is None or charge == 0:
        return None
    try:
        left = np.asarray(dataset.domain_left_edge.to("m"), dtype=np.float64)
        right = np.asarray(dataset.domain_right_edge.to("m"), dtype=np.float64)
        dimensions = np.asarray(dataset.domain_dimensions, dtype=int)
        # yt hands boxlib fields over as dimensionless, and cell_volume in code
        # units, so take the cell volume from the domain instead -- exact for
        # the uniform level-0 grid a WarpX plotfile holds. Directions the run
        # does not resolve span 1 m in WarpX, which is already in the edges.
        cell_volume = float(np.prod(np.abs(right - left) / np.maximum(dimensions, 1)))
        grid = dataset.covering_grid(
            level=0,
            left_edge=dataset.domain_left_edge,
            dims=dataset.domain_dimensions,
        )
        total_charge = float(np.sum(np.asarray(grid[field], dtype=np.float64)))
    except (KeyError, ValueError, TypeError, AttributeError):
        # yt's frontends differ in how they name and unit-tag mesh fields; a
        # cross-check that cannot be made is not a failure, it is just absent.
        return None
    return total_charge * cell_volume / charge


def _check_against_deposited_charge(
    dataset,
    *,
    species: str,
    label: str,
    charge,
    particle_weight: float,
    source: str,
) -> dict[str, Any] | None:
    """
    Compare the macroparticles being read against the charge the code deposited.

    Returns the comparison for the caller to record, or `None` when the plotfile
    carries no per-species charge density to compare against. Warns when the two
    disagree by more than `_DENSITY_CHECK_TOLERANCE`.
    """
    if charge is None:
        charge_si = float(Particle(label).charge.to(u.C).value)
    else:
        charge_si = float(_si_values(charge, u.C))
    deposited = _warpx_deposited_weight(dataset, species, charge_si)
    if deposited is None or deposited == 0:
        return None

    ratio = particle_weight / deposited
    if abs(ratio - 1.0) > _DENSITY_CHECK_TOLERANCE:
        hint = ""
        if 0 < ratio < 1:
            hint = (
                f" A ratio near {ratio:.3g} is what "
                f"`<diag>.{species}.random_fraction = {ratio:.3g}` produces: that "
                "option subsamples the particle output without reweighting, so "
                "every density built from it is low by exactly that factor. Read "
                "a diagnostic that writes all particles instead."
            )
        warnings.warn(
            f"the macroparticles read for {species!r} carry "
            f"{particle_weight:.6g} physical particles, but the charge density "
            f"{f'rho_{species}'!r} in {source} implies {deposited:.6g} -- a "
            f"ratio of {ratio:.6g}. Densities from this read will be wrong by "
            f"that factor.{hint}",
            RuntimeWarning,
            stacklevel=3,
        )
    return {
        "particle_weight": particle_weight,
        "deposited_weight": deposited,
        "ratio": ratio,
        "charge": charge_si,
        "source": source,
    }


def _warpx_frame(
    yt,
    plotfile: Path,
    species: str,
    position_fields: tuple[str, ...] | None = None,
    momentum_fields: tuple[str, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, tuple[np.ndarray, np.ndarray]]:
    """
    Read one species from one plotfile.

    Returns ``(positions, momenta, weight, time, domain)`` with *positions*
    shaped ``(n_spatial, n_particles)`` and *momenta* ``(n_momentum,
    n_particles)``, so callers can work in whatever dimensionality the run has.
    """
    dataset = yt.load(str(plotfile))
    data = dataset.all_data()
    if position_fields is None or momentum_fields is None:
        detected = _warpx_fields(dataset, species)
        position_fields = position_fields or detected[0]
        momentum_fields = momentum_fields or detected[1]
    try:
        positions = np.stack(
            [np.asarray(data[species, name].to("m")) for name in position_fields]
        )
        momenta = np.stack(
            [np.asarray(data[species, name].to("kg*m/s")) for name in momentum_fields]
        )
        weight = np.asarray(data[species, "particle_weight"])
    except Exception as error:
        raise KeyError(
            f"could not read {species!r} fields {position_fields} / "
            f"{momentum_fields} from {plotfile}; the plotfile carries particle "
            f"types {dataset.particle_types}."
        ) from error
    left = np.asarray(dataset.domain_left_edge.to("m"))
    right = np.asarray(dataset.domain_right_edge.to("m"))
    return positions, momenta, weight, float(dataset.current_time), (left, right)


def read_warpx_phase_space(  # noqa: C901, PLR0912, PLR0915
    path,
    species: str,
    *,
    mass,
    label: str | None = None,
    is_electron: bool | None = None,
    position_fields=None,
    momentum_fields=None,
    axis: int = 0,
    momentum_field: str | None = "auto",
    scatter_direction=None,
    transverse_position=None,
    transverse_reduction: str = "slab",
    slab_halfwidth=None,
    position=None,
    n_velocity_bins: int = 512,
    n_position_bins: int = 512,
    velocity_range=None,
    velocity_scan: str = "all",
    position_range=None,
    transverse_area: float | None = None,
    timesteps=None,
    prefix: str = "diag1",
    cache=None,
    validate_density: bool = True,
    density_reference: str | None = None,
    density_reference_path=None,
    charge=None,
    progress: bool = True,
) -> PICPhaseSpace:
    r"""
    Bin WarpX macroparticles into a `PICPhaseSpace`.

    WarpX writes raw macroparticles rather than a binned phase space, so unlike
    the OSIRIS reader this one has to build :math:`f(t, v, x)` itself, with a
    weighted two-dimensional histogram per timestep.

    Parameters
    ----------
    path : path-like
        Directory holding the plotfiles, conventionally ``<run>/diags``.
    species : `str`
        WarpX species name, for example ``"amb_ions"``.
    mass : `float` or `~astropy.units.Quantity`
        The mass the *simulation* gave this species, in kg if a bare float. This
        is what converts momentum to velocity, and in a reduced-mass-ratio run it
        is **not** the physical mass of the species named by *label* -- see the
        notes.
    label : `str`, optional
        `~plasmapy.particles` identifier for the physical species the population
        represents. Defaults to ``"e-"`` when *is_electron* is `True`; otherwise
        required.
    is_electron : `bool`, optional
        Whether this is an electron population. Inferred from *label* if omitted.
    position_fields, momentum_fields : sequence of `str`, optional
        Particle fields to read. Detected from the plotfile by default, which
        handles WarpX storing a 1-D run's only coordinate as
        ``particle_position_x`` even when the deck calls that direction ``z``, and
        a 2-D run's as ``x`` and ``y``.
    axis : `int`
        Which entry of *position_fields* is the diagnostic axis -- the one the
        spectrogram is resolved along. The rest are the transverse directions.
    momentum_field : `str`, optional
        Shorthand for pointing the scattering vector along a single momentum
        component; equivalent to passing that unit vector as *scatter_direction*.
        The default, ``"auto"``, resolves to ``"particle_momentum_z"`` for a 1-D
        run -- the only direction such a run resolves -- and **raises** for 2-D
        and 3-D, where momenta are stored for all three directions and any
        default would be a silently wrong projection. Set *scatter_direction*
        there instead.
    scatter_direction : array_like, optional
        The scattering vector :math:`\hat{k}` in *simulation* coordinates. The
        binned velocity is the component of the particle velocity along it, which
        is what the forward model assumes its distributions are resolved along.
        For a run with more than one velocity component this is the physically
        correct choice; naming a single component is only right when
        :math:`\hat{k}` happens to lie along that axis.
    transverse_position : array_like, optional
        Where the scattering volume sits along each transverse axis, in m. One
        value per transverse axis, in axis order. Defaults to the domain centre.
    transverse_reduction : ``'slab'`` or ``'chord'``
        ``"slab"`` -- the default -- keeps only particles within
        *slab_halfwidth* of *transverse_position*, which is what a scattering
        volume is. ``"chord"`` keeps every particle, for a measurement integrated
        along the transverse directions.
    slab_halfwidth : `float` or `~astropy.units.Quantity`, optional
        Half-thickness of the slab, in m. Defaults to half a simulation cell.
    position : `float` or `~astropy.units.Quantity`, optional
        Sample the diagnostic axis here too, in m, returning the single point the
        probe looks at rather than a profile. Omit to keep the whole axis and
        choose the position later.
    n_velocity_bins, n_position_bins : `int`
        Histogram resolution.
    velocity_range, position_range : pair of `float`, optional
        Histogram bounds, in m/s and m. *position_range* defaults to the
        simulation domain. *velocity_range* defaults to a symmetric range around
        zero with 20% headroom over the fastest particle found by the scan below
        -- pass it explicitly for a grid that does not depend on the frames read.
        Particles falling outside either range are discarded by the histogram; the
        reader counts them and warns if they carry an appreciable share of the
        weight.
    velocity_scan : ``'all'`` or ``'ends'``
        Which frames size the default *velocity_range*. ``"all"`` -- the default
        -- reads every frame first, costing a second pass but sizing the axis to
        what the run actually does. ``"ends"`` uses only the first and last
        frames, which is faster but clips whatever happens in between; for a
        shock, that is the measurement. Ignored when *velocity_range* is given.
    transverse_area : `float`, optional
        Cross-sectional area in m\ :sup:`2` used to turn macroparticle weights
        into a density. Defaults to the product of the domain's transverse
        extents, which is 1 m\ :sup:`2` for a 1-D run.
    timesteps : iterable of `int`, optional
        Indices into the sorted list of plotfiles. The default reads all of them.
    prefix : `str`
        Plotfile name prefix. Note that a field-only diagnostic written with
        ``write_species = 0`` carries no particles and cannot be used here.
    cache : path-like, optional
        ``.npz`` file to memoise the binned result in. Reading and histogramming
        millions of macroparticles across a long series is slow, and the result
        only depends on the settings recorded alongside it.
    validate_density : `bool`
        Cross-check the macroparticles being read against ``rho_<species>``, the
        charge density the code itself deposited, and warn if the two disagree.
        This is what catches a diagnostic written with ``random_fraction``, which
        subsamples particles without reweighting and so makes every density built
        from it low by that factor, with nothing in the file to say so. Skipped
        silently when no plotfile carries a charge density for *species*.
    density_reference : `str`, optional
        Prefix of another diagnostic whose plotfiles carry ``rho_<species>``,
        matched to the particle plotfiles by step number. Particles and fields
        routinely live in separate diagnostics -- one written with
        ``write_species = 0``, the other with ``fields_to_plot = none`` -- and
        the check is only possible across them. The default looks in the
        particle plotfile itself.
    density_reference_path : path-like, optional
        Directory holding the *density_reference* plotfiles, if not *path*.
    charge : `float` or `~astropy.units.Quantity`, optional
        Charge the *simulation* gave this species, for *validate_density*, in C
        if a bare float. Defaults to the physical charge of *label*, which is
        right unless the deck changed it.
    progress : `bool`
        Show a progress bar while reading.

    Returns
    -------
    `PICPhaseSpace`

    Notes
    -----
    Macroparticle **weights** are included in the histogram, so ``f`` is a
    distribution rather than a count. It is scaled by the volume the histogram
    actually covers -- the spatial bin, times each transverse extent kept, times
    the extents of any dimension the run does not resolve -- so that
    :math:`\int f \, dv` is the number density in m\ :sup:`-3` regardless of
    dimensionality or slab thickness. Pass ``reference_density=1 * u.m**-3`` to
    `spectra_from_phase_spaces`.

    The Lorentz factor is computed from the **full** momentum, then the velocity
    is projected onto :math:`\hat{k}`. Using a single component for both would
    understate :math:`\gamma` whenever the transverse momentum is appreciable.

    In a reduced-mass-ratio run, *mass* and *label* describe different things and
    both are needed. *mass* is the simulation's value, which converts the stored
    momentum into the velocities the simulation actually evolved; *label* names
    the physical species, from which the forward model takes the charge and mass
    entering the susceptibility. Correcting the velocities themselves is a
    separate step -- see ``velocity_scale_factor`` in `condition_phase_space`.

    Examples
    --------
    .. code-block:: python

       ions = read_warpx_phase_space(
           "runs/R1_paper/diags",
           "amb_ions",
           mass=100 * const.m_e,
           label="p+",
           cache="amb_ions.npz",
       )
    """
    path = Path(path)
    mass = float(_si_values(mass, u.kg))
    if mass <= 0:
        raise ValueError(f"mass must be positive; got {mass} kg.")

    if label is None:
        if not is_electron:
            raise ValueError(
                f"a species label is required for {species!r} so the forward model "
                'knows its charge and mass; pass e.g. label="p+", or '
                "is_electron=True for electrons."
            )
        label = "e-"

    settings = {
        "species": species,
        "mass": mass,
        "position_fields": None if position_fields is None else list(position_fields),
        "momentum_fields": None if momentum_fields is None else list(momentum_fields),
        "axis": axis,
        "momentum_field": momentum_field,
        "scatter_direction": (
            None if scatter_direction is None else list(np.ravel(scatter_direction))
        ),
        "transverse_position": (
            None if transverse_position is None else list(np.ravel(transverse_position))
        ),
        "transverse_reduction": transverse_reduction,
        "slab_halfwidth": slab_halfwidth,
        "position": position,
        "n_velocity_bins": n_velocity_bins,
        "n_position_bins": n_position_bins,
        "velocity_range": None if velocity_range is None else list(velocity_range),
        "position_range": None if position_range is None else list(position_range),
        # Both of the following change ``f`` -- transverse_area scales it
        # linearly, and timesteps changes which frames it holds -- so both have
        # to key the cache. Leaving them out let a cache built from a five-frame
        # test read be reused, silently, for the full series.
        "transverse_area": transverse_area,
        "timesteps": None if timesteps is None else [int(s) for s in timesteps],
        "velocity_scan": velocity_scan,
        "prefix": prefix,
        "density_reference": density_reference,
    }
    signature = repr(sorted(settings.items()))

    cache = None if cache is None else Path(cache)
    if cache is not None and cache.is_file():
        stored = np.load(cache, allow_pickle=False)
        if str(stored["signature"]) == signature:
            resolved = (
                json.loads(str(stored["resolved"])) if "resolved" in stored else {}
            )
            return from_arrays(
                f=stored["f"],
                v=stored["v"],
                x=stored["x"],
                t=stored["t"],
                label=label,
                is_electron=is_electron,
                meta={"code": "WarpX", "cache": str(cache), **settings, **resolved},
            )

    yt = _load_yt()
    plotfiles = _warpx_plotfiles(path, prefix, timesteps)

    # Detect what the run actually stores, so the caller need not know that a
    # 1-D WarpX run calls its only coordinate "x".
    probe_dataset = yt.load(str(plotfiles[0]))
    detected_positions, detected_momenta = _warpx_fields(probe_dataset, species)
    position_fields = tuple(position_fields or detected_positions)
    momentum_fields = tuple(momentum_fields or detected_momenta)
    n_spatial = len(position_fields)
    if not 0 <= axis < n_spatial:
        raise ValueError(
            f"axis must index one of the {n_spatial} position field(s) "
            f"{position_fields}; got {axis}."
        )

    # The velocity the diagnostic sees is the component along the scattering
    # vector. Naming a single momentum component is the same as pointing k at
    # that axis, so both routes go through one projection.
    if scatter_direction is None:
        if momentum_field in (None, "auto"):
            momentum_field = _warpx_auto_momentum_field(species, n_spatial, axis)
        if momentum_field not in momentum_fields:
            raise KeyError(
                f"{species!r} has no {momentum_field!r}; it carries "
                f"{list(momentum_fields)}. Pass momentum_field or "
                "scatter_direction explicitly."
            )
        component = momentum_fields.index(momentum_field)
        direction = np.zeros(len(momentum_fields))
        direction[component] = 1.0
    else:
        direction = np.asarray(scatter_direction, dtype=np.float64).ravel()
        if direction.size < len(momentum_fields):
            direction = np.pad(direction, (0, len(momentum_fields) - direction.size))
        direction = direction[: len(momentum_fields)]
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("scatter_direction must not be the zero vector.")
        direction = direction / norm

    def velocities(momenta):
        """Lab-frame velocity along the scattering vector, in m/s."""
        proper = momenta / (mass * const.c.si.value)
        # gamma comes from the full momentum, not just the projected part.
        gamma = np.sqrt(1.0 + np.sum(proper**2, axis=0))
        return (direction @ proper) * const.c.si.value / gamma

    # --- fix the histogram grid, so every frame shares one axis ---
    read = (position_fields, momentum_fields)
    if velocity_scan not in {"all", "ends"}:
        raise ValueError(
            f"velocity_scan must be 'all' or 'ends'; got {velocity_scan!r}."
        )
    # Only the domain edges are needed before the range is settled; take them
    # from the first frame, which every path reads anyway.
    first = _warpx_frame(yt, plotfiles[0], species, *read)
    left, right = first[4]

    density_check: dict[str, Any] | None = None
    if validate_density:
        reference_dataset, reference_name = probe_dataset, plotfiles[0].name
        if density_reference is not None:
            step = _warpx_step_number(plotfiles[0], prefix)
            match = _warpx_matching_plotfile(
                path if density_reference_path is None else density_reference_path,
                density_reference,
                step,
            )
            if match is None:
                raise FileNotFoundError(
                    f"no {density_reference}* plotfile at step {step} to check "
                    f"{plotfiles[0].name} against."
                )
            reference_dataset, reference_name = yt.load(str(match)), match.name
        density_check = _check_against_deposited_charge(
            reference_dataset,
            species=species,
            label=label,
            charge=charge,
            particle_weight=float(np.sum(first[2])),
            source=reference_name,
        )

    # --- which particles the scattering volume contains ---
    transverse_axes = [k for k in range(n_spatial) if k != axis]
    if transverse_reduction not in {"slab", "chord"}:
        raise ValueError(
            f"transverse_reduction must be 'slab' or 'chord'; got "
            f"{transverse_reduction!r}."
        )
    centres = np.array(
        [0.5 * (left[k] + right[k]) for k in range(n_spatial)], dtype=np.float64
    )
    if transverse_position is not None:
        supplied = np.asarray(transverse_position, dtype=np.float64).ravel()
        if supplied.size != len(transverse_axes):
            raise ValueError(
                f"transverse_position needs one coordinate per transverse axis "
                f"({len(transverse_axes)}); got {supplied.size}."
            )
        for value, k in zip(supplied, transverse_axes, strict=True):
            centres[k] = value
    half_width = (
        None if slab_halfwidth is None else float(_si_values(slab_halfwidth, u.m))
    )

    def select(positions):
        """Mask of particles inside the scattering volume."""
        keep = np.ones(positions.shape[1], dtype=bool)
        if transverse_reduction == "chord":
            return keep
        for k in transverse_axes:
            width = half_width
            if width is None:
                # Default to one cell of the simulation grid.
                width = (
                    0.5
                    * abs(right[k] - left[k])
                    / max(int(probe_dataset.domain_dimensions[k]), 1)
                )
            keep &= np.abs(positions[k] - centres[k]) <= width
        return keep

    if velocity_range is None:
        # Scanning every frame costs a second read pass, but the alternative --
        # inferring the axis from the first and last frames alone -- silently
        # clips whatever the run does in between, which for a shock is the
        # entire point of the measurement.
        if velocity_scan == "all":
            scanned = list(plotfiles)
        else:
            scanned = list(dict.fromkeys([plotfiles[0], plotfiles[-1]]))
        # The first frame is already in hand; re-reading it would double the
        # cost of the common two-frame scan for nothing.
        frames = (
            first
            if plotfile == plotfiles[0]
            else _warpx_frame(yt, plotfile, species, *read)
            for plotfile in (
                tqdm(scanned, desc=f"Scanning {species} velocity range")
                if progress and len(scanned) > 2
                else scanned
            )
        )
        extreme = max(
            (
                float(np.max(np.abs(velocities(momenta[:, keep]))))
                for positions, momenta, _, _, _ in frames
                if momenta.size and (keep := select(positions)).any()
            ),
            default=0.0,
        )
        if extreme <= 0:
            raise ValueError(
                f"species {species!r} has no particles in the scattering volume "
                f"in any of the {len(scanned)} plotfile(s) scanned, so no "
                "velocity range could be inferred; pass velocity_range "
                "explicitly, widen slab_halfwidth, or check transverse_position."
            )
        # Headroom, but never past the speed of light: no particle can live
        # there, and the empty bins would only give the taper more room to
        # invent a tail.
        headroom = min(1.2 * extreme, 0.999 * const.c.si.value)
        v_lo, v_hi = -headroom, headroom
    else:
        v_lo, v_hi = float(velocity_range[0]), float(velocity_range[1])

    if position_range is not None:
        x_lo, x_hi = float(position_range[0]), float(position_range[1])
    elif position is not None:
        centre = float(_si_values(position, u.m))
        width = half_width
        if width is None:
            width = (
                0.5
                * abs(right[axis] - left[axis])
                / max(int(probe_dataset.domain_dimensions[axis]), 1)
            )
        x_lo, x_hi = centre - width, centre + width
        n_position_bins = 1
    else:
        x_lo, x_hi = float(left[axis]), float(right[axis])

    # The bin volume has to match the region the histogram actually covers:
    # the spatial bin, times each transverse extent kept, times the extents of
    # any dimension the simulation does not resolve (1 m apiece in WarpX).
    if transverse_area is None:
        transverse_area = 1.0
        for k in transverse_axes:
            if transverse_reduction == "chord":
                transverse_area *= abs(right[k] - left[k])
            else:
                width = half_width
                if width is None:
                    width = (
                        0.5
                        * abs(right[k] - left[k])
                        / max(int(probe_dataset.domain_dimensions[k]), 1)
                    )
                transverse_area *= 2.0 * width
        for k in range(n_spatial, len(left)):
            transverse_area *= abs(right[k] - left[k])

    v_edges = np.linspace(v_lo, v_hi, n_velocity_bins + 1)
    x_edges = np.linspace(x_lo, x_hi, n_position_bins + 1)
    v_axis = 0.5 * (v_edges[:-1] + v_edges[1:])
    x_axis = 0.5 * (x_edges[:-1] + x_edges[1:])
    bin_volume = np.diff(v_edges)[0] * np.diff(x_edges)[0] * transverse_area

    blocks = []
    times = []
    kept_weight = 0.0
    clipped_weight = 0.0
    clipped_count = 0
    iterator = (
        tqdm(plotfiles, desc=f"Binning {species} macroparticles")
        if progress
        else plotfiles
    )
    for plotfile in iterator:
        positions, momenta, weight, time, _ = _warpx_frame(yt, plotfile, species, *read)
        keep = select(positions)
        v_kept = velocities(momenta[:, keep])
        x_kept = positions[axis][keep]
        w_kept = weight[keep]
        # np.histogram2d drops out-of-range samples without saying so. Count
        # the ones lost to the velocity axis, because a clipped distribution
        # tail is exactly the failure the taper then hides. Only particles the
        # probe volume contains count: narrowing the position range is a
        # deliberate selection, like the transverse slab, not a loss.
        inside_x = (x_kept >= x_lo) & (x_kept <= x_hi)
        outside_v = inside_x & ((v_kept < v_lo) | (v_kept > v_hi))
        clipped_count += int(np.count_nonzero(outside_v))
        clipped_weight += float(w_kept[outside_v].sum())
        kept_weight += float(w_kept[inside_x].sum())
        histogram, _, _ = np.histogram2d(
            v_kept,
            x_kept,
            bins=[v_edges, x_edges],
            weights=w_kept,
        )
        blocks.append(histogram / bin_volume)
        times.append(time)

    clipped_fraction = clipped_weight / kept_weight if kept_weight > 0 else 0.0
    if clipped_count and clipped_fraction > _CLIP_WARNING_FRACTION:
        warnings.warn(
            f"{clipped_count} {species!r} macroparticle(s) carrying "
            f"{100 * clipped_fraction:.2f}% of the weight inside the scattering "
            f"volume fell outside the velocity axis "
            f"[{v_lo:.3e}, {v_hi:.3e}] m/s and were discarded, so the "
            "distribution is missing that much of its tail. Widen "
            "velocity_range, or leave it unset with velocity_scan='all' so the "
            "axis is sized from every frame.",
            RuntimeWarning,
            stacklevel=2,
        )

    f = np.stack(blocks)
    t = np.asarray(times, dtype=np.float64)

    resolved = {
        "transverse_area": transverse_area,
        "position_fields": list(position_fields),
        "momentum_fields": list(momentum_fields),
        "scatter_direction": direction.tolist(),
        "n_spatial": n_spatial,
        "transverse_axes": [position_fields[k] for k in transverse_axes],
        # f is already a density in m^-3, so the driver needs no further scale.
        "reference_density": 1.0,
        "velocity_range": [v_lo, v_hi],
        "position_range": [x_lo, x_hi],
        "clipped_count": clipped_count,
        "clipped_weight_fraction": clipped_fraction,
        "density_check": density_check,
    }

    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache,
            f=f,
            v=v_axis,
            x=x_axis,
            t=t,
            signature=np.str_(signature),
            resolved=np.str_(json.dumps(resolved)),
        )

    return from_arrays(
        f=f,
        v=v_axis,
        x=x_axis,
        t=t,
        label=label,
        is_electron=is_electron,
        meta={
            "code": "WarpX",
            "path": str(path),
            "plotfiles": [p.name for p in plotfiles],
            **settings,
            **resolved,
        },
    )


# ---------------------------------------------------------------------------
# openPMD phase space, as WarpX's ParticleHistogram2D writes it
# ---------------------------------------------------------------------------

#: Velocity conventions an openPMD histogram axis can be in.
_VELOCITY_KINDS = ("beta", "proper", "velocity")


def _openpmd_iteration(handle) -> tuple[str, dict[str, Any]]:
    """
    Locate the single iteration in a file-based openPMD series.

    Returns its group path and the attributes of the group itself, which carry
    the physical time.
    """
    base = _attr_text(handle, "basePath") if "basePath" in handle.attrs else "/data/%T/"
    root = base.split("%T")[0].strip("/")
    group = handle[root] if root else handle
    iterations = sorted((str(key) for key in group), key=int)
    if len(iterations) != 1:
        raise ValueError(
            f"expected one iteration per file in a file-based openPMD series; "
            f"found {len(iterations)} in {handle.filename}."
        )
    only = iterations[0]
    path = f"{root}/{only}" if root else only
    return path, dict(group[only].attrs)


def _openpmd_mesh(handle, iteration: str, mesh: str) -> "h5py.Dataset":
    """The mesh record holding the histogram, as an h5py dataset."""
    meshes_path = (
        _attr_text(handle, "meshesPath") if "meshesPath" in handle.attrs else "meshes/"
    )
    node = handle[f"{iteration}/{meshes_path.strip('/')}"]
    if mesh not in node:
        raise KeyError(
            f"{handle.filename} has no mesh {mesh!r}; it holds {sorted(node.keys())}."
        )
    record = node[mesh]
    # A scalar record component is the record itself; a vector one would nest.
    if isinstance(record, h5py.Group):
        components = sorted(record.keys())
        if len(components) != 1:
            raise ValueError(
                f"mesh {mesh!r} in {handle.filename} has {len(components)} "
                "components; a phase-space histogram is a scalar record."
            )
        record = record[components[0]]
    return record


def read_openpmd_phase_space(  # noqa: C901, PLR0912, PLR0915
    path,
    *,
    label: str,
    is_electron: bool | None = None,
    velocity_axis: str = "ordinate",
    velocity_kind: str = "beta",
    transverse_area: float = 1.0,
    position=None,
    position_scale: float = 1.0,
    mesh: str = "data",
    pattern: str = "*.h5",
    timesteps=None,
    progress: bool = True,
) -> PICPhaseSpace:
    r"""
    Read a phase space that a PIC code binned itself, from openPMD output.

    Where `read_warpx_phase_space` has to bin raw macroparticles, this reads a
    histogram the code already built -- WarpX's ``ParticleHistogram2D`` reduced
    diagnostic, and anything else writing a 2-D openPMD mesh in the same layout.
    That is much cheaper, needs no ``yt``, and can be written at every timestep
    rather than at the cadence full plotfiles can afford.

    It is also *more* accurate, if the deck is written the way
    `histogram2d_deck_block` writes it. Both axes of that diagnostic are
    arbitrary expressions of the particle state, so the projection onto the
    scattering vector -- with the Lorentz factor taken from the full momentum --
    can be evaluated per particle inside the code, rather than reconstructed
    afterwards from a single stored component.

    Parameters
    ----------
    path : path-like
        Directory the reduced diagnostic wrote, holding one file per iteration.
    label : `str`
        `~plasmapy.particles` identifier for the species.
    is_electron : `bool`, optional
        Whether this is an electron population. Inferred from *label* if omitted.
    velocity_axis : ``'ordinate'`` or ``'abscissa'``
        Which histogram axis holds velocity. ``ParticleHistogram2D`` writes the
        ordinate first, and `histogram2d_deck_block` puts velocity there.
    velocity_kind : ``'beta'``, ``'proper'``, or ``'velocity'``
        What that axis actually is. ``"beta"`` -- the default, and what
        `histogram2d_deck_block` emits -- is :math:`v \cdot \hat{k} / c`, the
        lab-frame velocity itself. ``"proper"`` is :math:`u = \gamma v / c`, in
        which case the reader converts and applies the :math:`\gamma^3 / c`
        Jacobian; that conversion assumes the other momentum components are
        negligible, since a single stored component does not determine
        :math:`\gamma`. ``"velocity"`` is m/s.
    transverse_area : `float`
        Cross-section in m\ :sup:`2` of the volume the histogram covers, used to
        turn binned weights into a density. For a 1-D run with no
        ``filter_function`` this is 1 m\ :sup:`2`, the extent WarpX gives its
        unresolved directions. `histogram2d_deck_block` reports the right value
        for the block it generates.
    position : `float` or `~astropy.units.Quantity`, optional
        Reduce to the single grid point nearest here, in m.
    position_scale : `float`
        Metres per unit of the position axis, if the abscissa expression is not
        already in metres.
    mesh : `str`
        Name of the openPMD mesh record. ``ParticleHistogram2D`` writes
        ``"data"``.
    pattern : `str`
        Glob for the files. Use ``"*.bp"`` -- with a backend that can read it --
        if the deck did not set ``openpmd_backend = h5``.
    timesteps : iterable of `int`, optional
        Indices into the sorted list of files. The default reads all.
    progress : `bool`
        Show a progress bar while reading.

    Returns
    -------
    `PICPhaseSpace`
        With ``f`` in SI, so `spectra_from_phase_spaces` needs no density scale.

    Raises
    ------
    `FileNotFoundError`
        If no files match *pattern* in *path*.

    Notes
    -----
    The histogram holds :math:`\sum w`, a count of physical particles per bin,
    so the reader divides by the bin's phase-space volume -- velocity width,
    position width, and *transverse_area* -- to make :math:`\int f\,dv` a number
    density.

    Bin edges are fixed in the deck and particles outside them are discarded by
    the code, before anything is written. Nothing in the file records how many,
    so unlike `read_warpx_phase_space` this reader cannot warn about a clipped
    tail. Choose the bounds generously.

    Examples
    --------
    .. code-block:: python

       electrons = read_openpmd_phase_space(
           "runs/R1/diags/reducedfiles/eps_electrons",
           label="e-",
           transverse_area=2.0e-8,
       )
    """
    path = Path(path)
    if velocity_axis not in {"ordinate", "abscissa"}:
        raise ValueError(
            f"velocity_axis must be 'ordinate' or 'abscissa'; got {velocity_axis!r}."
        )
    if velocity_kind not in _VELOCITY_KINDS:
        raise ValueError(
            f"velocity_kind must be one of {_VELOCITY_KINDS}; got {velocity_kind!r}."
        )

    files = sorted(p for p in path.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {path}")
    if timesteps is not None:
        try:
            files = [files[int(step)] for step in timesteps]
        except IndexError as error:
            raise IndexError(
                f"requested a file beyond the {len(files)} present in {path}."
            ) from error

    blocks = []
    times = []
    axes: tuple[np.ndarray, np.ndarray] | None = None
    functions: dict[str, str] = {}
    iterator = (
        tqdm(files, desc=f"Reading {path.name} phase space") if progress else files
    )
    for file in iterator:
        with h5py.File(file, "r") as handle:
            iteration, iteration_attrs = _openpmd_iteration(handle)
            record = _openpmd_mesh(handle, iteration, mesh)
            data = np.asarray(record[()], dtype=np.float64)
            attrs = dict(record.attrs)
            time_unit = float(iteration_attrs.get("timeUnitSI", 1.0))
            times.append(float(iteration_attrs["time"]) * time_unit)

            grid_unit = float(attrs.get("gridUnitSI", 1.0))
            spacing = np.asarray(attrs["gridSpacing"], dtype=np.float64) * grid_unit
            offset = np.asarray(attrs["gridGlobalOffset"], dtype=np.float64) * grid_unit
            centre = np.asarray(
                attrs.get("position", [0.5] * data.ndim), dtype=np.float64
            )
            built = tuple(
                offset[k] + spacing[k] * (np.arange(data.shape[k]) + centre[k])
                for k in range(data.ndim)
            )
            if axes is None:
                axes = built
                for name in ("function_abscissa", "function_ordinate", "filter"):
                    if name in attrs:
                        functions[name] = _attr_text(record, name)
            elif not all(np.allclose(a, b) for a, b in zip(axes, built, strict=True)):
                raise ValueError(
                    f"{file.name} is binned on a different grid from the first "
                    "file; a phase space needs one axis for the whole series."
                )
        blocks.append(data)

    if axes is None or len(axes) != 2:  # pragma: no cover - guarded by the reads above
        raise ValueError("expected a 2-D histogram.")

    # openPMD orders the axes (ordinate, abscissa), matching (velocity,
    # position) for the block this module's deck helper generates.
    stacked = np.stack(blocks)
    ordinate, abscissa = axes
    if velocity_axis == "ordinate":
        velocity_raw, x_axis = ordinate, abscissa
    else:
        velocity_raw, x_axis = abscissa, ordinate
        stacked = np.transpose(stacked, (0, 2, 1))

    x_axis = x_axis * position_scale
    velocity_width = float(np.mean(np.diff(velocity_raw)))
    position_width = float(np.mean(np.diff(x_axis)))

    speed_of_light = const.c.si.value
    if velocity_kind == "velocity":
        v_axis = velocity_raw
        jacobian = 1.0
    elif velocity_kind == "beta":
        v_axis = velocity_raw * speed_of_light
        velocity_width *= speed_of_light
        jacobian = 1.0
    else:
        gamma = np.sqrt(1.0 + velocity_raw**2)
        v_axis = velocity_raw * speed_of_light / gamma
        # Binning is uniform in u, so the density per unit v carries du/dv.
        jacobian = (gamma**3 / speed_of_light)[np.newaxis, :, np.newaxis]

    if np.any(np.diff(v_axis) <= 0):
        raise ValueError(
            "the velocity axis is not strictly increasing after conversion; "
            "check velocity_kind and the deck's bin_min/bin_max ordering."
        )

    bin_volume = abs(velocity_width) * abs(position_width) * transverse_area
    f = stacked * jacobian / bin_volume

    phase_space = from_arrays(
        f=f,
        v=v_axis,
        x=x_axis,
        t=np.asarray(times, dtype=np.float64),
        label=label,
        is_electron=is_electron,
        meta={
            "code": "openPMD histogram",
            "path": str(path),
            "files": [p.name for p in files],
            "mesh": mesh,
            "velocity_axis": velocity_axis,
            "velocity_kind": velocity_kind,
            "transverse_area": transverse_area,
            "reference_density": 1.0,
            **functions,
        },
    )
    return phase_space if position is None else phase_space.at_position(position)


def histogram2d_deck_block(
    name: str,
    species: str,
    *,
    position_range,
    velocity_range,
    scatter_direction=(0.0, 0.0, 1.0),
    position_axis: str = "z",
    n_position_bins: int = 512,
    n_velocity_bins: int = 512,
    intervals: int | str = 100,
    transverse_slab=None,
    backend: str = "h5",
) -> tuple[str, float]:
    r"""
    Generate the deck block for a WarpX phase-space diagnostic, and its area.

    `read_openpmd_phase_space` can only be as good as the diagnostic it reads,
    and the two have to agree about what the axes mean. This writes the
    ``ParticleHistogram2D`` block that produces exactly what that reader expects:
    position along the abscissa in metres, and the lab-frame velocity along
    :math:`\hat{k}` -- in units of :math:`c` -- along the ordinate.

    The velocity expression is worth reading:

    .. code-block:: text

       (kx*ux + ky*uy + kz*uz) / sqrt(1 + ux*ux + uy*uy + uz*uz)

    WarpX gives the parser ``ux`` as :math:`\gamma v_x / c`, so this is
    :math:`(\gamma \vec{v} \cdot \hat{k} / c) / \gamma`, the projection of the
    lab-frame velocity onto the scattering vector, with :math:`\gamma` taken
    from the **full** momentum. It is evaluated per particle inside the code,
    which is something no reader working from a single stored momentum component
    can reproduce.

    Parameters
    ----------
    name : `str`
        Name of the reduced diagnostic.
    species : `str`
        WarpX species name.
    position_range : pair of `float` or `~astropy.units.Quantity`
        Bounds of the position axis, in m if bare. Particles outside are
        discarded by the code, silently -- cover the whole region of interest.
    velocity_range : pair of `float` or `~astropy.units.Quantity`
        Bounds of the velocity axis, in m/s if bare, converted to units of
        :math:`c` for the deck.
    scatter_direction : array_like
        The scattering vector :math:`\hat{k}` in simulation coordinates.
    position_axis : ``'x'``, ``'y'``, or ``'z'``
        Which coordinate the abscissa bins.
    n_position_bins, n_velocity_bins : `int`
        Histogram resolution. Fixed at deck time; there is no rebinning later.
    intervals : `int` or `str`
        WarpX ``intervals`` string for the diagnostic. A reduced diag is cheap
        enough to write far more often than a full plotfile, which is most of
        the reason to use one.
    transverse_slab : `dict`, optional
        Scattering volume across the beam, as ``{axis: (centre, half_width)}``
        in m -- for example ``{"y": (0.0, 50e-6)}``. Becomes a
        ``filter_function``, so the selection happens in the code rather than
        after the fact. Omit to integrate over the whole transverse extent.
    backend : `str`
        openPMD backend. ``"h5"`` keeps the output readable by `h5py` alone,
        which is what `read_openpmd_phase_space` needs; WarpX otherwise defaults
        to ``bp5`` wherever ADIOS2 is compiled in.

    Returns
    -------
    block : `str`
        Deck text, ready to paste in.
    transverse_area : `float`
        Cross-section in m\ :sup:`2` the block selects -- pass this straight to
        `read_openpmd_phase_space` so its densities come out right.

    Notes
    -----
    The area returned assumes the unresolved directions of the run span 1 m,
    which is WarpX's convention, and that any axis not named in
    *transverse_slab* is left unfiltered and so contributes 1 m as well. For a
    2-D or 3-D run, filter every transverse direction and the area is the
    product of the slab thicknesses.

    Examples
    --------
    .. code-block:: python

       block, area = histogram2d_deck_block(
           "eps_electrons",
           "electrons",
           position_range=(0, 2e-3) * u.m,
           velocity_range=(-2e8, 2e8) * u.m / u.s,
           scatter_direction=(0, 0, 1),
       )
       print(block)
    """
    if position_axis not in "xyz":
        raise ValueError(f"position_axis must be x, y, or z; got {position_axis!r}.")
    direction = np.asarray(scatter_direction, dtype=np.float64).ravel()
    if direction.size != 3:
        raise ValueError(f"scatter_direction must be a 3-vector; got {direction.size}.")
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("scatter_direction must not be the zero vector.")
    direction = direction / norm

    x_lo, x_hi = (float(_si_values(bound, u.m)) for bound in position_range)
    v_lo, v_hi = (
        float(_si_values(bound, u.m / u.s)) / const.c.si.value
        for bound in velocity_range
    )
    if x_hi <= x_lo or v_hi <= v_lo:
        raise ValueError("ranges must be given as (low, high) with high > low.")

    # Drop the components k does not touch: the expression is read by people as
    # well as by the parser, and "0*ux + 0*uy + 1*uz" hides which axis it is.
    terms = [
        f"{weight:.12g}*u{letter}"
        for weight, letter in zip(direction, "xyz", strict=True)
        if weight != 0.0
    ]
    projection = " + ".join(terms).replace("+ -", "- ")
    velocity_function = f"({projection})/sqrt(1+ux*ux+uy*uy+uz*uz)"

    lines = [
        f"# --- phase space for plasmapy.diagnostics.pic_thomson, species {species} ---",
        f"# Append {name} to warpx.reduced_diags_names.",
        f"{name}.type = ParticleHistogram2D",
        f"{name}.species = {species}",
        f"{name}.intervals = {intervals}",
        f"{name}.openpmd_backend = {backend}",
        "",
        "# abscissa: position along the diagnostic axis, in metres",
        f"{name}.histogram_function_abs(t,x,y,z,ux,uy,uz,w) = {position_axis}",
        f"{name}.bin_number_abs = {n_position_bins}",
        f"{name}.bin_min_abs = {x_lo:.12g}",
        f"{name}.bin_max_abs = {x_hi:.12g}",
        "",
        "# ordinate: lab-frame velocity along k-hat, in units of c. ux is",
        "# gamma*vx/c, so dividing by gamma gives the velocity itself -- with",
        "# gamma from the full momentum, evaluated per particle in the code.",
        f"{name}.histogram_function_ord(t,x,y,z,ux,uy,uz,w) = {velocity_function}",
        f"{name}.bin_number_ord = {n_velocity_bins}",
        f"{name}.bin_min_ord = {v_lo:.12g}",
        f"{name}.bin_max_ord = {v_hi:.12g}",
        "",
        "# The macroparticle weight is the documented default, but WarpX has a",
        "# bug here: ParticleHistogram2D reads value_function into m_do_parser_value",
        "# and then never checks it, so with the option absent the kernel calls a",
        "# default-constructed (null) ParserExecutor. The histogram then fills with",
        "# uninitialised memory -- NaN and DBL_MAX in some bins, zero in the rest --",
        "# rather than failing. Stating the default explicitly avoids that, and is",
        "# harmless once the bug is fixed.",
        f"{name}.value_function(t,x,y,z,ux,uy,uz,w) = w",
    ]

    transverse_area = 1.0
    if transverse_slab:
        conditions = []
        for coordinate, bounds in sorted(transverse_slab.items()):
            if coordinate not in "xyz":
                raise ValueError(
                    f"transverse_slab keys must be x, y, or z; got {coordinate!r}."
                )
            slab_centre = float(_si_values(bounds[0], u.m))
            slab_half_width = float(_si_values(bounds[1], u.m))
            if slab_half_width <= 0:
                raise ValueError(
                    f"the {coordinate} slab half-width must be positive; "
                    f"got {slab_half_width}."
                )
            conditions.append(
                f"({coordinate}>{slab_centre - slab_half_width:.12g})"
                f"*({coordinate}<{slab_centre + slab_half_width:.12g})"
            )
            transverse_area *= 2.0 * slab_half_width
        lines += [
            "",
            "# the scattering volume across the beam, selected in the code",
            f"{name}.filter_function(t,x,y,z,ux,uy,uz,w) = " + "*".join(conditions),
        ]

    lines += [
        "",
        "# Read it back with:",
        f"#   read_openpmd_phase_space(<diags>/reducedfiles/{name},",
        f"#                            label=..., transverse_area={transverse_area:.12g})",
    ]
    return "\n".join(lines), transverse_area


# ---------------------------------------------------------------------------
# Hybrid (kinetic-ion / fluid-electron) WarpX output
# ---------------------------------------------------------------------------


def _warpx_mesh_field(dataset, name: str) -> tuple[str, str] | None:
    """Locate a mesh field by name whatever frontend prefix yt gave it."""
    for kind, mesh_name in dataset.field_list:
        if mesh_name == name and kind != "io":
            return (kind, mesh_name)
    return None


def _warpx_field_frame(
    yt, plotfile: Path, names: list[str]
) -> tuple[dict[str, np.ndarray], float, tuple, dict[int, float], list[int]]:
    """
    Read named mesh fields from one plotfile onto a uniform grid.

    Returns ``(fields, time, (left, right), spacing, resolved)`` where *fields*
    maps each name to an array over the run's resolved directions only,
    *spacing* is the cell size along each of those, and *resolved* lists the
    physical direction index (0, 1, 2 for x, y, z) each grid axis corresponds
    to. Squeezing the unresolved directions is what lets one code path serve
    1-D, 2-D, and 3-D runs.
    """
    dataset = yt.load(str(plotfile))
    dimensions = np.asarray(dataset.domain_dimensions, dtype=int)
    left = np.asarray(dataset.domain_left_edge.to("m"), dtype=np.float64)
    right = np.asarray(dataset.domain_right_edge.to("m"), dtype=np.float64)

    grid = dataset.covering_grid(
        level=0, left_edge=dataset.domain_left_edge, dims=dataset.domain_dimensions
    )
    fields = {}
    for name in names:
        located = _warpx_mesh_field(dataset, name)
        if located is None:
            raise KeyError(
                f"{plotfile.name} carries no mesh field {name!r}; it has "
                f"{sorted({f for _, f in dataset.field_list})}. Add it to the "
                "diagnostic's fields_to_plot."
            )
        fields[name] = np.asarray(grid[located], dtype=np.float64)

    # WarpX pads a 1-D or 2-D run out to three grid axes; drop the flat ones,
    # remembering which physical direction each surviving axis is.
    grid_axes = [k for k in range(len(dimensions)) if dimensions[k] > 1]
    n_grid = len(dimensions)
    deck = _WARPX_DECK_AXES.get(len(grid_axes), tuple("xyz"[:n_grid]))
    resolved = ["xyz".index(direction) for direction in deck]
    for name, data in fields.items():
        fields[name] = data.reshape([dimensions[k] for k in grid_axes])

    spacing = {
        physical: float(abs(right[k] - left[k]) / max(int(dimensions[k]), 1))
        for physical, k in zip(resolved, grid_axes, strict=True)
    }
    return fields, float(dataset.current_time), (left, right), spacing, resolved


def _curl_along(direction, magnetic, spacing, resolved) -> np.ndarray | float:
    r"""
    The component of :math:`\nabla \times \vec{B}` along *direction*.

    *magnetic* maps a physical direction index to that component of
    :math:`\vec{B}`; a component that is absent from the run is taken to be
    zero, and so are derivatives along unresolved directions. That second point
    is what makes the 1-D case exact and free: with only :math:`\partial_z`
    surviving, :math:`(\nabla \times \vec{B})_z \equiv 0`, so a probe looking
    along the axis of a 1-D run needs no magnetic field at all.
    """

    def derivative(along: int, component: int) -> np.ndarray | float:
        if along not in resolved or component not in magnetic:
            return 0.0
        return np.gradient(
            magnetic[component], spacing[along], axis=resolved.index(along)
        )

    curl = [
        derivative(1, 2) - derivative(2, 1),
        derivative(2, 0) - derivative(0, 2),
        derivative(0, 1) - derivative(1, 0),
    ]
    total = 0.0
    for weight, term in zip(direction, curl, strict=True):
        if weight != 0.0:
            total = total + weight * term
    return total


def _curl_needs(direction, resolved) -> set[int]:
    """Which magnetic components a curl along *direction* actually differentiates."""
    needed: set[int] = set()
    pairs = {0: ((1, 2), (2, 1)), 1: ((2, 0), (0, 2)), 2: ((0, 1), (1, 0))}
    for component, weight in enumerate(direction):
        if weight == 0.0:
            continue
        for along, source in pairs[component]:
            if along in resolved:
                needed.add(source)
    return needed


def read_warpx_hybrid_electrons(  # noqa: C901, PLR0912, PLR0915
    path,
    *,
    scatter_direction=None,
    axis: int = 0,
    label: str = "e-",
    mass=None,
    charge_state: float = 1.0,
    density_field: str = "rho",
    temperature_field: str | None = "Te",
    temperature_unit=u.eV,
    closure: dict[str, Any] | None = None,
    drift: str = "ampere",
    current_prefix: str = "j",
    magnetic_prefix: str = "B",
    transverse_position=None,
    transverse_reduction: str = "slab",
    slab_halfwidth=None,
    position=None,
    v=None,
    n_velocity_bins: int = 512,
    timesteps=None,
    prefix: str = "diag1",
    progress: bool = True,
) -> PICPhaseSpace:
    r"""
    Reconstruct the electron phase space of a hybrid WarpX run from its moments.

    WarpX's kinetic-ion / fluid-electron solver has no electron macroparticles:
    the electrons are an inertialess, quasineutral fluid, and everything the run
    knows about them is in three mesh fields. This reads those fields and hands
    the resulting drifting Maxwellian to `from_moments`, so a hybrid run can be
    forward-modelled by exactly the same pipeline as a fully kinetic one.

    The three moments come from:

    * **Density.** :math:`n_e = \rho / (Z e)`. Quasineutrality is how the solver
      defines the electron density, so with only ions carried as particles the
      deposited charge density *is* the electron density -- this is exact, not an
      estimate.
    * **Temperature.** The ``Te`` field, when the run solves an electron energy
      equation and dumps it. Otherwise the barotropic closure
      :math:`T_e = T_{e0} (n_e / n_0)^{\gamma - 1}`, via *closure*.
    * **Drift.** :math:`\vec{J}_e = \nabla \times \vec{B} / \mu_0 - \vec{J}_i`
      and :math:`\vec{u}_e = -\vec{J}_e / (e n_e)`, where :math:`\vec{J}_i` is
      the current the ion macroparticles deposited -- what the ``j`` fields hold
      in this solver, since the electrons deposit nothing.

    Parameters
    ----------
    path : path-like
        Directory holding the plotfiles, conventionally ``<run>/diags``.
    scatter_direction : array_like, optional
        The scattering vector :math:`\hat{k}` in simulation coordinates, as a
        3-vector. Defaults to the run's own axis for a 1-D run, and is required
        otherwise.
    axis : `int`
        Which resolved direction is the diagnostic axis.
    label : `str`
        Species identifier; ``"e-"`` unless the fluid stands for something else.
    mass : `float` or `~astropy.units.Quantity`, optional
        Mass setting the thermal width. Defaults to the electron mass. Note that
        the solver's massless-electron approximation does not apply here -- see
        the notes on `from_moments`.
    charge_state : `float`
        Mean ion charge :math:`\bar{Z}` relating the deposited charge density to
        the electron density. ``1`` unless *density_field* is a per-species
        ``rho_<name>`` rather than the total.
    density_field : `str`
        Mesh field holding the charge density, in C/m\ :sup:`3`.
    temperature_field : `str`, optional
        Mesh field holding the electron temperature. `None` forces the closure.
    temperature_unit : `~astropy.units.UnitBase`
        Unit of *temperature_field*. WarpX dumps ``Te`` in eV, matching the unit
        ``hybrid_pic_model.elec_temp`` is given in.
    closure : `dict`, optional
        Fallback barotropic closure, as ``{"T_e0": ..., "n0_ref": ...,
        "gamma": ...}`` -- the deck's ``hybrid_pic_model`` parameters. Used only
        when *temperature_field* is `None` or absent from the plotfiles.
    drift : ``'ampere'``, ``'ion_current'``, or ``'zero'``
        How to get the electron fluid velocity. ``"ampere"`` -- the default --
        subtracts the deposited ion current from
        :math:`\nabla \times \vec{B} / \mu_0`. ``"ion_current"`` assumes the
        total current vanishes along :math:`\hat{k}`, which is **exact** for a
        1-D run probed along its own axis and an approximation otherwise.
        ``"zero"`` ignores the drift entirely.
    current_prefix, magnetic_prefix : `str`
        Name stems of the current and magnetic field components, so that
        ``"j"`` means ``jx``, ``jy``, ``jz``.
    transverse_position, transverse_reduction, slab_halfwidth : optional
        The scattering volume, as in `read_warpx_phase_space`. Fields are
        **averaged** over the transverse directions kept, since they are
        intensive.
    position : `float` or `~astropy.units.Quantity`, optional
        Sample the diagnostic axis here, returning the single point the probe
        looks at rather than a profile.
    v : array_like, optional
        Velocity axis, passed to `from_moments`. The default is sized from the
        moments themselves.
    n_velocity_bins : `int`
        Resolution of the default velocity axis.
    timesteps : iterable of `int`, optional
        Indices into the sorted list of plotfiles. The default reads all.
    prefix : `str`
        Plotfile name prefix. A hybrid run usually writes a field-only
        diagnostic at high cadence -- point this at that one, since it is the
        only thing this reader needs.
    progress : `bool`
        Show a progress bar while reading.

    Returns
    -------
    `PICPhaseSpace`
        Electron phase space in SI, ready for `spectra_from_phase_spaces`.

    Raises
    ------
    `ValueError`
        If the deposited charge density is not positive, which means the run is
        not hybrid -- an explicit run carries electron macroparticles too, and
        its total ``rho`` is near zero.

    Notes
    -----
    The ions of a hybrid run are still macroparticles, so read them with
    `read_warpx_phase_space` as usual and pass both to the driver.

    What this reconstruction cannot do is invent the electron kinetics the model
    discarded: no Landau damping off a non-Maxwellian tail, no electron-scale
    structure. It gives the spectrum the model's own electron population
    implies, which is the right thing to compare against a kinetic twin.

    Examples
    --------
    .. code-block:: python

       electrons = read_warpx_hybrid_electrons(
           "runs/H3_470eV_eheat/diags",
           prefix="diag_fields",
           position=0.5 * u.mm,
       )
       ions = read_warpx_phase_space(
           "runs/H3_470eV_eheat/diags",
           "amb_ions",
           mass=100 * const.m_e,
           label="p+",
           prefix="diag_phase",
       )
    """
    path = Path(path)
    if drift not in {"ampere", "ion_current", "zero"}:
        raise ValueError(
            f"drift must be 'ampere', 'ion_current', or 'zero'; got {drift!r}."
        )
    if transverse_reduction not in {"slab", "chord"}:
        raise ValueError(
            f"transverse_reduction must be 'slab' or 'chord'; got "
            f"{transverse_reduction!r}."
        )

    yt = _load_yt()
    plotfiles = _warpx_plotfiles(path, prefix, timesteps)

    # --- work out what to read, from the first frame's geometry ---
    probe_dataset = yt.load(str(plotfiles[0]))
    probe, _, (left, right), spacing, resolved = _warpx_field_frame(
        yt, plotfiles[0], [density_field]
    )
    n_spatial = len(resolved)
    if not 0 <= axis < n_spatial:
        raise ValueError(
            f"axis must index one of the {n_spatial} resolved direction(s); got {axis}."
        )

    if scatter_direction is None:
        if n_spatial > 1:
            raise ValueError(
                f"this is a {n_spatial}-D run, so the scattering direction has "
                "to be given: pass scatter_direction as the unit vector k in "
                "simulation coordinates."
            )
        direction = np.zeros(3)
        direction[resolved[axis]] = 1.0
    else:
        direction = np.asarray(scatter_direction, dtype=np.float64).ravel()
        if direction.size != 3:
            raise ValueError(
                f"scatter_direction must be a 3-vector; got {direction.size} "
                "component(s)."
            )
        norm = np.linalg.norm(direction)
        if norm == 0:
            raise ValueError("scatter_direction must not be the zero vector.")
        direction = direction / norm

    # Exactly one of these is set below, and the rest of the reader keys off
    # which: the solved field when the run dumps one, the closure otherwise.
    names = [density_field]
    solved_temperature: str | None = None
    if (
        temperature_field is not None
        and _warpx_mesh_field(probe_dataset, temperature_field) is not None
    ):
        solved_temperature = temperature_field
        names.append(solved_temperature)
    elif closure is None:
        raise KeyError(
            f"the plotfiles carry no {temperature_field!r} field and no "
            "closure was given, so the electron temperature cannot be "
            "determined. Either add Te to the diagnostic's fields_to_plot, "
            "or pass closure={'T_e0': ..., 'n0_ref': ..., 'gamma': ...} "
            "from the deck's hybrid_pic_model block."
        )
    else:
        missing = {"T_e0", "n0_ref", "gamma"} - set(closure)
        if missing:
            raise ValueError(f"closure is missing {sorted(missing)}.")
        reference_density = float(_si_values(closure["n0_ref"], u.m**-3))
        closure_energy = _thermal_energy(closure["T_e0"] * temperature_unit)
        closure_exponent = float(closure["gamma"]) - 1.0

    current_components: dict[int, str] = {}
    magnetic_components: dict[int, str] = {}
    if drift != "zero":
        for component, letter in enumerate("xyz"):
            if direction[component] != 0.0:
                current_components[component] = f"{current_prefix}{letter}"
        if drift == "ampere":
            for component in _curl_needs(direction, resolved):
                magnetic_components[component] = f"{magnetic_prefix}{'xyz'[component]}"
        names.extend(current_components.values())
        names.extend(magnetic_components.values())

    # --- the scattering volume, in the transverse directions ---
    transverse_axes = [k for k in range(n_spatial) if k != axis]
    grid_shape = probe[density_field].shape
    centres = np.array(
        [0.5 * (left[resolved[k]] + right[resolved[k]]) for k in range(n_spatial)],
        dtype=np.float64,
    )
    if transverse_position is not None:
        supplied = np.asarray(transverse_position, dtype=np.float64).ravel()
        if supplied.size != len(transverse_axes):
            raise ValueError(
                f"transverse_position needs one coordinate per transverse axis "
                f"({len(transverse_axes)}); got {supplied.size}."
            )
        for value, k in zip(supplied, transverse_axes, strict=True):
            centres[k] = value
    half_width = (
        None if slab_halfwidth is None else float(_si_values(slab_halfwidth, u.m))
    )

    def cell_centres(k: int) -> np.ndarray:
        physical = resolved[k]
        step = spacing[physical]
        return left[physical] + step * (np.arange(grid_shape[k]) + 0.5)

    keep_slices: list[slice] = []
    for k in range(n_spatial):
        if k == axis or transverse_reduction == "chord":
            keep_slices.append(slice(None))
            continue
        width = half_width if half_width is not None else 0.5 * spacing[resolved[k]]
        lo, hi = _slab_indices(cell_centres(k), centres[k], width)
        keep_slices.append(slice(lo, hi))

    x_axis = cell_centres(axis)

    def reduce_to_profile(data: np.ndarray) -> np.ndarray:
        """Slab-average a field down to the diagnostic axis alone."""
        sliced = data[tuple(keep_slices)]
        # Average, not sum: these fields are intensive, and summing would scale
        # them by however many transverse cells happened to be kept.
        collapse = tuple(k for k in range(n_spatial) if k != axis)
        return sliced.mean(axis=collapse) if collapse else sliced

    # --- read every frame ---
    densities = []
    temperatures = []
    drifts = []
    times = []
    iterator = (
        tqdm(plotfiles, desc="Reading hybrid electron moments")
        if progress
        else plotfiles
    )
    elementary_charge = const.e.si.value
    for plotfile in iterator:
        fields, time, _, frame_spacing, frame_resolved = _warpx_field_frame(
            yt, plotfile, names
        )
        charge_density = reduce_to_profile(fields[density_field])
        density = charge_density / (charge_state * elementary_charge)

        if solved_temperature is not None:
            temperature = (
                reduce_to_profile(fields[solved_temperature]) * temperature_unit
            )
            thermal_energy = _thermal_energy(temperature)
        else:
            # Where there is no plasma the closure has nothing to say; leave the
            # ratio at one so the temperature stays finite and the zero density
            # carries the point.
            ratio = np.where(density > 0, density / reference_density, 1.0)
            thermal_energy = closure_energy * ratio**closure_exponent

        if drift == "zero":
            velocity = np.zeros_like(density)
        else:
            ion_current = sum(
                direction[component] * fields[name]
                for component, name in current_components.items()
            )
            if drift == "ampere" and magnetic_components:
                magnetic = {
                    component: fields[name]
                    for component, name in magnetic_components.items()
                }
                total_current = (
                    _curl_along(direction, magnetic, frame_spacing, frame_resolved)
                    / const.mu0.si.value
                )
            else:
                # Either the run resolves nothing that could make a total
                # current along k -- the exact 1-D axial case, where
                # (curl B).z vanishes identically -- or the caller asked for
                # that assumption directly.
                total_current = 0.0
            electron_current = reduce_to_profile(
                np.broadcast_to(
                    total_current - ion_current, fields[density_field].shape
                )
            )
            # J_e = n_e q_e u_e with q_e = -e.
            velocity = np.divide(
                -electron_current,
                elementary_charge * density,
                out=np.zeros_like(density),
                where=density > 0,
            )

        densities.append(density)
        temperatures.append(thermal_energy)
        drifts.append(velocity)
        times.append(time)

    density_block = np.stack(densities)
    if not np.any(density_block > 0):
        raise ValueError(
            f"{density_field!r} is nowhere positive in {path}. In a hybrid run "
            "only ions are macroparticles, so the deposited charge density is "
            "the electron density; a total rho near zero means the run carries "
            "electron macroparticles too and is not hybrid. Read its electrons "
            "with read_warpx_phase_space instead."
        )

    phase_space = from_moments(
        density_block,
        np.stack(temperatures) * u.J,
        np.stack(drifts),
        t=np.asarray(times, dtype=np.float64),
        x=x_axis,
        label=label,
        v=v,
        mass=const.m_e.si.value if mass is None else mass,
        is_electron=True,
        n_velocity_bins=n_velocity_bins,
        meta={
            "code": "WarpX (hybrid)",
            "path": str(path),
            "plotfiles": [p.name for p in plotfiles],
            "prefix": prefix,
            "n_spatial": n_spatial,
            "scatter_direction": direction.tolist(),
            "density_field": density_field,
            "charge_state": charge_state,
            "temperature_source": (
                solved_temperature
                if solved_temperature is not None
                else "barotropic closure"
            ),
            "closure": None if solved_temperature is not None else closure,
            "drift": drift,
            "current_fields": sorted(current_components.values()),
            "magnetic_fields": sorted(magnetic_components.values()),
            "transverse_reduction": transverse_reduction if transverse_axes else None,
            "slab_halfwidth": half_width,
        },
    )
    return phase_space if position is None else phase_space.at_position(position)


# ---------------------------------------------------------------------------
# Instrument response, output, and plotting
# ---------------------------------------------------------------------------

#: Ratio of a Gaussian's full width at half maximum to its standard deviation.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def _smooth_preserving_gaps(data, sigmas, kernel: str) -> np.ndarray:
    """
    Smooth *data* while leaving `~numpy.nan` gaps out of the average.

    A spectrogram has all-`~numpy.nan` rows wherever no plasma was present, and
    an ordinary filter would spread those over every point the kernel reaches,
    destroying good data. Filtering the values and the validity mask separately
    and then dividing gives the average over the valid samples alone.

    A gap is still *filled* from whatever valid data lies within the kernel's
    reach, which is what a real instrument integrating over a finite gate would
    do. Only where the kernel reaches no valid sample at all does the result
    stay `~numpy.nan`.
    """
    data = np.asarray(data, dtype=np.float64)
    valid = np.isfinite(data)
    filled = np.where(valid, data, 0.0)

    if kernel == "gaussian":
        numerator = gaussian_filter(filled, sigma=sigmas, mode="nearest")
        denominator = gaussian_filter(
            valid.astype(np.float64), sigma=sigmas, mode="nearest"
        )
    elif kernel == "boxcar":
        sizes = [max(1, round(sigma * FWHM_PER_SIGMA)) for sigma in sigmas]
        numerator = uniform_filter(filled, size=sizes, mode="nearest")
        denominator = uniform_filter(
            valid.astype(np.float64), size=sizes, mode="nearest"
        )
    else:
        raise ValueError(f"kernel must be 'gaussian' or 'boxcar'; got {kernel!r}.")

    return np.divide(
        numerator,
        denominator,
        out=np.full(data.shape, np.nan),
        where=denominator > 1e-6,
    )


def _sigma_in_bins(fwhm, axis: np.ndarray, unit: u.UnitBase) -> float:
    """Convert a full width at half maximum into Gaussian sigma, in bins."""
    if fwhm is None or axis.size < 2:
        return 0.0
    spacing = float(np.median(np.abs(np.diff(axis))))
    if spacing <= 0:
        return 0.0
    return float(_si_values(fwhm, unit)) / spacing / FWHM_PER_SIGMA
