"""
Defines the Thomson scattering analysis module as part of
`plasmapy.diagnostics`.
"""

__all__ = [
    "spectral_density",
    "spectral_density_arbitrary",
    "spectral_density_model",
]
__lite_funcs__ = ["spectral_density_lite", "spectral_density_arbitrary_lite"]

import warnings
from collections.abc import Callable
from typing import Any

import astropy.constants as const
import astropy.units as u
import numpy as np
from lmfit import Model

from plasmapy.formulary import (
    permittivity_1D_Maxwellian_lite,
    plasma_frequency_lite,
    thermal_speed_coefficients,
    thermal_speed_lite,
)
from plasmapy.particles import Particle, ParticleLike
from plasmapy.particles.exceptions import ChargeError
from plasmapy.particles.particle_collections import ParticleList
from plasmapy.utils.decorators import (
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __lite_funcs__

c_si_unitless = const.c.si.value
e_si_unitless = const.e.si.value
m_p_si_unitless = const.m_p.si.value
m_e_si_unitless = const.m_e.si.value


# TODO: interface for inputting a multi-species configuration could be  # noqa: FIX002
#     simplified using the plasmapy.classes.plasma_base class if that class
#     included ion and electron drift velocities and information about the ion
#     atomic species.


@preserve_signature
def spectral_density_lite(
    wavelengths,
    probe_wavelength: float,
    n: float,
    T_e: np.ndarray,
    T_i: np.ndarray,
    efract: np.ndarray,
    ifract: np.ndarray,
    ion_z: np.ndarray,
    ion_mass: np.ndarray,
    electron_vel: np.ndarray,
    ion_vel: np.ndarray,
    probe_vec: np.ndarray,
    scatter_vec: np.ndarray,
    instr_func_arr: np.ndarray | None = None,
    notch: np.ndarray | None = None,
) -> tuple[np.floating | np.ndarray, np.ndarray]:
    r"""
    The :term:`lite-function` version of
    `~plasmapy.diagnostics.thomson.spectral_density`.  Performs the same
    thermal speed calculations as
    `~plasmapy.diagnostics.thomson.spectral_density`, but is intended
    for computational use and thus has data conditioning safeguards
    removed.

    Parameters
    ----------
    wavelengths : (Nλ,) `~numpy.ndarray`
        The wavelengths in meters over which the spectral density
        function will be calculated.

    probe_wavelength : real number
        Wavelength of the probe laser in meters.

    n : `~numpy.ndarray`
        Total combined number density of all electron populations in
        m\ :sup:`-3`\ .

    T_e : (Ne,) `~numpy.ndarray`
        Temperature of each electron population in kelvin, where Ne is
        the number of electron populations.

    T_i : (Ni,) `~numpy.ndarray`
        Temperature of each ion population in kelvin, where Ni is the
        number of ion populations.

    efract : (Ne,) `~numpy.ndarray`
        An `~numpy.ndarray` where each element represents the fraction
        (or ratio) of the electron population number density to the
        total electron number density. Must sum to 1.0. Default is a
        single electron population.

    ifract : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` object where each element represents the
        fraction (or ratio) of the ion population number density to the
        total ion number density. Must sum to 1.0. Default is a single
        ion species.

    ion_z : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the charge number :math:`Z` of each ion
        species.

    ion_mass : (Ni,) `~numpy.ndarray`
        An `~numpy.ndarray` of the mass number of each ion species in kg.

    electron_vel : (Ne, 3) `~numpy.ndarray`
        Velocity of each electron population in the rest frame (in m/s).
        If set, overrides ``electron_vdir`` and ``electron_speed``.
        Defaults to a stationary plasma ``[0, 0, 0]`` m/s.

    ion_vel : (Ni, 3) `~numpy.ndarray`
        Velocity vectors for each electron population in the rest frame
        (in  m/s). If set, overrides ``ion_vdir`` and ``ion_speed``.
        Defaults to zero drift for all specified ion species.

    probe_vec : (3,) float `~numpy.ndarray`
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : (3,) float `~numpy.ndarray`
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default ``probe_vec``,
        corresponds to a 90 degree scattering angle geometry.

    instr_func_arr : `~numpy.ndarray`, shape (Nwavelengths,) optional
        The instrument function evaluated at a linearly spaced range of
        wavelengths ranging from :math:`-W` to :math:`W`, where

        .. math::
            W = 0.5*(\max{λ} - \min{λ})

        Here :math:`λ` is the ``wavelengths`` array. This array will be
        convolved with the spectral density function before it is
        returned.

    notch : (2,) or (N, 2) `~numpy.ndarray`, |keyword-only|, optional
        A pair of wavelengths in meters which are the endpoints of a notch over
        which the output Skw is set to 0. Can also be input as a 2D array
        which contains many such pairs if multiple notches are needed.
        Defaults to no notch.

    Returns
    -------
    alpha : float
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates non-collective
        scattering. The scattering parameter is calculated based on the
        total plasma density :math:`n`.

    Skw : `~numpy.ndarray`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.
    """
    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))

    # Calculate plasma parameters
    # Temperatures here in K!
    coefs = thermal_speed_coefficients("most_probable", 3)
    vT_e = thermal_speed_lite(T_e, m_e_si_unitless, coefs)
    vT_i = thermal_speed_lite(T_i, ion_mass, coefs)

    # Compute electron and ion densities
    ne = efract * n
    zbar = np.sum(ifract * ion_z)
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    wpe = plasma_frequency_lite(n, m_e_si_unitless, 1)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * c_si_unitless / wavelengths
    wl = 2 * np.pi * c_si_unitless / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws**2 - wpe**2) / c_si_unitless
    kl = np.sqrt(wl**2 - wpe**2) / c_si_unitless

    # Compute the wavenumber shift (required by momentum conservation)
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks**2 + kl**2 - 2 * ks * kl * np.cos(scattering_angle))
    # Normal vector along k
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot products over all wavelengths
    # and ion populations
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vT_e)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xe = np.outer(1 / vT_e, 1 / k) * w_e
    xi = np.outer(1 / vT_i, 1 / k) * w_i

    # Calculate the susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(efract):
        wpe = plasma_frequency_lite(ne[i], m_e_si_unitless, 1)
        chiE[i, :] = permittivity_1D_Maxwellian_lite(w_e[i, :], k, vT_e[i], wpe)

    # Treatment of multiple species is an extension of the discussion in
    # Sheffield Sec. 5.1
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i, _fract in enumerate(ifract):
        wpi = plasma_frequency_lite(ni[i], ion_mass[i], ion_z[i])
        chiI[i, :] = permittivity_1D_Maxwellian_lite(w_i[i, :], k, vT_i[i], wpi)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m, :] = efract[m] * (
            2
            * np.sqrt(np.pi)
            / k
            / vT_e[m]
            * np.power(np.abs(1 - np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-(xe[m, :] ** 2))
        )

    icontr = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for m in range(ifract.size):
        icontr[m, :] = ifract[m] * (
            2
            * np.sqrt(np.pi)
            * ion_z[m] ** 2
            / zbar
            / k
            / vT_i[m]
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-(xi[m, :] ** 2))
        )

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Apply an instrument function if one is provided
    if instr_func_arr is not None:
        Skw = np.convolve(Skw, instr_func_arr, mode="same")

    # add notch(es) to the spectrum if any are provided
    if notch is not None:
        # If only one notch is included, create a dummy second dimension
        if np.ndim(notch) == 1:
            notch = np.array(
                [
                    notch,
                ],
            )

        for notch_i in notch:
            # For each notch, identify the index for the beginning and end
            # wavelengths and set Skw to zero between those indices
            x0 = np.argmin(np.abs(wavelengths - notch_i[0]))
            x1 = np.argmin(np.abs(wavelengths - notch_i[1]))
            Skw[x0:x1] = 0

    return np.mean(alpha), Skw


@validate_quantities(
    wavelengths={"can_be_negative": False, "can_be_zero": False},
    probe_wavelength={"can_be_negative": False, "can_be_zero": False},
    n={"can_be_negative": False, "can_be_zero": False},
    T_e={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    T_i={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
@bind_lite_func(spectral_density_lite)
def spectral_density(  # noqa: C901, PLR0912, PLR0915
    wavelengths: u.Quantity[u.nm],
    probe_wavelength: u.Quantity[u.nm],
    n: u.Quantity[u.m**-3],
    *,
    T_e: u.Quantity[u.K],
    T_i: u.Quantity[u.K],
    efract=None,
    ifract=None,
    ions: ParticleLike = "p+",
    electron_vel: u.Quantity[u.m / u.s] = None,
    ion_vel: u.Quantity[u.m / u.s] = None,
    probe_vec=None,
    scatter_vec=None,
    instr_func: Callable | None = None,
    notch: u.m = None,
) -> tuple[np.floating | np.ndarray, np.ndarray]:
    r"""Calculate the spectral density function for Thomson scattering of
    a probe laser beam by a multi-species Maxwellian plasma.

    Parameters
    ----------
    wavelengths : `~astropy.units.Quantity`
        The wavelengths over which the spectral density function will be
        calculated, in units convertible to m.

    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser, in units convertible to m.

    n : `~astropy.units.Quantity`
        Total combined number density of all electron populations, in
        units convertible to m\ :sup:`-3`\ .

    T_e : (Ne,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each electron population in units convertible to
        K or eV, where Ne is the number of electron populations.

    T_i : (Ni,) `~astropy.units.Quantity`, |keyword-only|
        Temperature of each ion population in units convertible to K or
        eV, where Ni is the number of ion populations.

    efract : (Ne,) |array_like|, |keyword-only|, optional
        The ratio of the number density of each electron population to
        the total electron number density, denoted by :math:`F_e` below.
        Must sum to one. The default corresponds to a single electron
        population.

    ifract : (Ni,) |array_like|, |keyword-only|, optional
        The fractional number densities of each ion population, denoted
        by :math:`F_i` below. Must sum to one. The default corresponds
        to a single ion population.

    ions : (Ni,) |particle-like|, |keyword-only|, default: "p+"
        One or more positively charged ions representing each ion
        population.

    electron_vel : (Ne, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each electron population in the rest frame,
        in units convertible to m/s. If set, overrides ``electron_vdir``
        and ``electron_speed``.  Defaults to a stationary plasma at
        :math:`[0, 0, 0]` m/s.

    ion_vel : (Ni, 3) `~astropy.units.Quantity`, |keyword-only|, optional
        Velocity vectors for each ion population in the rest frame, in
        units convertible to m/s. If set, overrides ``ion_vdir`` and
        ``ion_speed``. Defaults to zero drift for all specified ion
        species.

    probe_vec : (3,) |array_like|, |keyword-only|, default: [1, 0, 0]
        Unit vector in the direction of the probe laser.

    scatter_vec : (3,) |array_like|, |keyword-only|, default: [0, 1, 0]
        Unit vector pointing from the scattering volume to the
        detector. The default, along with the default for ``probe_vec``,
        corresponds to a 90° scattering angle geometry.

    instr_func : function

        A function representing the instrument function that takes a
        `~astropy.units.Quantity` of wavelengths (centered on zero)
        and returns the instrument point spread function. The
        resulting array will be convolved with the spectral density
        function before it is returned.

    notch : (2,) or (N, 2) `~astropy.units.Quantity`, |keyword-only|, optional
        A pair of wavelengths in units convertible to meters which are the
        endpoints of a notch over which the output Skw is set to 0. Can also
        be input as a 2D array which contains many such pairs if multiple
        notches are needed. If the ``notch`` and ``instr_func`` keywords are both
        set, the notch is applied after the instrument function such the
        instrument function does convolve the values of the theoretical
        spectrum originally in the notch region. Defaults to no notch.


    Returns
    -------
    alpha : `float`
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering and ``alpha`` < 1 indicates
        non-collective scattering. The scattering parameter is
        calculated based on the total plasma density ``n``.

    Skw : `~astropy.units.Quantity`
        Computed spectral density function over the input
        ``wavelengths`` array with units of s/rad.

    Notes
    -----
    This function calculates the spectral density function for Thomson
    scattering of a probe laser beam by a plasma consisting of one or
    more ion species and one or more thermal electron populations (the
    entire plasma is assumed to be quasi-neutral):

    .. math::
        S(k,ω) = \sum_e \frac{2π}{k}
        \bigg |1 - \frac{χ_e}{ε} \bigg |^2
        f_{e0,e} \bigg (\frac{ω}{k} \bigg ) +
        \sum_i \frac{2π}{k} \frac{Z_i^2}{\bar Z}
        \bigg |\frac{χ_e}{ε} \bigg |^2 f_{i0,i}
        \bigg ( \frac{ω}{k} \bigg )

    where :math:`χ_e` is the electron population susceptibility of the
    plasma and :math:`ε = 1 + ∑_e χ_e + ∑_i χ_i` is the total plasma
    dielectric function (with :math:`χ_i` being the ion population of
    the susceptibility),
    :math:`k` is the scattering wavenumber, :math:`ω` is the scattering
    frequency, and :math:`f_{e0,e}` and :math:`f_{i0,i}` are the
    electron and ion velocity distribution functions, respectively. In
    this function, the electron and ion velocity distribution functions
    are assumed to be Maxwellian, making this function equivalent to Eq.
    3.4.6 in :cite:t:`sheffield:2011`\ .

    The number density of the e\ :sup:`th` electron populations is
    defined as

    .. math::
        n_e = F_e n

    where :math:`n` is the total number density of all electron
    populations combined and :math:`F_e` is the fractional number
    density of each electron population such that

    .. math::
        \sum_e n_e = n

    .. math::
        \sum_e F_e = 1

    and

    .. math::
        \bar Z = \sum_k F_k Z_k

    The plasma is assumed to be quasineutral, and therefore the number
    density of the i\ :sup:`th` ion population is

    .. math::
        n_i = \frac{F_i n}{∑_i F_i Z_i}

    with :math:`F_i` defined in the same way as :math:`F_e`.

    For details, see "Plasma Scattering of Electromagnetic Radiation"
    by :cite:t:`sheffield:2011`. This code is a modified version of
    the program described therein.

    For a summary of the relevant physics, see Chapter 5 of the
    :cite:t:`schaeffer:2014` thesis.

    """
    # Validate efract
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)
        if np.sum(efract) != 1:
            raise ValueError(f"The provided efract does not sum to 1: {efract}")

    # Validate ifract
    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)
        if np.sum(ifract) != 1:
            raise ValueError(f"The provided ifract does not sum to 1: {ifract}")

    if probe_vec is None:
        probe_vec = np.array([1, 0, 0])

    if scatter_vec is None:
        scatter_vec = np.array([0, 1, 0])

    # If electron velocity is not specified, create an array corresponding
    # to zero drift
    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3]) * u.m / u.s

    # Condition the electron velocity keywords
    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3]) * u.m / u.s

    # Condition ions
    # If a single value is provided, turn into a particle list
    if isinstance(ions, ParticleList):
        pass
    elif isinstance(ions, str):
        ions = ParticleList([Particle(ions)])  # ty:ignore[invalid-assignment]
    # If a list is provided, ensure all values are Particles, then convert
    # to a ParticleList
    elif isinstance(ions, list):
        for ii, ion in enumerate(ions):
            if isinstance(ion, Particle):
                continue
            ions[ii] = Particle(ion)  # ty:ignore[invalid-assignment, invalid-argument-type]
        ions = ParticleList(ions)  # ty:ignore[invalid-assignment]
    else:
        raise TypeError(
            "The type of object provided to the ``ions`` keyword "
            f"is not supported: {type(ions)}",
        )

    # Validate ions
    if len(ions) == 0:  # ty:ignore[invalid-argument-type]
        raise ValueError("At least one ion species needs to be defined.")

    try:
        if sum(ion.charge_number <= 0 for ion in ions):  # ty:ignore[not-iterable]
            raise ValueError("All ions must be positively charged.")
    # Catch error if charge information is missing
    except ChargeError as ex:
        raise ValueError("All ions must be positively charged.") from ex

    # Condition T_i
    if T_i.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_i.size != len(ions), assume same temp. for all species
        T_i = np.array([T_i.value]) * T_i.unit

    # Make sure the sizes of ions, ifract, ion_vel, and T_i all match
    if (
        (len(ions) != ifract.size)  # ty:ignore[invalid-argument-type]
        or (ion_vel.shape[0] != ifract.size)
        or (T_i.size != ifract.size)
    ):
        raise ValueError(
            f"Inconsistent number of ion species in ifract ({ifract}), "
            f"ions ({len(ions)}), T_i ({T_i.size}), "  # ty:ignore[invalid-argument-type]
            f"and/or ion_vel ({ion_vel.shape[0]}).",
        )

    # Condition T_e
    if T_e.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If T_e.size != len(efract), assume same temp. for all species
        T_e = np.array([T_e.value]) * T_e.unit

    # Make sure the sizes of efract, electron_vel, and T_e all match
    if (electron_vel.shape[0] != efract.size) or (T_e.size != efract.size):
        raise ValueError(
            f"Inconsistent number of electron populations in efract ({efract.size}), "
            f"T_e ({T_e.size}), or electron velocity ({electron_vel.shape[0]}).",
        )

    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Apply the instrument function
    if instr_func is not None and callable(instr_func):
        # Create an array of wavelengths of the same size as wavelengths
        # but centered on zero
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w)

        if type(instr_func_arr) is not np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}",
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape}).",
            )

        instr_func_arr /= np.sum(instr_func_arr)
    else:
        instr_func_arr = None

    # Valildate notch input
    if notch is not None:
        notch_unitless = notch.to(u.m).value

        if np.ndim(notch_unitless) == 1:
            notch_unitless = np.array([notch_unitless])

        for notch_i in notch_unitless:
            if np.shape(notch_i) != (2,):
                raise ValueError("Notches must be pairs of values.")
            if notch_i[0] > notch_i[1]:
                raise ValueError(
                    "The first element of the notch cannot be greater than "
                    "the second element.",
                )
    else:
        notch_unitless = None

    alpha, Skw = spectral_density_lite(
        wavelengths.to(u.m).value,
        probe_wavelength.to(u.m).value,
        n.to(u.m**-3).value,
        T_e.to(u.K).value,
        T_i.to(u.K).value,
        efract=efract,
        ifract=ifract,
        ion_z=ions.charge_number,
        ion_mass=ions.mass.to(u.kg).value,
        ion_vel=ion_vel.to(u.m / u.s).value,
        electron_vel=electron_vel.to(u.m / u.s).value,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
        notch=notch_unitless,
    )

    return alpha, Skw * u.s / u.rad


def _finite_difference_derivative(
    f: np.ndarray, x: np.ndarray, order: int
) -> np.ndarray:
    r"""
    Fourth-order finite-difference derivative of ``f`` sampled on the
    uniformly spaced grid ``x``.

    Parameters
    ----------
    f : `~numpy.ndarray`
        Function values sampled at the points in ``x``.

    x : `~numpy.ndarray`
        Uniformly spaced sample points.

    order : `int`
        The order of the derivative to compute; either ``1`` or ``2``.

    Returns
    -------
    `~numpy.ndarray`
        The ``order``\ -th derivative of ``f`` with respect to ``x``,
        evaluated at the points in ``x``.
    """
    dx = x[1] - x[0]
    f_prime = np.empty(len(f))

    if order == 1:
        # 4th-order one-sided differences for the two points at each end
        f_prime[0] = (
            -25 / 12 * f[0] + 4 * f[1] - 3 * f[2] + 4 / 3 * f[3] - f[4] / 4
        ) / dx
        f_prime[1] = (
            -25 / 12 * f[1] + 4 * f[2] - 3 * f[3] + 4 / 3 * f[4] - f[5] / 4
        ) / dx
        f_prime[-1] = (
            f[-5] / 4 - 4 / 3 * f[-4] + 3 * f[-3] - 4 * f[-2] + 25 / 12 * f[-1]
        ) / dx
        f_prime[-2] = (
            f[-6] / 4 - 4 / 3 * f[-5] + 3 * f[-4] - 4 * f[-3] + 25 / 12 * f[-2]
        ) / dx
        # 4th-order centered difference for the interior
        f_prime[2:-2] = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * dx)
    elif order == 2:
        f_prime[0] = (
            15 / 4 * f[0]
            - 77 / 6 * f[1]
            + 107 / 6 * f[2]
            - 13 * f[3]
            + 61 / 12 * f[4]
            - 5 / 6 * f[5]
        ) / dx**2
        f_prime[1] = (
            15 / 4 * f[1]
            - 77 / 6 * f[2]
            + 107 / 6 * f[3]
            - 13 * f[4]
            + 61 / 12 * f[5]
            - 5 / 6 * f[6]
        ) / dx**2
        f_prime[-1] = (
            15 / 4 * f[-1]
            - 77 / 6 * f[-2]
            + 107 / 6 * f[-3]
            - 13 * f[-4]
            + 61 / 12 * f[-5]
            - 5 / 6 * f[-6]
        ) / dx**2
        f_prime[-2] = (
            15 / 4 * f[-2]
            - 77 / 6 * f[-3]
            + 107 / 6 * f[-4]
            - 13 * f[-5]
            + 61 / 12 * f[-6]
            - 5 / 6 * f[-7]
        ) / dx**2
        f_prime[2:-2] = (
            -f[:-4] / 12
            + 4 / 3 * f[1:-3]
            - 5 / 2 * f[2:-2]
            + 4 / 3 * f[3:-1]
            - f[4:] / 12
        ) / dx**2
    else:
        raise ValueError("``order`` must be either 1 or 2.")

    return f_prime


def _chi_arbitrary(
    f: np.ndarray,
    u_axis: np.ndarray,
    k: np.ndarray,
    xi: np.ndarray,
    v_th: float,
    n: float,
    mass: float,
    Z: float,
    phi: float = 1e-5,
    n_points: int = 1000,
    inner_range: float = 0.1,
    inner_frac: float = 0.8,
) -> np.ndarray:
    r"""
    Longitudinal susceptibility :math:`χ(k, ω)` of a single species with an
    arbitrary one-dimensional velocity distribution function.

    This evaluates the susceptibility integral (Sheffield Eq. 3.3.9) by
    numerically computing the principal value of

    .. math::
        χ = -\frac{ω_p^2}{2 k^2 v_{th}^2}
            \int \frac{∂f/∂u}{u - ξ}\, du

    using a quadrature that is refined near the :math:`u = ξ` pole, plus the
    Landau residue contribution :math:`i π\, f'(ξ)`.

    Parameters
    ----------
    f : `~numpy.ndarray`
        The (normalized) velocity distribution function sampled on
        ``u_axis``.

    u_axis : `~numpy.ndarray`
        Velocity axis normalized by :math:`\sqrt{2}\, v_{th}` and centered
        on the distribution's bulk velocity.

    k : `~numpy.ndarray`
        Scattering wavenumbers in radians per meter.

    xi : `~numpy.ndarray`
        Normalized phase velocities :math:`ω / (\sqrt{2}\, k\, v_{th})`.

    v_th : `float`
        Thermal speed of the distribution (its standard deviation in m/s).

    n : `float`
        Number density of the species in m\ :sup:`-3`.

    mass : `float`
        Mass of the species in kg.

    Z : `float`
        Charge number of the species.

    phi : `float`, optional
        Small standoff offset used to avoid the integrand singularity.

    n_points : `int`, optional
        Number of sample points used for the principal-value integral.

    inner_range, inner_frac : `float`, optional
        Fraction of the integration range and of the sample points,
        respectively, devoted to the finely sampled region near the pole.

    Returns
    -------
    `~numpy.ndarray`
        The complex susceptibility evaluated at each value in ``xi``.
    """
    f_prime = _finite_difference_derivative(f, u_axis, 1)
    f_double_prime = _finite_difference_derivative(f, u_axis, 2)

    # f' and f'' interpolated onto the phase velocities
    g = np.interp(xi, u_axis, f_prime)
    g_prime = np.interp(xi, u_axis, f_double_prime)

    # Set up integration sample points, finely spaced near the pole
    outer_frac = 1 - inner_frac
    m_inner = np.linspace(0, inner_range, int(np.floor(n_points / 2 * inner_frac)))
    p_inner = np.linspace(0, inner_range, int(np.ceil(n_points / 2 * inner_frac)))
    m_outer = np.linspace(inner_range, 1, int(np.floor(n_points / 2 * outer_frac)))
    p_outer = np.linspace(inner_range, 1, int(np.ceil(n_points / 2 * outer_frac)))
    m = np.concatenate((m_inner, m_outer))
    p = np.concatenate((p_inner, p_outer))

    delta_u_max = np.max(u_axis) - np.min(u_axis)
    m_points = phi + m * delta_u_max
    p_points = phi + p * delta_u_max
    m_deltas = np.append(m_points[1:] - m_points[:-1], 0)
    p_deltas = np.append(p_points[1:] - p_points[:-1], 0)

    # Offsets above (zm) and below (zp) each phase velocity
    zm = xi[:, np.newaxis] + m_points[np.newaxis, :]
    zp = xi[:, np.newaxis] - p_points[np.newaxis, :]
    gm = np.interp(zm, u_axis, f_prime)
    gp = np.interp(zp, u_axis, f_prime)

    integral = (
        np.sum(m_deltas * gm / m_points, axis=1)
        - np.sum(p_deltas * gp / p_points, axis=1)
        + 1j * np.pi * g
        + 2 * phi * g_prime
    )

    wpl_sq = plasma_frequency_lite(n, mass, Z) ** 2
    coefficient = -wpl_sq / k**2 / (np.sqrt(2) * v_th)
    return coefficient * integral


@preserve_signature
def spectral_density_arbitrary_lite(  # noqa: C901, PLR0915
    wavelengths,
    probe_wavelength: float,
    n: float,
    e_velocity_axes: np.ndarray,
    i_velocity_axes: np.ndarray,
    efn: np.ndarray,
    ifn: np.ndarray,
    efract: np.ndarray,
    ifract: np.ndarray,
    ion_z: np.ndarray,
    ion_mass: np.ndarray,
    probe_vec: np.ndarray,
    scatter_vec: np.ndarray,
    notch: np.ndarray | None = None,
    inner_range: float = 0.1,
    inner_frac: float = 0.8,
) -> tuple[np.floating, np.ndarray]:
    r"""
    The :term:`lite-function` version of
    `~plasmapy.diagnostics.thomson.spectral_density_arbitrary`.  Performs the
    same calculation but is intended for computational use and thus has data
    conditioning safeguards removed.

    Unlike
    `~plasmapy.diagnostics.thomson.spectral_density_lite`, which assumes
    Maxwellian populations, this function accepts arbitrary one-dimensional
    velocity distribution functions for each electron and ion population.
    The bulk velocity and thermal speed of each population are computed as
    moments of the supplied distributions.

    Parameters
    ----------
    wavelengths : (Nλ,) `~numpy.ndarray`
        The wavelengths in meters over which the spectral density function
        is calculated.

    probe_wavelength : real number
        Wavelength of the probe laser in meters.

    n : real number
        Total combined number density of all electron populations in
        m\ :sup:`-3`.

    e_velocity_axes : (Ne, Nv) `~numpy.ndarray`
        The velocity axis in m/s for each electron population's
        distribution function.

    i_velocity_axes : (Ni, Nv) `~numpy.ndarray`
        The velocity axis in m/s for each ion population's distribution
        function.

    efn : (Ne, Nv) `~numpy.ndarray`
        The distribution function of each electron population, sampled on
        the corresponding row of ``e_velocity_axes`` and normalized so that
        its integral over velocity is unity (units of s/m).

    ifn : (Ni, Nv) `~numpy.ndarray`
        As ``efn``, but for each ion population.

    efract : (Ne,) `~numpy.ndarray`
        Fraction of the total electron number density in each electron
        population; must sum to 1.

    ifract : (Ni,) `~numpy.ndarray`
        Fraction of the total ion number density in each ion population;
        must sum to 1.

    ion_z : (Ni,) `~numpy.ndarray`
        Charge number :math:`Z` of each ion species.

    ion_mass : (Ni,) `~numpy.ndarray`
        Mass of each ion species in kg.

    probe_vec : (3,) `~numpy.ndarray`
        Unit vector in the direction of the probe laser.

    scatter_vec : (3,) `~numpy.ndarray`
        Unit vector pointing from the scattering volume to the detector.

    notch : (2,) or (N, 2) `~numpy.ndarray`, optional
        Pairs of wavelengths in meters between which the output is set to
        zero. Defaults to no notch.

    inner_range, inner_frac : `float`, optional
        Quadrature-refinement controls passed to the susceptibility
        integral; see
        `~plasmapy.diagnostics.thomson.spectral_density_arbitrary`.

    Returns
    -------
    alpha : `float`
        Mean scattering parameter over the input wavelengths.

    Skw : `~numpy.ndarray`
        The spectral density function, normalized so that its integral over
        ``wavelengths`` is unity.
    """
    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Normal vector along k; all velocities are assumed to lie along it
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Bulk velocity and thermal speed (standard deviation) of each population
    # are obtained as moments of the supplied distribution functions.
    def _moments(velocity_axes, fns):
        drift_1d = np.empty(len(fns))
        v_th = np.empty(len(fns))
        for i, fn in enumerate(fns):
            v_axis = velocity_axes[i]
            bulk = np.trapezoid(fn * v_axis, v_axis)
            drift_1d[i] = bulk
            v_th[i] = np.sqrt(np.trapezoid(fn * (v_axis - bulk) ** 2, v_axis))
        drift = np.outer(drift_1d, k_vec)
        return drift, drift_1d, v_th

    electron_vel, electron_vel_1d, vTe = _moments(e_velocity_axes, efn)
    ion_vel, ion_vel_1d, vTi = _moments(i_velocity_axes, ifn)

    # Plasma parameters
    zbar = np.sum(ifract * ion_z)
    ne = efract * n
    ni = ifract * n / zbar  # ne / zbar = sum(ni)
    wpe = plasma_frequency_lite(n, m_e_si_unitless, 1)

    # Convert wavelengths to angular frequencies (phase speed c)
    ws = 2 * np.pi * c_si_unitless / wavelengths
    wl = 2 * np.pi * c_si_unitless / probe_wavelength
    w = ws - wl  # frequency shift (energy conservation)

    # Wavenumbers in the plasma (Sheffield Sec. 1.8.1, Eqs. 5.4.1-5.4.2)
    ks = np.sqrt(ws**2 - wpe**2) / c_si_unitless
    kl = np.sqrt(wl**2 - wpe**2) / c_si_unitless
    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))
    k = np.sqrt(ks**2 + kl**2 - 2 * ks * kl * np.cos(scattering_angle))

    # Doppler-shifted frequencies for each population
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Scattering parameter alpha = 1 / (k * lambda_De), where the Debye
    # length lambda_De = sigma / wpe and ``vTe`` is the distribution's
    # standard deviation sigma.
    alpha = wpe / np.outer(k, vTe)

    # Normalized phase velocities, xi = w / (sqrt(2) k v_th)
    xie = (np.outer(1 / vTe, 1 / k) * w_e) / np.sqrt(2)
    xii = (np.outer(1 / vTi, 1 / k) * w_i) / np.sqrt(2)

    # Susceptibilities from the arbitrary-distribution integral
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i in range(efract.size):
        u_axis = (e_velocity_axes[i] - electron_vel_1d[i]) / (np.sqrt(2) * vTe[i])
        chiE[i, :] = _chi_arbitrary(
            efn[i],
            u_axis,
            k,
            xie[i],
            vTe[i],
            ne[i],
            m_e_si_unitless,
            1,
            inner_range=inner_range,
            inner_frac=inner_frac,
        )

    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i in range(ifract.size):
        u_axis = (i_velocity_axes[i] - ion_vel_1d[i]) / (np.sqrt(2) * vTi[i])
        chiI[i, :] = _chi_arbitrary(
            ifn[i],
            u_axis,
            k,
            xii[i],
            vTi[i],
            ni[i],
            ion_mass[i],
            ion_z[i],
            inner_range=inner_range,
            inner_frac=inner_frac,
        )

    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    # Electron and ion contributions to S(k, w) (Sheffield Sec. 5.1), with
    # the distribution function evaluated at the phase velocity.
    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for j in range(efract.size):
        u_axis = (e_velocity_axes[j] - electron_vel_1d[j]) / (np.sqrt(2) * vTe[j])
        econtr[j] = efract[j] * (
            2
            * np.pi
            / k
            * np.power(np.abs(1 - np.sum(chiE, axis=0) / epsilon), 2)
            * np.interp(xie[j], u_axis, efn[j])
        )

    icontr = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for j in range(ifract.size):
        u_axis = (i_velocity_axes[j] - ion_vel_1d[j]) / (np.sqrt(2) * vTi[j])
        icontr[j] = ifract[j] * (
            2
            * np.pi
            * ion_z[j]
            / k
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.interp(xii[j], u_axis, ifn[j])
        )

    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Notches
    if notch is not None:
        if np.ndim(notch) == 1:
            notch = np.array([notch])
        for notch_i in notch:
            if len(notch_i) != 2:
                raise ValueError("Notches must be pairs of wavelengths.")
            x0 = np.argmin(np.abs(wavelengths - notch_i[0]))
            x1 = np.argmin(np.abs(wavelengths - notch_i[1]))
            Skw[x0:x1] = 0

    # Normalize so the spectrum integrates to unity over wavelength
    Skw = Skw / np.trapezoid(Skw, wavelengths)

    return np.mean(alpha), Skw


@bind_lite_func(spectral_density_arbitrary_lite)
def spectral_density_arbitrary(
    wavelengths: u.Quantity[u.nm],
    probe_wavelength: u.Quantity[u.nm],
    n: u.Quantity[u.m**-3],
    e_velocity_axes: u.Quantity[u.m / u.s],
    i_velocity_axes: u.Quantity[u.m / u.s],
    efn: u.Quantity[u.s / u.m],
    ifn: u.Quantity[u.s / u.m],
    *,
    efract: np.ndarray | None = None,
    ifract: np.ndarray | None = None,
    ions: ParticleLike | list[ParticleLike] = "p+",
    probe_vec=None,
    scatter_vec=None,
    notch: u.Quantity[u.nm] | None = None,
    inner_range: float = 0.1,
    inner_frac: float = 0.8,
) -> tuple[np.floating, u.Quantity]:
    r"""
    Calculate the Thomson scattering spectral density for plasmas with
    arbitrary velocity distribution functions.

    **Lite Version:**
    `~plasmapy.diagnostics.thomson.spectral_density_arbitrary_lite`

    This is a generalization of
    `~plasmapy.diagnostics.thomson.spectral_density` that accepts an
    arbitrary (not necessarily Maxwellian) one-dimensional velocity
    distribution function for each electron and ion population.  The bulk
    velocity and thermal speed of each population are computed as moments of
    the supplied distributions, and the longitudinal susceptibilities are
    obtained by direct numerical integration rather than from the analytic
    Maxwellian form.

    Parameters
    ----------
    wavelengths : (Nλ,) `~astropy.units.Quantity`
        The wavelengths over which the spectral density function is
        calculated, convertible to nanometers.

    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser, convertible to nanometers.

    n : `~astropy.units.Quantity`
        Total combined number density of all electron populations,
        convertible to m\ :sup:`-3`.

    e_velocity_axes : (Ne, Nv) `~astropy.units.Quantity`
        Velocity axis (in m/s) for each electron population's distribution
        function.

    i_velocity_axes : (Ni, Nv) `~astropy.units.Quantity`
        Velocity axis (in m/s) for each ion population's distribution
        function.

    efn : (Ne, Nv) `~astropy.units.Quantity`
        Distribution function of each electron population, sampled on the
        corresponding row of ``e_velocity_axes`` and normalized so that its
        integral over velocity is unity (units of s/m).

    ifn : (Ni, Nv) `~astropy.units.Quantity`
        As ``efn``, but for each ion population.

    efract : (Ne,) array_like, optional
        Fraction of the total electron number density in each population;
        must sum to 1. Defaults to a single population.

    ifract : (Ni,) array_like, optional
        Fraction of the total ion number density in each population; must
        sum to 1. Defaults to a single population.

    ions : |particle-list-like|, optional
        Ion species. Defaults to a single proton population (``"p+"``).

    probe_vec : (3,) `~numpy.ndarray`, optional
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : (3,) `~numpy.ndarray`, optional
        Unit vector pointing from the scattering volume to the detector.
        Defaults to ``[0, 1, 0]``.

    notch : (2,) or (N, 2) `~astropy.units.Quantity`, optional
        Pairs of wavelengths between which the output is set to zero.

    inner_range, inner_frac : `float`, optional
        Controls for the susceptibility quadrature: the fraction of the
        integration range and of the sample points, respectively, devoted
        to the finely sampled region near the :math:`u = ξ` pole.

    Returns
    -------
    alpha : `float`
        Mean scattering parameter, where ``alpha`` > 1 corresponds to
        collective scattering.

    Skw : `~astropy.units.Quantity`
        The spectral density function, normalized so that its integral over
        ``wavelengths`` is unity (units of 1/m).

    See Also
    --------
    spectral_density
    """
    efract = np.ones(1) if efract is None else np.asarray(efract, dtype=np.float64)
    ifract = np.ones(1) if ifract is None else np.asarray(ifract, dtype=np.float64)

    if probe_vec is None:
        probe_vec = np.array([1, 0, 0])
    if scatter_vec is None:
        scatter_vec = np.array([0, 1, 0])

    if isinstance(ions, str | Particle):
        ions = [ions]
    ions = ParticleList(
        [Particle(ion) if not isinstance(ion, Particle) else ion for ion in ions]
    )
    if len(ions) == 0:
        raise ValueError("At least one ion species must be provided.")

    ion_z = np.array([ion.charge_number for ion in ions])
    ion_mass = np.array([ion.mass.to(u.kg).value for ion in ions])

    notch_si = None if notch is None else notch.to(u.m).value

    alpha, Skw = spectral_density_arbitrary_lite(
        wavelengths=wavelengths.to(u.m).value,
        probe_wavelength=probe_wavelength.to(u.m).value,
        n=n.to(u.m**-3).value,
        e_velocity_axes=np.atleast_2d(e_velocity_axes.to(u.m / u.s).value),
        i_velocity_axes=np.atleast_2d(i_velocity_axes.to(u.m / u.s).value),
        efn=np.atleast_2d(efn.to(u.s / u.m).value),
        ifn=np.atleast_2d(ifn.to(u.s / u.m).value),
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        notch=notch_si,
        inner_range=inner_range,
        inner_frac=inner_frac,
    )

    return alpha, Skw / u.m


# ***************************************************************************
# These functions are necessary to interface scalar Parameter objects with
# the array inputs of spectral_density
# ***************************************************************************


def _count_populations_in_params(params: dict[str, Any], prefix: str) -> int:
    """
    Counts the number of electron or ion populations in a ``params``
    `dict`.

    The number of populations is determined by counting the number of
    items in the ``params`` `dict` with a key that starts with the
    string defined by ``prefix``.
    """
    return len([key for key in params if key.startswith(prefix)])


def _params_to_array(
    params: dict[str, Any],
    prefix: str,
    vector: bool = False,  # noqa: FBT001, FBT002
) -> np.ndarray:
    """
    Constructs an array from the values contained in the dictionary
    ``params`` associated with keys starting with the prefix defined by
    ``prefix``.

    If ``vector == False``, then values for keys matching the expression
    ``prefix_[0-9]+`` are gathered into a 1D array.

    If ``vector == True``, then values for keys matching the expression
    ``prefix_[xyz]_[0-9]+`` are gathered into a 2D array of shape
    ``(N, 3)``.

    Notes
    -----
    This function allows `lmfit.parameter.Parameter` inputs to be
    converted into the array-type inputs required by the spectral
    density function.
    """
    if vector:
        npop = _count_populations_in_params(params, f"{prefix}_x")
        output = np.zeros([npop, 3])
        for i in range(npop):
            for j, ax in enumerate(["x", "y", "z"]):
                output[i, j] = params[f"{prefix}_{ax}_{i}"].value

    else:
        npop = _count_populations_in_params(params, prefix)
        output = np.zeros([npop])
        for i in range(npop):
            output[i] = params[f"{prefix}_{i}"]

    return output


# ***************************************************************************
# Fitting functions
# ***************************************************************************


def _spectral_density_model(wavelengths, settings=None, **params):
    """
    `lmfit` model function for fitting Thomson spectra.

    For descriptions of arguments, see the `thomson_model` function.
    """
    # LOAD FROM SETTINGS
    probe_vec = settings["probe_vec"]  # ty:ignore[not-subscriptable]
    scatter_vec = settings["scatter_vec"]  # ty:ignore[not-subscriptable]
    electron_vdir = settings["electron_vdir"]  # ty:ignore[not-subscriptable]
    ion_vdir = settings["ion_vdir"]  # ty:ignore[not-subscriptable]
    probe_wavelength = settings["probe_wavelength"]  # ty:ignore[not-subscriptable]
    instr_func_arr = settings["instr_func_arr"]  # ty:ignore[not-subscriptable]
    notch = settings["notch"]  # ty:ignore[not-subscriptable]

    # LOAD FROM PARAMS
    n = params["n"]
    background = params["background"]
    T_e = _params_to_array(params, "T_e")
    T_i = _params_to_array(params, "T_i")
    ion_mu = _params_to_array(params, "ion_mu")
    ion_z = _params_to_array(params, "ion_z")
    efract = _params_to_array(params, "efract")
    ifract = _params_to_array(params, "ifract")

    electron_speed = _params_to_array(params, "electron_speed")
    ion_speed = _params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to kelvin (required by fast_spectral_density)
    T_e *= 11604.51812155
    T_i *= 11604.51812155

    # lite function takes ion mass, not mu=m_i/m_p
    ion_mass = ion_mu * m_p_si_unitless

    _alpha, model_Skw = spectral_density_lite(
        wavelengths,
        probe_wavelength,
        n,
        T_e,
        T_i,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_mass=ion_mass,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        instr_func_arr=instr_func_arr,
        notch=notch,
    )

    model_Skw *= 1 / np.max(model_Skw)

    # Add background after normalization
    model_Skw += background

    return model_Skw


def spectral_density_model(  # noqa: C901, PLR0912, PLR0915
    wavelengths,
    settings,
    params,
):
    r"""
    Returns a `lmfit.model.Model` function for Thomson spectral density
    function.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Wavelength array, in meters.

    settings : dict
        A dictionary of non-variable inputs to the spectral density
        function which must include the following keys:

        - ``"probe_wavelength"``: Probe wavelength in meters
        - ``"probe_vec"`` : (3,) unit vector in the probe direction
        - ``"scatter_vec"``: (3,) unit vector in the scattering
          direction
        - ``"ions"`` : list of particle strings,
          `~plasmapy.particles.particle_class.Particle` objects, or a
          `~plasmapy.particles.particle_collections.ParticleList`
          describing each ion species. All ions must be positive. Ion mass
          and charge number from this list will be automatically
          added as fixed parameters, overridden by any ``ion_mass`` or ``ion_z``
          parameters explicitly created.

        and may contain the following optional variables:

        - ``"electron_vdir"`` : (e#, 3) array of electron velocity unit
          vectors
        - ``"ion_vdir"`` : (e#, 3) array of ion velocity unit vectors
        - ``"instr_func"`` : A function that takes a wavelength
          |Quantity| array and returns a spectrometer instrument
          function as an `~numpy.ndarray`.
        - ``"notch"`` : A wavelength range or array of multiple wavelength
          ranges over which the spectral density is set to 0.

        These quantities cannot be varied during the fit.

    params : `~lmfit.parameter.Parameters` object
        A `~lmfit.parameter.Parameters` object that must contain the
        following variables:

        - n: Total combined density of the electron populations in
          m\ :sup:`-3`
        - :samp:`T_e_{e#}` : Temperature in eV
        - :samp:`T_i_{i#}` : Temperature in eV

        where where :samp:`{i#}` and where :samp:`{e#}` are replaced by
        the number of electron and ion populations, zero-indexed,
        respectively (e.g., 0, 1, 2, ...). The
        `~lmfit.parameter.Parameters` object may also contain the
        following optional variables:

        - :samp:`"efract_{e#}"` : Fraction of each electron population
          (must sum to 1)
        - :samp:`"ifract_{i#}"` : Fraction of each ion population (must
          sum to 1)
        - :samp:`"electron_speed_{e#}"` : Electron speed in m/s
        - :samp:`"ion_speed_{ei}"` : Ion speed in m/s
        - :samp:`"ion_mu_{i#}"` : Ion mass number, :math:`μ = m_i/m_p`
        - :samp:`"ion_z_{i#}"` : Ion charge number
        - :samp:`"background"` : Background level, as fraction of max signal

        These quantities can be either fixed or varying.

    Returns
    -------
    model : `lmfit.model.Model`
        An `lmfit.model.Model` of the spectral density function for the
        provided settings and parameters that can be used to fit Thomson
        scattering data.

    Notes
    -----
    If an instrument function is included, the data should not include
    any `numpy.nan` values — instead regions with no data should be
    removed from both the data and wavelength arrays using
    `numpy.delete`.
    """
    required_settings = {
        "probe_wavelength",
        "probe_vec",
        "scatter_vec",
    }

    if missing_settings := required_settings - set(settings):
        raise ValueError(
            f"The following required settings were not provided in the "
            f"'settings' argument: {missing_settings}",
        )

    required_params = {"n"}
    if missing_params := required_params - set(params):
        raise ValueError(
            f"The following required parameters were not provided in the "
            f"'params': {missing_params}",
        )

    # Add background if not provided
    if "background" not in params:
        params.add("background", value=0.0, vary=False)

    # Add ion values as fixed parameters if a particle list is provided
    # in settings
    # Do not override any existing parameters
    if "ions" in settings:
        for i, ion in enumerate(settings["ions"]):
            _ion = Particle(ion)
            if f"ion_mu_{i!s}" not in params:
                params.add(
                    f"ion_mu_{i!s}",
                    value=_ion.mass.to(u.kg).value / m_p_si_unitless,
                    vary=False,
                )
            if f"ion_z_{i!s}" not in params:
                params.add(f"ion_z_{i!s}", value=_ion.charge_number, vary=False)

    # **********************
    # Count number of populations
    # **********************
    if "efract_0" not in params:
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in params:
        params.add("ifract_0", value=1.0, vary=False)

    num_e = _count_populations_in_params(params, "efract")
    num_i = _count_populations_in_params(params, "ifract")

    # **********************
    # Required settings and parameters per population
    # **********************
    for p, nums in zip(
        ["T_e", "T_i", "ion_mu", "ion_z"],
        [num_e, num_i, num_i, num_i],
        strict=False,
    ):
        for num in range(nums):
            key = f"{p}_{num!s}"
            if key not in params:
                raise ValueError(
                    f"{p} was not provided in kwarg 'parameters', but is required.",
                )

    # **************
    # efract and ifract
    # **************

    # Automatically add an expression to the last efract parameter to
    # indicate that it depends on the others (so they sum to 1.0)
    # The resulting expression for the last of three will look like
    # efract_2.expr = "1.0 - efract_0 - efract_1"
    if num_e > 1:
        nums = ["1.0"] + [f"efract_{i}" for i in range(num_e - 1)]
        params[f"efract_{num_e - 1}"].expr = " - ".join(nums)

    if num_i > 1:
        nums = ["1.0"] + [f"ifract_{i}" for i in range(num_i - 1)]
        params[f"ifract_{num_i - 1}"].expr = " - ".join(nums)

    # **************
    # Electron velocity
    # **************
    electron_speed = np.zeros(num_e)
    for num in range(num_e):
        k = f"electron_speed_{num}"
        if k in params:
            electron_speed[num] = params[k].value
        else:
            # electron_speed[e] = 0 already
            params.add(k, value=0, vary=False)

    if "electron_vdir" not in settings:
        if np.all(electron_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["electron_vdir"] = np.ones([num_e, 3])
        else:
            raise ValueError(
                "Key 'electron_vdir' must be defined in kwarg 'settings' if "
                "any electron population has a non-zero speed (i.e. any "
                "params['electron_speed_<#>'] is non-zero).",
            )
    norm = np.linalg.norm(settings["electron_vdir"], axis=-1)
    settings["electron_vdir"] = settings["electron_vdir"] / norm[:, np.newaxis]

    # **************
    # Ion velocity
    # **************
    ion_speed = np.zeros(num_i)
    for num in range(num_i):
        k = f"ion_speed_{num}"
        if k in params:
            ion_speed[num] = params[k].value
        else:
            # ion_speed[i] = 0 already
            params.add(k, value=0, vary=False)

    if "ion_vdir" not in list(settings.keys()):
        if np.all(ion_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["ion_vdir"] = np.ones([num_i, 3])
        else:
            raise ValueError(
                "Key 'ion_vdir' must be defined in kwarg 'settings' if "
                "any ion population has a non-zero speed (i.e. any "
                "params['ion_speed_<#>'] is non-zero).",
            )
    norm = np.linalg.norm(settings["ion_vdir"], axis=-1)
    settings["ion_vdir"] = settings["ion_vdir"] / norm[:, np.newaxis]

    if "instr_func" not in settings or settings["instr_func"] is None:
        settings["instr_func_arr"] = None
    else:
        # Create instr_func array from instr_func
        instr_func = settings["instr_func"]
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        instr_func_arr = instr_func(eval_w * u.m)

        if type(instr_func_arr) is not np.ndarray:
            raise ValueError(
                "instr_func must be a function that returns a "
                "np.ndarray, but the provided function returns "
                f" a {type(instr_func_arr)}",
            )

        if wavelengths.shape != instr_func_arr.shape:
            raise ValueError(
                "The shape of the array returned from the "
                f"instr_func ({instr_func_arr.shape}) "
                "does not match the shape of the wavelengths "
                f"array ({wavelengths.shape}).",
            )

        instr_func_arr *= 1 / np.sum(instr_func_arr)
        settings["instr_func_arr"] = instr_func_arr

        warnings.warn(
            "If an instrument function is included, the data "
            "should not include any `numpy.nan` values. "
            "Instead regions with no data should be removed from "
            "both the data and wavelength arrays using "
            "`numpy.delete`.",
            stacklevel=2,
        )

    if "notch" not in settings:
        settings["notch"] = None

    # TODO: raise an exception if the number of any of the ion or electron  # noqa: FIX002
    #       quantities isn't consistent with the number of that species defined
    #       by ifract or efract.

    def _spectral_density_model_lambda(wavelengths, **params):
        return _spectral_density_model(wavelengths, settings=settings, **params)

    # Create and return the lmfit.Model
    return Model(
        _spectral_density_model_lambda,
        independent_vars=["wavelengths"],
        nan_policy="omit",
    )
