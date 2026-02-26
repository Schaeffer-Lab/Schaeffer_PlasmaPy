"""Unified dielectric / permittivity module.

This file combines the functionality of:
- dielectric-2.py (canonical reference implementation)
- dielectric_fast.py (optimized / fast variants)

Design goals:
- Minimal changes to original code.
- Preserve the original public API from dielectric-2.py unchanged.
- Provide access to the fast implementations without name collisions via *_fast aliases.

Notes:
- The fast module source is executed into an isolated namespace `_FAST_NS` so its
  original definitions remain unchanged.
"""

# =============================================================================
# Canonical implementation (from dielectric-2.py)
# =============================================================================

"""Plasma dielectric parameters."""

__all__ = [
    "cold_plasma_permittivity_SDP",
    "cold_plasma_permittivity_LRP",
    "permittivity_1D_Maxwellian",
    "RotatingTensorElements",
    "StixTensorElements",
]
__lite_funcs__ = ["permittivity_1D_Maxwellian_lite"]

from collections import namedtuple
from collections.abc import Sequence

import astropy.units as u
import numpy as np

from plasmapy.dispersion.dispersion_functions import plasma_dispersion_func_deriv
from plasmapy.formulary.frequencies import gyrofrequency, plasma_frequency
from plasmapy.formulary.speeds import thermal_speed
from plasmapy.particles.particle_class import ParticleLike
from plasmapy.particles.particle_collections import ParticleListLike
from plasmapy.utils.decorators import (
    bind_lite_func,
    preserve_signature,
    validate_quantities,
)

__all__ += __lite_funcs__

r"""
Values should be returned as a `~astropy.units.Quantity` in SI units.
"""

StixTensorElements = namedtuple("StixTensorElements", ["sum", "difference", "plasma"])
"""Output type for `~plasmapy.formulary.dielectric.cold_plasma_permittivity_SDP`."""


RotatingTensorElements = namedtuple(
    "RotatingTensorElements", ["left", "right", "plasma"]
)
"""Output type for `~plasmapy.formulary.dielectric.cold_plasma_permittivity_LRP`."""


@validate_quantities(B={"can_be_negative": False}, omega={"can_be_negative": False})
def cold_plasma_permittivity_SDP(
    B: u.Quantity[u.T],
    species: ParticleListLike,
    n: Sequence[u.Quantity[u.m**-3]] | u.Quantity[u.m**-3],
    omega: u.Quantity[u.rad / u.s],
) -> StixTensorElements:
    r"""
    Magnetized cold plasma dielectric permittivity tensor elements.

    Elements (S, D, P) are given in the "Stix" frame, i.e. with
    :math:`B ∥ \hat{z}` :cite:p:`stix:1992`.

    The :math:`\exp(-i ω t)` time-harmonic convention is assumed.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        Magnetic field magnitude in units convertible to tesla.

    species : |particle-list-like|
        List of the plasma particle species,
        e.g.: ``['e-', 'D+']`` or ``['e-', 'D+', 'He+']``.

    n : `list` of `~astropy.units.Quantity`
        `list` of species density in units convertible to per cubic meter
        The order of the species densities should follow species.

    omega : `~astropy.units.Quantity`
        Electromagnetic wave frequency in rad/s.

    Returns
    -------
    sum : `~astropy.units.Quantity`
        The "sum" dielectric tensor element, :math:`S`.

    difference : `~astropy.units.Quantity`
        The "difference" dielectric tensor element, :math:`D`.

    plasma : `~astropy.units.Quantity`
        The "plasma" dielectric tensor element, :math:`P`.

    Notes
    -----
    The dielectric permittivity tensor is expressed in the Stix frame
    with the :math:`\exp(-i ω t)` time-harmonic convention as
    :math:`ε = ε_0 A`, with :math:`A` being

    .. math::

        ε = ε_0 \left(\begin{matrix}  S & -i D & 0 \\
                              +i D & S & 0 \\
                              0 & 0 & P \end{matrix}\right)

    where:

    .. math::
        S = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2 - Ω_{c,s}^2}

        D = \sum_s \frac{Ω_{c,s}}{ω}
            \frac{ω_{p,s}^2}{ω^2 - Ω_{c,s}^2}

        P = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2}

    where :math:`ω_{p,s}` is the plasma frequency and
    :math:`Ω_{c,s}` is the signed version of the cyclotron frequency
    for the species :math:`s`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> B = 2*u.T
    >>> species = ['e-', 'D+']
    >>> n = [1e18*u.m**-3, 1e18*u.m**-3]
    >>> omega = 3.7e9*(2*pi)*(u.rad/u.s)
    >>> permittivity = S, D, P = cold_plasma_permittivity_SDP(B, species, n, omega)
    >>> S
    <Quantity 1.02422...>
    >>> permittivity.sum   # namedtuple-style access
    <Quantity 1.02422...>
    >>> D
    <Quantity 0.39089...>
    >>> P
    <Quantity -4.8903...>
    """
    S, D, P = 1, 0, 1

    for s, n_s in zip(species, n, strict=False):
        omega_c = gyrofrequency(B=B, particle=s, signed=True)
        omega_p = plasma_frequency(n=n_s, particle=s)

        S += -(omega_p**2) / (omega**2 - omega_c**2)
        D += omega_c / omega * omega_p**2 / (omega**2 - omega_c**2)
        P += -(omega_p**2) / omega**2
    return StixTensorElements(S, D, P)


@validate_quantities(B={"can_be_negative": False}, omega={"can_be_negative": False})
def cold_plasma_permittivity_LRP(
    B: u.Quantity[u.T],
    species: ParticleListLike,
    n: list[u.Quantity[u.m**-3]] | u.Quantity[u.m**-3],
    omega: u.Quantity[u.rad / u.s],
) -> RotatingTensorElements:
    r"""
    Magnetized cold plasma dielectric permittivity tensor elements.

    Elements (L, R, P) are given in the "rotating" basis, i.e. in the basis
    :math:`(\mathbf{u}_{+}, \mathbf{u}_{-}, \mathbf{u}_z)`,
    where the tensor is diagonal and with :math:`B ∥ z`\ .

    The :math:`\exp(-i ω t)` time-harmonic convention is assumed.

    Parameters
    ----------
    B : `~astropy.units.Quantity`
        Magnetic field magnitude in units convertible to tesla.

    species : (k,) |particle-list-like|
        The plasma particle species (e.g.: ``['e-', 'D+']`` or
        ``['e-', 'D+', 'He+']``.

    n : (k,) `list` of `~astropy.units.Quantity`
        `list` of species density in units convertible to per cubic meter.
        The order of the species densities should follow species.

    omega : `~astropy.units.Quantity`
        Electromagnetic wave frequency in rad/s.

    Returns
    -------
    left : `~astropy.units.Quantity`
        L ("Left") Left-handed circularly polarization tensor element.

    right : `~astropy.units.Quantity`
        R ("Right") Right-handed circularly polarization tensor element.

    plasma : `~astropy.units.Quantity`
        P ("Plasma") dielectric tensor element.

    Notes
    -----
    In the rotating frame defined by
    :math:`(\mathbf{u}_{+}, \mathbf{u}_{-}, \mathbf{u}_z)`
    with :math:`\mathbf{u}_{±}=(\mathbf{u}_x ± \mathbf{u}_y)/\sqrt{2}`,
    the dielectric tensor takes a diagonal form with elements L, R, P with:

    .. math::
        L = 1 - \sum_s
                \frac{ω_{p,s}^2}{ω\left(ω - Ω_{c,s}\right)}

        R = 1 - \sum_s
                \frac{ω_{p,s}^2}{ω\left(ω + Ω_{c,s}\right)}

        P = 1 - \sum_s \frac{ω_{p,s}^2}{ω^2}

    where :math:`ω_{p,s}` is the plasma frequency and
    :math:`Ω_{c,s}` is the signed version of the cyclotron frequency
    for the species :math:`s` :cite:p:`stix:1992`.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> B = 2 * u.T
    >>> species = ["e-", "D+"]
    >>> n = [1e18 * u.m**-3, 1e18 * u.m**-3]
    >>> omega = 3.7e9 * (2 * pi) * (u.rad / u.s)
    >>> L, R, P = permittivity = cold_plasma_permittivity_LRP(B, species, n, omega)
    >>> L
    <Quantity 0.63333...>
    >>> permittivity.left  # namedtuple-style access
    <Quantity 0.63333...>
    >>> R
    <Quantity 1.41512...>
    >>> P
    <Quantity -4.8903...>
    """
    L, R, P = 1, 1, 1

    for s, n_s in zip(species, n, strict=False):
        omega_c = gyrofrequency(B=B, particle=s, signed=True)
        omega_p = plasma_frequency(n=n_s, particle=s)

        L += -(omega_p**2) / (omega * (omega - omega_c))
        R += -(omega_p**2) / (omega * (omega + omega_c))
        P += -(omega_p**2) / omega**2
    return RotatingTensorElements(L, R, P)


@preserve_signature
def permittivity_1D_Maxwellian_lite(omega, kWave, vth, wp):
    r"""
    The :term:`lite-function` for
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian`.
    Performs the same calculations as
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian`, but is
    intended for computational use and, thus, has data conditioning
    safeguards removed.

    Parameters
    ----------
    omega : |array_like| of real positive values
        The frequency, in rad/s, of the electromagnetic wave propagating
        through the plasma.

    kWave : |array_like| of real values
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    vth : `float`
        The 3D, most probable thermal speed, in m/s. (i.e. it includes
        the factor of :math:`\sqrt{2}`, see
        :ref:`thermal speed notes <thermal-speed-notes>`)

    wp : `float`
        The plasma frequency, in rad/s.

    Returns
    -------
    chi : |array_like| of complex values
        The ion or the electron dielectric permittivity of the plasma.
        This is a dimensionless quantity.

    See Also
    --------
    ~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian

    Examples
    --------
    >>> import astropy.units as u
    >>> from plasmapy.formulary import thermal_speed, plasma_frequency
    >>> T = 30 * u.eV
    >>> n = 1e18 * u.cm**-3
    >>> particle = "Ne"
    >>> Z = 8
    >>> omega = 3.541e15  # in rad/s
    >>> vth = thermal_speed(T=T, particle=particle).value
    >>> wp = plasma_frequency(n=n, particle=particle, Z=Z).value
    >>> k_wave = omega / vth
    >>> permittivity_1D_Maxwellian_lite(omega, k_wave, vth=vth, wp=wp)
    np.complex128(-6.72794...e-08+5.76024...e-07j)
    """
    
    # scattering parameter alpha.
    # explicitly removing factor of sqrt(2) to be consistent with Froula
    
    alpha = np.sqrt(2) * wp / (kWave * vth)

    # The dimensionless phase velocity of the propagating EM wave.
    zeta = omega / (kWave * vth)
    return -0.5 * (alpha**2) * plasma_dispersion_func_deriv(zeta)


@bind_lite_func(permittivity_1D_Maxwellian_lite)
@validate_quantities(
    kWave={"none_shall_pass": True}, validations_on_return={"can_be_complex": True}
)
def permittivity_1D_Maxwellian(
    omega: u.Quantity[u.rad / u.s],
    kWave: u.Quantity[u.rad / u.m],
    T: u.Quantity[u.K],
    n: u.Quantity[u.m**-3],
    particle: ParticleLike,
    z_mean: float | None = None,
) -> u.Quantity[u.dimensionless_unscaled]:
    r"""
    Compute the classical dielectric permittivity for a 1D Maxwellian
    plasma.

    This function can calculate both the ion and electron
    permittivities.  No additional effects are considered (e.g.
    magnetic fields, relativistic effects, strongly coupled regime, etc.).

    Parameters
    ----------
    omega : `~astropy.units.Quantity`
        The frequency, in rad/s, of the electromagnetic wave propagating
        through the plasma.

    kWave : `~astropy.units.Quantity`
        The corresponding wavenumber, in rad/m, of the electromagnetic
        wave propagating through the plasma.

    T : `~astropy.units.Quantity`
        The plasma temperature — this can be either the electron or the
        ion temperature, but should be consistent with density and
        particle.

    n : `~astropy.units.Quantity`
        The plasma density — this can be either the electron or the ion
        density, but should be consistent with temperature and particle.

    particle : |particle-like|
        The plasma particle species.

    z_mean : `float`
        The average ionization of the plasma. This is only required for
        calculating the ion permittivity.

    Returns
    -------
    chi : `~astropy.units.Quantity`
        The ion or the electron dielectric permittivity of the plasma.
        This is a dimensionless quantity.

    Notes
    -----
    The dielectric permittivities for a Maxwellian plasma are described
    by the following equations (see p. 106 of :cite:t:`froula:2011`):

    .. math::
        χ_e(k, ω) = - \frac{α_e^2}{2} Z'(x_e)

        χ_i(k, ω) = - \frac{α_i^2}{2}\frac{Z}{} Z'(x_i)

        α = \frac{ω_p}{k v_{Th}}

        x = \frac{ω}{k v_{Th}}

    :math:`χ_e` and :math:`χ_i` are the electron and ion permittivities,
    respectively. :math:`Z'` is the derivative of the plasma dispersion
    function. :math:`α` is the scattering parameter which delineates
    the difference between the collective and non-collective Thomson
    scattering regimes. :math:`x` is the dimensionless phase velocity
    of the electromagnetic wave propagating through the plasma.

    Examples
    --------
    >>> import astropy.units as u
    >>> from numpy import pi
    >>> from plasmapy.formulary import thermal_speed
    >>> T = 30 * 11600 * u.K
    >>> n = 1e18 * u.cm**-3
    >>> particle = "Ne"
    >>> Z = 8
    >>> vth = thermal_speed(T, particle, method="most_probable")
    >>> omega = 5.635e14 * 2 * pi * u.rad / u.s
    >>> k_wave = omega / vth
    >>> permittivity_1D_Maxwellian(omega, k_wave, T, n, particle, Z)
    <Quantity -6.72955...e-08+5.76163...e-07j>

    For user convenience
    `~plasmapy.formulary.dielectric.permittivity_1D_Maxwellian_lite`
    is bound to this function and can be used as follows:

    >>> from plasmapy.formulary import plasma_frequency
    >>> wp = plasma_frequency(n, particle, Z=Z)
    >>> permittivity_1D_Maxwellian.lite(
    ...     omega.value, k_wave.value, vth=vth.value, wp=wp.value
    ... )
    np.complex128(-6.72955...e-08+5.76163...e-07j)
    """
    vth = thermal_speed(T=T, particle=particle, method="most_probable").value
    wp = plasma_frequency(n=n, particle=particle, Z=z_mean).value

    chi = permittivity_1D_Maxwellian_lite(
        omega.value,
        kWave.value,
        vth,
        wp,
    )
    return chi * u.dimensionless_unscaled

# =============================================================================
# Fast implementations (from dielectric_fast.py)
# Executed in an isolated namespace to avoid collisions.
# =============================================================================

# --- begin embedded dielectric_fast.py source ---
_FAST_SOURCE = "\"\"\"Functions to calculate plasma dielectric parameters\"\"\"\n__all__ = [\n    \"cold_plasma_permittivity_SDP\",\n    \"cold_plasma_permittivity_LRP\",\n    \"permittivity_1D_Maxwellian\",\n]\n\nimport numpy as np\n\nfrom astropy import units as u\nfrom collections import namedtuple\n\nfrom plasmapy.dispersion.dispersionfunction import plasma_dispersion_func_deriv\nfrom plasmapy.formulary import parameters\nfrom plasmapy.utils.decorators import validate_quantities\n\nr\"\"\"\nValues should be returned as a `~astropy.units.Quantity` in SI units.\n\"\"\"\n\nStixTensorElements = namedtuple(\"StixTensorElements\", [\"sum\", \"difference\", \"plasma\"])\nRotatingTensorElements = namedtuple(\n    \"RotatingTensorElements\", [\"left\", \"right\", \"plasma\"]\n)\n\n\n@validate_quantities(B={\"can_be_negative\": False}, omega={\"can_be_negative\": False})\ndef cold_plasma_permittivity_SDP(B: u.T, species, n, omega: u.rad / u.s):\n    r\"\"\"\n    Magnetized cold plasma dielectric permittivity tensor elements.\n\n    Elements (S, D, P) are given in the \"Stix\" frame, i.e. with\n    :math:`B \u2225 \\hat{z}`\\ .\n\n    The :math:`\\exp(-i \u03c9 t)` time-harmonic convention is assumed.\n\n    Parameters\n    ----------\n    B : `~astropy.units.Quantity`\n        Magnetic field magnitude in units convertible to tesla.\n\n    species : `list` of `str`\n        List of the plasma particle species,\n        e.g.: ``['e', 'D+']`` or ``['e', 'D+', 'He+']``.\n\n    n : `list` of `~astropy.units.Quantity`\n        `list` of species density in units convertible to per cubic meter\n        The order of the species densities should follow species.\n\n    omega : `~astropy.units.Quantity`\n        Electromagnetic wave frequency in rad/s.\n\n    Returns\n    -------\n    sum : `~astropy.units.Quantity`\n        S (\"Sum\") dielectric tensor element.\n\n    difference : `~astropy.units.Quantity`\n        D (\"Difference\") dielectric tensor element.\n\n    plasma : `~astropy.units.Quantity`\n        P (\"Plasma\") dielectric tensor element.\n\n    Notes\n    -----\n    The dielectric permittivity tensor is expressed in the Stix frame with\n    the :math:`\\exp(-i \u03c9 t)` time-harmonic convention as\n    :math:`\u03b5 = \u03b5_0 A`, with :math:`A` being\n\n    .. math::\n\n        \u03b5 = \u03b5_0 \\left(\\begin{matrix}  S & -i D & 0 \\\\\n                              +i D & S & 0 \\\\\n                              0 & 0 & P \\end{matrix}\\right)\n\n    where:\n\n    .. math::\n        S = 1 - \\sum_s \\frac{\u03c9_{p,s}^2}{\u03c9^2 - \u03a9_{c,s}^2}\n\n        D = \\sum_s \\frac{\u03a9_{c,s}}{\u03c9}\n            \\frac{\u03c9_{p,s}^2}{\u03c9^2 - \u03a9_{c,s}^2}\n\n        P = 1 - \\sum_s \\frac{\u03c9_{p,s}^2}{\u03c9^2}\n\n    where :math:`\u03c9_{p,s}` is the plasma frequency and\n    :math:`\u03a9_{c,s}` is the signed version of the cyclotron frequency\n    for the species :math:`s`.\n\n    References\n    ----------\n    - T.H. Stix, Waves in Plasma, 1992.\n\n    Examples\n    --------\n    >>> from astropy import units as u\n    >>> from numpy import pi\n    >>> B = 2*u.T\n    >>> species = ['e', 'D+']\n    >>> n = [1e18*u.m**-3, 1e18*u.m**-3]\n    >>> omega = 3.7e9*(2*pi)*(u.rad/u.s)\n    >>> permittivity = S, D, P = cold_plasma_permittivity_SDP(B, species, n, omega)\n    >>> S\n    <Quantity 1.02422...>\n    >>> permittivity.sum   # namedtuple-style access\n    <Quantity 1.02422...>\n    >>> D\n    <Quantity 0.39089...>\n    >>> P\n    <Quantity -4.8903...>\n    \"\"\"\n    S, D, P = 1, 0, 1\n\n    for s, n_s in zip(species, n):\n        omega_c = parameters.gyrofrequency(B=B, particle=s, signed=True)\n        omega_p = parameters.plasma_frequency(n=n_s, particle=s)\n\n        S += -(omega_p ** 2) / (omega ** 2 - omega_c ** 2)\n        D += omega_c / omega * omega_p ** 2 / (omega ** 2 - omega_c ** 2)\n        P += -(omega_p ** 2) / omega ** 2\n    return StixTensorElements(S, D, P)\n\n\n@validate_quantities(B={\"can_be_negative\": False}, omega={\"can_be_negative\": False})\ndef cold_plasma_permittivity_LRP(B: u.T, species, n, omega: u.rad / u.s):\n    r\"\"\"\n    Magnetized cold plasma dielectric permittivity tensor elements.\n\n    Elements (L, R, P) are given in the \"rotating\" basis, i.e. in the basis\n    :math:`(\\mathbf{u}_{+}, \\mathbf{u}_{-}, \\mathbf{u}_z)`,\n    where the tensor is diagonal and with :math:`B \u2225 z`\\ .\n\n    The :math:`\\exp(-i \u03c9 t)` time-harmonic convention is assumed.\n\n    Parameters\n    ----------\n    B : `~astropy.units.Quantity`\n        Magnetic field magnitude in units convertible to tesla.\n\n    species : `list` of `str`\n        The plasma particle species (e.g.: ``['e', 'D+']`` or\n        ``['e', 'D+', 'He+']``.\n\n    n : `list` of `~astropy.units.Quantity`\n        `list` of species density in units convertible to per cubic meter.\n        The order of the species densities should follow species.\n\n    omega : `~astropy.units.Quantity`\n        Electromagnetic wave frequency in rad/s.\n\n    Returns\n    -------\n    left : `~astropy.units.Quantity`\n        L (\"Left\") Left-handed circularly polarization tensor element.\n\n    right : `~astropy.units.Quantity`\n        R (\"Right\") Right-handed circularly polarization tensor element.\n\n    plasma : `~astropy.units.Quantity`\n        P (\"Plasma\") dielectric tensor element.\n\n    Notes\n    -----\n    In the rotating frame defined by\n    :math:`(\\mathbf{u}_{+}, \\mathbf{u}_{-}, \\mathbf{u}_z)`\n    with :math:`\\mathbf{u}_{\\pm}=(\\mathbf{u}_x \\pm \\mathbf{u}_y)/\\sqrt{2}`,\n    the dielectric tensor takes a diagonal form with elements L, R, P with:\n\n    .. math::\n        L = 1 - \\sum_s\n                \\frac{\u03c9_{p,s}^2}{\u03c9\\left(\u03c9 - \u03a9_{c,s}\\right)}\n\n        R = 1 - \\sum_s\n                \\frac{\u03c9_{p,s}^2}{\u03c9\\left(\u03c9 + \u03a9_{c,s}\\right)}\n\n        P = 1 - \\sum_s \\frac{\u03c9_{p,s}^2}{\u03c9^2}\n\n    where :math:`\u03c9_{p,s}` is the plasma frequency and\n    :math:`\u03a9_{c,s}` is the signed version of the cyclotron frequency\n    for the species :math:`s`.\n\n    References\n    ----------\n    - T.H. Stix, Waves in Plasma, 1992.\n\n    Examples\n    --------\n    >>> from astropy import units as u\n    >>> from numpy import pi\n    >>> B = 2*u.T\n    >>> species = ['e', 'D+']\n    >>> n = [1e18*u.m**-3, 1e18*u.m**-3]\n    >>> omega = 3.7e9*(2*pi)*(u.rad/u.s)\n    >>> L, R, P = permittivity = cold_plasma_permittivity_LRP(B, species, n, omega)\n    >>> L\n    <Quantity 0.63333...>\n    >>> permittivity.left    # namedtuple-style access\n    <Quantity 0.63333...>\n    >>> R\n    <Quantity 1.41512...>\n    >>> P\n    <Quantity -4.8903...>\n    \"\"\"\n    L, R, P = 1, 1, 1\n\n    for s, n_s in zip(species, n):\n        omega_c = parameters.gyrofrequency(B=B, particle=s, signed=True)\n        omega_p = parameters.plasma_frequency(n=n_s, particle=s)\n\n        L += -(omega_p ** 2) / (omega * (omega - omega_c))\n        R += -(omega_p ** 2) / (omega * (omega + omega_c))\n        P += -(omega_p ** 2) / omega ** 2\n    return RotatingTensorElements(L, R, P)\n\n\ndef fast_permittivity_1D_Maxwellian(omega, kWave, vTh, wp):\n    # scattering parameter alpha.\n    # explicitly removing factor of sqrt(2) to be consistent with Froula\n    alpha = np.sqrt(2) * wp / (kWave * vTh)\n    # The dimensionless phase velocity of the propagating EM wave.\n    zeta = omega / (kWave * vTh)\n    chi = alpha ** 2 * (-1 / 2) * plasma_dispersion_func_deriv(zeta)\n    return chi\n\n\n@validate_quantities(\n    kWave={\"none_shall_pass\": True}, validations_on_return={\"can_be_complex\": True}\n)\ndef permittivity_1D_Maxwellian(\n    omega: u.rad / u.s,\n    kWave: u.rad / u.m,\n    T: u.K,\n    n: u.m ** -3,\n    particle,\n    z_mean: u.dimensionless_unscaled = None,\n) -> u.dimensionless_unscaled:\n    r\"\"\"\n    Compute the classical dielectric permittivity for a 1D Maxwellian plasma.\n\n    This function can calculate both the ion and electron permittivities.\n    No additional effects are considered (e.g. magnetic fields, relativistic\n    effects, strongly coupled regime, etc.).\n\n    Parameters\n    ----------\n    omega : `~astropy.units.Quantity`\n        The frequency in rad/s of the electromagnetic wave propagating\n        through the plasma.\n\n    kWave : `~astropy.units.Quantity`\n        The corresponding wavenumber, in rad/m, of the electromagnetic wave\n        propagating through the plasma. This is often modulated by the\n        dispersion of the plasma or by relativistic effects. See em_wave.py\n        for ways to calculate this.\n\n    T : `~astropy.units.Quantity`\n        The plasma temperature \u2014 this can be either the electron or the ion\n        temperature, but should be consistent with density and particle.\n\n    n : `~astropy.units.Quantity`\n        The plasma density \u2014 this can be either the electron or the ion\n        density, but should be consistent with temperature and particle.\n\n    particle : `str`\n        The plasma particle species.\n\n    z_mean : `str`\n        The average ionization of the plasma. This is only required for\n        calculating the ion permittivity.\n\n    Returns\n    -------\n    chi : `~astropy.units.Quantity`\n        The ion or the electron dielectric permittivity of the plasma.\n        This is a dimensionless quantity.\n\n    Notes\n    -----\n    The dielectric permittivities for a Maxwellian plasma are described\n    by the following equations [1]_\n\n    .. math::\n        \u03c7_e(k, \u03c9) = - \\frac{\u03b1_e^2}{2} Z'(x_e)\n\n        \u03c7_i(k, \u03c9) = - \\frac{\u03b1_i^2}{2}\\frac{Z}{} Z'(x_i)\n\n        \u03b1 = \\frac{\u03c9_p}{k v_{Th}}\n\n        x = \\frac{\u03c9}{k v_{Th}}\n\n    :math:`\u03c7_e` and :math:`\u03c7_i` are the electron and ion permittivities,\n    respectively. :math:`Z'` is the derivative of the plasma dispersion\n    function. :math:`\u03b1` is the scattering parameter which delineates\n    the difference between the collective and non-collective Thomson\n    scattering regimes. :math:`x` is the dimensionless phase velocity\n    of the electromagnetic wave propagating through the plasma.\n\n    References\n    ----------\n    .. [1] J. Sheffield, D. Froula, S. H. Glenzer, and N. C. Luhmann Jr,\n       Plasma scattering of electromagnetic radiation: theory and measurement\n       techniques. Chapter 5 Pg 106 (Academic Press, 2010).\n\n    Example\n    -------\n    >>> from astropy import units as u\n    >>> from numpy import pi\n    >>> from astropy.constants import c\n    >>> T = 30 * 11600 * u.K\n    >>> n = 1e18 * u.cm**-3\n    >>> particle = 'Ne'\n    >>> z_mean = 8 * u.dimensionless_unscaled\n    >>> vTh = parameters.thermal_speed(T, particle, method=\"most_probable\")\n    >>> omega = 5.635e14 * 2 * pi * u.rad / u.s\n    >>> kWave = omega / vTh\n    >>> permittivity_1D_Maxwellian(omega, kWave, T, n, particle, z_mean)\n    <Quantity -6.72809...e-08+5.76037...e-07j>\n    \"\"\"\n    # thermal velocity\n    vTh = parameters.thermal_speed(T=T, particle=particle, method=\"most_probable\")\n    # plasma frequency\n    wp = parameters.plasma_frequency(n=n, particle=particle, z_mean=z_mean)\n\n    return fast_permittivity_1D_Maxwellian(omega, kWave, vTh, wp)\n"
# --- end embedded dielectric_fast.py source ---

_FAST_NS = {}
# Provide a minimal module-like global dict for exec.
_FAST_NS.update({
    "__name__": __name__ + "._fast",
    "__package__": __package__,
    "__file__": __file__,
})
exec(_FAST_SOURCE, _FAST_NS)

# Public aliases to fast implementations (avoid name collisions)
cold_plasma_permittivity_SDP_fast = _FAST_NS.get("cold_plasma_permittivity_SDP")
cold_plasma_permittivity_LRP_fast = _FAST_NS.get("cold_plasma_permittivity_LRP")
permittivity_1D_Maxwellian_fast = _FAST_NS.get("permittivity_1D_Maxwellian")
fast_permittivity_1D_Maxwellian = _FAST_NS.get("fast_permittivity_1D_Maxwellian")

# A small __all__ that preserves canonical names and adds *_fast aliases.
try:
    __all__  # type: ignore
except NameError:
    __all__ = []

__all__ = list(dict.fromkeys(list(__all__) + [
    "cold_plasma_permittivity_SDP",
    "cold_plasma_permittivity_LRP",
    "permittivity_1D_Maxwellian_lite",
    "permittivity_1D_Maxwellian",
    "cold_plasma_permittivity_SDP_fast",
    "cold_plasma_permittivity_LRP_fast",
    "permittivity_1D_Maxwellian_fast",
    "fast_permittivity_1D_Maxwellian",
]))
