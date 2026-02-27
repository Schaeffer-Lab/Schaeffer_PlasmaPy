"""
Unified Thomson scattering diagnostics module.

This file merges two existing implementations into a single module:

1) Arbitrary-VDF (NumPy/Astropy-units) implementation originally from thomsonVDF.py
2) Autodiff (PyTorch) implementation originally from cpu_autodiff_thomson.py

Design goals
------------
- Preserve the original public API from thomsonVDF.py as the default, so existing
  code that used `spectral_density_arbdist`, `spectral_density_maxwellian`,
  and `scattered_power_model_*` continues to work.
- Expose the autodiff forward model side-by-side with minimal disruption.
- Provide two clear entry points:
    * `arbitrary_forwardmodel`  -> arbitrary-VDF scattered power / spectral density
    * `autodiff_forwardmodel`   -> torch/autodiff scattered power / spectral density

Notes
-----
- Internals from each source file are namespaced with prefixes:
    arbitrary_*  and  autodiff_*
  to avoid name collisions while keeping the original logic intact.

Generated on 2026-02-26T21:21:08.
"""

from __future__ import annotations

# Standard library
from typing import List, Union, Optional

# Public exports
__all__ = [
    # Primary forward-model entry points
    "arbitrary_forwardmodel",
    "autodiff_forwardmodel",
    "experimental_forwardmodel",
    "thomson_forwardmodel",

    # Helper aliases (optional convenience)
    "chi_arbitrary_forwardmodel",
    "chi_autodiff_forwardmodel",

    # Experimental (PlasmaPy-style) API (suffixed)
    "spectral_density_experimental",
    "spectral_density_lite_experimental",
    "spectral_density_model_experimental",
]



# ----------------------------------------------------------------------------
# Arbitrary-VDF implementation (from thomsonVDF.py)
# ----------------------------------------------------------------------------
import astropy.constants as const
import astropy.units as u
import inspect
import numpy as np
import re
import warnings

from lmfit import Model
from numba import jit
from typing import List, Tuple, Union
import torch


from plasmapy.formulary.dielectric_functions import fast_permittivity_1D_Maxwellian
from plasmapy.formulary.parameters import fast_plasma_frequency, fast_thermal_speed
from plasmapy.particles import Particle, particle_mass
from plasmapy.utils.decorators import validate_quantities

_c = const.c.si.value  # Make sure C is in SI units
_e = const.e.si.value
_m_p = const.m_p.si.value
_m_e = const.m_e.si.value


# TODO: interface for inputting a multi-species configuration could be
# simplified using the plasmapy.classes.plasma_base class if that class
# included ion and electron drift velocities and information about the ion
# atomic species.


# TODO: If we can make this object pickle-able then we can set
# workers=-1 as a kw to differential_evolution to parallelize execution for fitting!
# The probem is a lambda function used in the Particle class...


# Computes the arbitrary_derivative of a function to 4th order precision. Used for the arbitrary scattered power function.
@jit(nopython=True)
def arbitrary_derivative(f, x, order):
    """
    x: array of x axis points
    f: array of f points
    unit: the unit that the output will be in (should be compatible with units of x, f)
    order: order of arbitrary_derivative, 1 or 2
    Computes df/dx
    """
    # Assume that x spacing is uniform
    dx = x[1] - x[0]

    fPrime = np.empty(len(f))

    # First arbitrary_derivative case
    if order == 1:
        # 4th order forward/backward difference approximations for endpoints
        fPrime[0] = (
            -25 / 12 * f[0] + 4 * f[1] - 3 * f[2] + 4 / 3 * f[3] - 1 / 4 * f[4]
        ) / dx
        fPrime[1] = (
            -25 / 12 * f[1] + 4 * f[2] - 3 * f[3] + 4 / 3 * f[4] - 1 / 4 * f[5]
        ) / dx
        fPrime[-1] = (
            1 / 4 * f[-5] - 4 / 3 * f[-4] + 3 * f[-3] - 4 * f[-2] + 25 / 12 * f[-1]
        ) / dx
        fPrime[-2] = (
            1 / 4 * f[-6] - 4 / 3 * f[-5] + 3 * f[-4] - 4 * f[-3] + 25 / 12 * f[-2]
        ) / dx
        # 4th order centered difference for everything else
        for j in range(2, len(f) - 2):
            fPrime[j] = (-f[j + 2] + 8 * f[j + 1] - 8 * f[j - 1] + f[j - 2]) / (12 * dx)

    # Second arbitrary_derivative case
    elif order == 2:
        # Endpoints
        fPrime[0] = (
            15 / 4 * f[0]
            - 77 / 6 * f[1]
            + 107 / 6 * f[2]
            - 13 * f[3]
            + 61 / 12 * f[4]
            - 5 / 6 * f[5]
        ) / dx ** 2
        fPrime[1] = (
            15 / 4 * f[1]
            - 77 / 6 * f[2]
            + 107 / 6 * f[3]
            - 13 * f[4]
            + 61 / 12 * f[5]
            - 5 / 6 * f[6]
        ) / dx ** 2
        fPrime[-1] = (
            15 / 4 * f[-1]
            - 77 / 6 * f[-2]
            + 107 / 6 * f[-3]
            - 13 * f[-4]
            + 61 / 12 * f[-5]
            - 5 / 6 * f[-6]
        ) / dx ** 2
        fPrime[-2] = (
            15 / 4 * f[-2]
            - 77 / 6 * f[-3]
            + 107 / 6 * f[-4]
            - 13 * f[-5]
            + 61 / 12 * f[-6]
            - 5 / 6 * f[-7]
        ) / dx ** 2

        # Central points
        for j in range(2, len(f) - 2):
            fPrime[j] = (
                -1 / 12 * f[j - 2]
                + 4 / 3 * f[j - 1]
                - 5 / 2 * f[j]
                + 4 / 3 * f[j + 1]
                - 1 / 12 * f[j + 2]
            ) / dx ** 2

    else:
        raise ValueError("Please input order 1 or 2")

    return fPrime

@jit(nopython=True)
def arbitrary_chi(
    f,
    u_axis,
    k,
    xi,
    v_th,
    n,
    particle_m,
    particle_q,
    phi=1e-5,
    nPoints=1e3,
    inner_range=0.3,
    inner_frac=0.8,
):
    """
    f: array, distribution function of velocities
    u_axis: normalized velocity axis
    k: wavenumber
    xi: normalized phase velocities
    v_th: thermal velocity of the distribution, used to normalize the velocity axis
    n: ion density
    m: particle mass in atomic mass units
    q: particle charge in fundamental charges
    phi: standoff variable used to avoid singularities
    nPoints: number of points used in integration
    deltauMax: maximum distance on the u axis to integrate to
    """

    # Take f' = df/du and f" = d^2f/d^2u
    fPrime = arbitrary_derivative(f=f, x=u_axis, order=1)
    fDoublePrime = arbitrary_derivative(f=f, x=u_axis, order=2)

    # Interpolate f' and f" onto xi
    g = np.interp(xi, u_axis, fPrime)
    gPrime = np.interp(xi, u_axis, fDoublePrime)

    # Set up integration ranges and spacing
    # We need fine divisions near the asymptote, but not at infinity
    """
    #the fractional range of the inner fine divisions near the asymptote
    inner_range = 0.1
    #the fraction of total divisions used in the inner range; should be > inner_range
    inner_frac = 0.8"""

    outer_frac = 1 - inner_frac

    m_inner = np.linspace(0, inner_range, int(np.floor(nPoints / 2 * inner_frac)))
    p_inner = np.linspace(0, inner_range, int(np.ceil(nPoints / 2 * inner_frac)))
    m_outer = np.linspace(inner_range, 1, int(np.floor(nPoints / 2 * outer_frac)))
    p_outer = np.linspace(inner_range, 1, int(np.ceil(nPoints / 2 * outer_frac)))

    m = np.concatenate((m_inner, m_outer))
    p = np.concatenate((p_inner, p_outer))

    # Generate integration sample points that avoid the singularity
    # Create empty arrays of the correct size
    zm = np.empty((len(xi), len(m)))
    zp = np.empty((len(xi), len(p)))

    # Compute maximum width of integration range based on the size of the input array of normalized velocities
    deltauMax = max(u_axis) - min(u_axis)

    # Compute arrays of offsets to add to the central points in xi
    m_point_array = phi + m * deltauMax
    p_point_array = phi + p * deltauMax

    # Intervals between each integration point
    m_deltas = np.append(m_point_array[1:] - m_point_array[:-1], [0])
    p_deltas = np.append(p_point_array[1:] - p_point_array[:-1], [0])

    # The integration points on u
    for i in range(len(xi)):
        zm[i, :] = xi[i] + m_point_array
        zp[i, :] = xi[i] - p_point_array

    # interpolate to get f at the sample points
    gm = np.interp(zm, u_axis, fPrime)
    gp = np.interp(zp, u_axis, fPrime)

    # Evaluate integral (df/du / (u - xi)) du
    M_array = m_deltas * gm / m_point_array
    P_array = p_deltas * gp / p_point_array

    integral = (
        np.sum(M_array, axis=1)
        - np.sum(P_array, axis=1)
        + 1j * np.pi * g
        + 2 * phi * gPrime
    )

    # Convert mass and charge to SI units
    m_SI = particle_m * 1.6605e-27
    q_SI = particle_q * 1.6022e-19

    # Compute plasma frequency squared
    wpl2 = n * q_SI ** 2 / (m_SI * 8.8541878e-12)

    # Coefficient
    coefficient = -wpl2 / k ** 2 / (np.sqrt(2) * v_th)

    return coefficient * integral
        
        
    
    


def arbitrary_fast_spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,
    ifn,
    n,
    notches: u.nm = None,
    efract: np.ndarray = np.array([1.0]),
    ifract: np.ndarray = np.array([1.0]),
    ion_z=np.array([1]),
    ion_m=np.array([1]),
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
    return_chi = False
) -> Union[
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
]:
    
    # Ensure unit vectors are normalized
    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Normal vector along k, assume all velocities lie in this direction

    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)  # normalization

    # print("k_vec:", k_vec)

    # Compute drift velocities and thermal speeds for all electrons and ion species
    electron_vel = []  # drift velocities (vector)
    electron_vel_1d = [] # 1D drift velocities (scalar)
    vTe = []  # thermal speeds (scalar)

    # Note that we convert to SI, strip units, then reintroduce them outside the loop to get the correct objects
    for i, fn in enumerate(efn):
        v_axis = e_velocity_axes[i]
        moment1_integrand = np.multiply(fn, v_axis)
        bulk_velocity = np.trapz(moment1_integrand, v_axis)
        moment2_integrand = np.multiply(fn, (v_axis - bulk_velocity) ** 2)
        electron_vel.append(bulk_velocity * k_vec / np.linalg.norm(k_vec))
        electron_vel_1d.append(bulk_velocity)
        vTe.append(np.sqrt(np.trapz(moment2_integrand, v_axis)))

    electron_vel = np.array(electron_vel)
    electron_vel_1d = np.array(electron_vel_1d)
    vTe = np.array(vTe)

    # print("electron_vel:", electron_vel)
    # print("electron_vel_1d:", electron_vel_1d)
    # print("vTe:", vTe)

    ion_vel = []
    ion_vel_1d = []
    vTi = []
    for i, fn in enumerate(ifn):
        v_axis = i_velocity_axes[i]
        moment1_integrand = np.multiply(fn, v_axis)
        bulk_velocity = np.trapz(moment1_integrand, v_axis)
        moment2_integrand = np.multiply(fn, (v_axis - bulk_velocity) ** 2)
        ion_vel.append(bulk_velocity * k_vec / np.linalg.norm(k_vec))
        ion_vel_1d.append(bulk_velocity)
        vTi.append(np.sqrt(np.trapz(moment2_integrand, v_axis)))

    ion_vel = np.array(ion_vel)
    ion_vel_1d = np.array(ion_vel_1d)
    vTi = np.array(vTi)

    # print("ion_vel:", ion_vel)
    # print("ion_vel_1d:", ion_vel_1d)
    # print("vTi:", vTi)

    # Define some constants
    C = 299792458  # speed of light

    # Calculate plasma parameters

    zbar = np.sum(ifract * ion_z)
    ne = efract * n
    ni = ifract * n / zbar  # ne/zbar = sum(ni)
    # wpe is calculated for the entire plasma (all electron populations combined)
    # wpe = plasma_frequency(n=n, particle="e-").to(u.rad / u.s).value

    wpe = np.sqrt(n * 3182.60735)
    # print("wpe:", wpe)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * C / wavelengths
    wl = 2 * np.pi * C / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl
    # print("w:", w)

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws ** 2 - wpe ** 2) / C
    kl = np.sqrt(wl ** 2 - wpe ** 2) / C

    # Compute the wavenumber shift (required by momentum conservation)
    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks ** 2 + kl ** 2 - 2 * ks * kl * np.cos(scattering_angle))
    # print("k:", k)

    # print("ion_vel:", ion_vel)
    # print("np.outer(k, k_vec).T:", np.outer(k, k_vec).T)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xie = (np.outer(1 / vTe, 1 / k) * w_e) / np.sqrt(2)
    xii = (np.outer(1 / vTi, 1 / k) * w_i) / np.sqrt(2)

    # Calculate the susceptibilities
    # Apply Sheffield (3.3.9) with the following substitutions
    # xi = w / (sqrt2 k v_th), u = v / (sqrt2 v_th)
    # Then arbitrary_chi = -w_pl ** 2 / (2 v_th ** 2 k ** 2) integral (df/du / (u - xi)) du

    # Electron susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i in range(len(efract)):
        chiE[i, :] = arbitrary_chi(
            f=efn[i],
            u_axis=(
                e_velocity_axes[i] - electron_vel_1d[i]
            )
            / (np.sqrt(2) * vTe[i]),
            k=k,
            xi=xie[i],
            v_th=vTe[i],
            n=ne[i],
            particle_m=5.4858e-4,
            particle_q=-1,
            inner_range = inner_range,
            inner_frac = inner_frac
        )
    # print("chiE:", chiE)

    # Ion susceptibilities
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i in range(len(ifract)):
        chiI[i, :] = arbitrary_chi(
            f=ifn[i],
            u_axis=(i_velocity_axes[i] - ion_vel_1d[i])
            / (np.sqrt(2) * vTi[i]),
            k=k,
            xi=xii[i],
            v_th=vTi[i],
            n=ni[i],
            particle_m=ion_m[i],
            particle_q=ion_z[i],
            inner_range = inner_range,
            inner_frac = inner_frac
        )
    # print("chiI:", chiI)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)
    # print("epsilon:", epsilon)

    # Electron component of Skw from Sheffield 5.1.2
    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m] = efract[m] * (
            2
            * np.pi
            / k
            * np.power(np.abs(1 - np.sum(chiE, axis=0) / epsilon), 2)
            * np.interp(
                xie[m],
                (e_velocity_axes[m] - electron_vel_1d[m])
                / (np.sqrt(2) * vTe[m]),
                efn[m],
            )
        )
    # print("econtr:", econtr)

    # ion component
    icontr = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for m in range(ifract.size):
        icontr[m] = ifract[m] * (
            2
            * np.pi
            * ion_z[m]
            / k
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.interp(
                xii[m],
                (i_velocity_axes[m] - ion_vel_1d[m])
                / (np.sqrt(2) * vTi[m]),
                ifn[m],
            )
        )
    # print("icontr:", icontr)

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Convert to power spectrum if option is enabled
    if scattered_power:
        # Conversion factor
        Skw = Skw * (1 + 2 * w / wl) * 2 / (wavelengths ** 2) 
        #this is to convert from S(frequency) to S(wavelength), there is an 
        #extra 2 * pi * c here but that should be removed by normalization
        
    
    #Account for notch(es)
    for myNotch in notches:
        if len(myNotch) != 2:
            raise ValueError("Notches must be pairs of values")
            
        x0 = np.argmin(np.abs(wavelengths - myNotch[0]))
        x1 = np.argmin(np.abs(wavelengths - myNotch[1]))
        Skw[x0:x1] = 0

    # print("S(k,w) before normaliation:", Skw)

    # Normalize result to have integral 1
    Skw = Skw / np.trapz(Skw, wavelengths)

    if not torch.is_tensor(alpha):
        alpha = torch.tensor(alpha, dtype=torch.float64)
    alpha_mean = torch.mean(alpha)

    # print("alpha:", np.mean(alpha))
    # print("Skw:", Skw)
    if return_chi:
        return alpha_mean, Skw, chiE, chiI
    return alpha_mean, Skw
    
    


def arbitrary_spectral_density_arbdist(
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    e_velocity_axes: u.m / u.s,
    i_velocity_axes: u.m / u.s,
    efn: u.nm ** -1,
    ifn: u.nm ** -1,
    n: u.m ** -3,
    notches: u.nm = None,
    efract: np.ndarray = None,
    ifract: np.ndarray = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
    return_chi: bool = False,   # <-- NEW
    ) -> Union[
    Tuple[np.floating, np.ndarray],
    Tuple[np.floating, np.ndarray, np.ndarray, np.ndarray]
]:
    
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)

    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)
        
    #Check for notches
    if notches is None:
        notches = [(0, 0)] * u.nm

    # Convert everything to SI, strip units
    wavelengths = wavelengths.to(u.m).value
    notches = notches.to(u.m).value
    probe_wavelength = probe_wavelength.to(u.m).value
    e_velocity_axes = e_velocity_axes.to(u.m / u.s).value
    i_velocity_axes = i_velocity_axes.to(u.m / u.s).value
    efn = efn.to(u.s / u.m).value
    ifn = ifn.to(u.s / u.m).value
    n = n.to(u.m ** -3).value
    
    
    # Condition ion_species
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]
    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")
    for ii, ion in enumerate(ion_species):
        if isinstance(ion, Particle):
            continue
        ion_species[ii] = Particle(ion)
    
    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(len(ion_species))
    ion_m = np.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = ion_species[i].mass_number
        
    
    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)
    
    
    return arbitrary_fast_spectral_density_arbdist(
        wavelengths, 
        probe_wavelength, 
        e_velocity_axes, 
        i_velocity_axes, 
        efn, 
        ifn,
        n,
        notches,
        efract,
        ifract,
        ion_z,
        ion_m,
        probe_vec,
        scatter_vec,
        scattered_power,
        inner_range,
        inner_frac,
        return_chi=return_chi
        )
    
    
    


def arbitrary_fast_spectral_density_maxwellian(
    wavelengths,
    probe_wavelength,
    n,
    Te,
    Ti,
    notches: np.ndarray = None,
    efract: np.ndarray = np.array([1.0]),
    ifract: np.ndarray = np.array([1.0]),
    ion_z=np.array([1]),
    ion_m=np.array([1]),
    ion_vel=None,
    electron_vel=None,
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    inst_fcn_arr=None,
    scattered_power=False,
):

    """


    Te : np.ndarray
        Temperature in Kelvin


    """

    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3])

    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3])
    
    if notches is None:
        notches = [(0, 0)]

    scattering_angle = np.arccos(np.dot(probe_vec, scatter_vec))

    # Calculate plasma parameters
    # Temperatures here in K!
    vTe = fast_thermal_speed(Te, _m_e)
    vTi = fast_thermal_speed(Ti, ion_m * _m_p)
    zbar = np.sum(ifract * ion_z)

    # Compute electron and ion densities
    ne = efract * n
    ni = ifract * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    wpe = fast_plasma_frequency(n, 1, _m_e)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * np.pi * _c / wavelengths
    wl = 2 * np.pi * _c / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl

    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = np.sqrt(ws ** 2 - wpe ** 2) / _c
    kl = np.sqrt(wl ** 2 - wpe ** 2) / _c

    # Compute the wavenumber shift (required by momentum conservation)\
    # Eq. 1.7.10 in Sheffield
    k = np.sqrt(ks ** 2 + kl ** 2 - 2 * ks * kl * np.cos(scattering_angle))
    # Normal vector along k
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / np.linalg.norm(k_vec)

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components
    w_e = w - np.matmul(electron_vel, np.outer(k, k_vec).T)
    w_i = w - np.matmul(ion_vel, np.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = np.sqrt(2) * wpe / np.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xe = np.outer(1 / vTe, 1 / k) * w_e
    xi = np.outer(1 / vTi, 1 / k) * w_i

    # Calculate the susceptibilities
    chiE = np.zeros([efract.size, w.size], dtype=np.complex128)
    for i, fract in enumerate(efract):
        wpe = fast_plasma_frequency(ne[i], 1, _m_e)
        chiE[i, :] = fast_permittivity_1D_Maxwellian(w_e[i, :], k, vTe[i], wpe)

    # Treatment of multiple species is an extension of the discussion in
    # Sheffield Sec. 5.1
    chiI = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for i, fract in enumerate(ifract):
        wpi = fast_plasma_frequency(ni[i], ion_z[i], ion_m[i] * _m_p)
        chiI[i, :] = fast_permittivity_1D_Maxwellian(w_i[i, :], k, vTi[i], wpi)

    # Calculate the longitudinal dielectric function
    epsilon = 1 + np.sum(chiE, axis=0) + np.sum(chiI, axis=0)

    econtr = np.zeros([efract.size, w.size], dtype=np.complex128)
    for m in range(efract.size):
        econtr[m, :] = efract[m] * (
            2
            * np.sqrt(np.pi)
            / k
            / vTe[m]
            * np.power(np.abs(1 - np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xe[m, :] ** 2)
        )

    icontr = np.zeros([ifract.size, w.size], dtype=np.complex128)
    for m in range(ifract.size):
        icontr[m, :] = ifract[m] * (
            2
            * np.sqrt(np.pi)
            * ion_z[m]
            / k
            / vTi[m]
            * np.power(np.abs(np.sum(chiE, axis=0) / epsilon), 2)
            * np.exp(-xi[m, :] ** 2)
        )

    # Recast as real: imaginary part is already zero
    Skw = np.real(np.sum(econtr, axis=0) + np.sum(icontr, axis=0))

    # Apply an insturment function if one is provided
    if inst_fcn_arr is not None:
        Skw = np.convolve(Skw, inst_fcn_arr, mode="same")

    if scattered_power:
        Skw = Skw * (1 + 2 * w / wl) * 2 / (wavelengths ** 2) 
    
    #Account for notch(es)
    for myNotch in notches:
        if len(myNotch) != 2:
            raise ValueError("Notches must be pairs of values")
            
        x0 = np.argmin(np.abs(wavelengths - myNotch[0]))
        x1 = np.argmin(np.abs(wavelengths - myNotch[1]))
        Skw[x0:x1] = 0
        
    Skw = Skw / np.trapz(Skw, wavelengths)

    return np.mean(alpha), Skw


@validate_quantities(
    wavelengths={"can_be_negative": False, "can_be_zero": False},
    probe_wavelength={"can_be_negative": False, "can_be_zero": False},
    n={"can_be_negative": False, "can_be_zero": False},
    Te={"can_be_negative": False, "equivalencies": u.temperature_energy()},
    Ti={"can_be_negative": False, "equivalencies": u.temperature_energy()},
)
def arbitrary_spectral_density_maxwellian(
    wavelengths: u.nm,
    probe_wavelength: u.nm,
    n: u.m ** -3,
    Te: u.K,
    Ti: u.K,
    notches: u.nm = None,
    efract: np.ndarray = None,
    ifract: np.ndarray = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    electron_vel: u.m / u.s = None,
    ion_vel: u.m / u.s = None,
    probe_vec=np.array([1, 0, 0]),
    scatter_vec=np.array([0, 1, 0]),
    inst_fcn=None,
    scattered_power=False,
) -> Tuple[Union[np.floating, np.ndarray], np.ndarray]:
    r"""
    Calculate the spectral density function for Thomson scattering of a
    probe laser beam by a multi-species Maxwellian plasma.

    This function calculates the spectral density function for Thomson
    scattering of a probe laser beam by a plasma consisting of one or more ion
    species and a one or more thermal electron populations (the entire plasma
    is assumed to be quasi-neutral)

    .. math::
        S(k,\omega) = \sum_e \frac{2\pi}{k}
        \bigg |1 - \frac{\chi_e}{\epsilon} \bigg |^2
        f_{e0,e} \bigg (\frac{\omega}{k} \bigg ) +
        \sum_i \frac{2\pi Z_i}{k}
        \bigg |\frac{\chi_e}{\epsilon} \bigg |^2 f_{i0,i}
        \bigg ( \frac{\omega}{k} \bigg )

    where :math:`\chi_e` is the electron component susceptibility of the
    plasma and :math:`\epsilon = 1 + \sum_e \chi_e + \sum_i \chi_i` is the total
    plasma dielectric  function (with :math:`\chi_i` being the ion component
    of the susceptibility), :math:`Z_i` is the charge of each ion, :math:`k`
    is the scattering wavenumber, :math:`\omega` is the scattering frequency,
    and :math:`f_{e0,e}` and :math:`f_{i0,i}` are the electron and ion velocity
    distribution functions respectively. In this function the electron and ion
    velocity distribution functions are assumed to be Maxwellian, making this
    function equivalent to Eq. 3.4.6 in `Sheffield`_.

    Parameters
    ----------

    wavelengths : `~astropy.units.Quantity`
        Array of wavelengths over which the spectral density function
        will be calculated. (convertible to nm)

    probe_wavelength : `~astropy.units.Quantity`
        Wavelength of the probe laser. (convertible to nm)

    n : `~astropy.units.Quantity`
        Mean (0th order) density of all plasma components combined.
        (convertible to cm^-3.)

    Te : `~astropy.units.Quantity`, shape (Ne, )
        Temperature of each electron component. Shape (Ne, ) must be equal to the
        number of electron populations Ne. (in K or convertible to eV)

    Ti : `~astropy.units.Quantity`, shape (Ni, )
        Temperature of each ion component. Shape (Ni, ) must be equal to the
        number of ion populations Ni. (in K or convertible to eV)

    efract : array_like, shape (Ne, ), optional
        An array-like object where each element represents the fraction (or ratio)
        of the electron population number density to the total electron number density.
        Must sum to 1.0. Default is a single electron component.

    ifract : array_like, shape (Ni, ), optional
        An array-like object where each element represents the fraction (or ratio)
        of the ion population number density to the total ion number density.
        Must sum to 1.0. Default is a single ion species.

    ion_species : str or `~plasmapy.particles.Particle`, shape (Ni, ), optional
        A list or single instance of `~plasmapy.particles.Particle`, or strings
        convertible to `~plasmapy.particles.Particle`. Default is ``'H+'``
        corresponding to a single species of hydrogen ions.

    electron_vel : `~astropy.units.Quantity`, shape (Ne, 3), optional
        Velocity of each electron population in the rest frame. (convertible to m/s)
        If set, overrides electron_vdir and electron_speed.
        Defaults to a stationary plasma [0, 0, 0] m/s.

    ion_vel : `~astropy.units.Quantity`, shape (Ni, 3), optional
        Velocity vectors for each electron population in the rest frame
        (convertible to m/s). If set, overrides ion_vdir and ion_speed.
        Defaults zero drift for all specified ion species.

    probe_vec : float `~numpy.ndarray`, shape (3, )
        Unit vector in the direction of the probe laser. Defaults to
        ``[1, 0, 0]``.

    scatter_vec : float `~numpy.ndarray`, shape (3, )
        Unit vector pointing from the scattering volume to the detector.
        Defaults to [0, 1, 0] which, along with the default `probe_vec`,
        corresponds to a 90 degree scattering angle geometry.

    inst_fcn : function
        A function representing the instrument function that takes an `~astropy.units.Quantity`
        of wavelengths (centered on zero) and returns the instrument point
        spread function. The resulting array will be convolved with the
        spectral density function before it is returned.

    Returns
    -------
    alpha : float
        Mean scattering parameter, where `alpha` > 1 corresponds to collective
        scattering and `alpha` < 1 indicates non-collective scattering. The
        scattering parameter is calculated based on the total plasma density n.

    Skw : `~astropy.units.Quantity`
        Computed spectral density function over the input `wavelengths` array
        with units of s/rad.

    Notes
    -----

    For details, see "Plasma Scattering of Electromagnetic Radiation" by
    Sheffield et al. `ISBN 978\\-0123748775`_. This code is a modified version
    of the program described therein.

    For a concise summary of the relevant physics, see Chapter 5 of Derek
    Schaeffer's thesis, DOI: `10.5281/zenodo.3766933`_.

    .. _`ISBN 978\\-0123748775`: https://www.sciencedirect.com/book/9780123748775/plasma-scattering-of-electromagnetic-radiation
    .. _`10.5281/zenodo.3766933`: https://doi.org/10.5281/zenodo.3766933
    .. _`Sheffield`: https://doi.org/10.1016/B978-0-12-374877-5.00003-8
    """

    # Validate efract
    if efract is None:
        efract = np.ones(1)
    else:
        efract = np.asarray(efract, dtype=np.float64)

    # Validate ifract
    if ifract is None:
        ifract = np.ones(1)
    else:
        ifract = np.asarray(ifract, dtype=np.float64)

    if electron_vel is None:
        electron_vel = np.zeros([efract.size, 3]) * u.m / u.s

    # Condition the electron velocity keywords
    if ion_vel is None:
        ion_vel = np.zeros([ifract.size, 3]) * u.m / u.s
    
    if notches is None:
        notches = [(0, 0)] * u.nm

    # Condition ion_species
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]
    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")
    for ii, ion in enumerate(ion_species):
        if isinstance(ion, Particle):
            continue
        ion_species[ii] = Particle(ion)

    # Condition Ti
    if Ti.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If Ti.size != len(ion_species), assume same temp. for all species
        Ti = [Ti.value] * len(ion_species) * Ti.unit
    elif Ti.size != len(ion_species):
        raise ValueError(
            f"Got {Ti.size} ion temperatures and expected {len(ion_species)}."
        )

    # Make sure the sizes of ion_species, ifract, ion_vel, and Ti all match
    if (
        (len(ion_species) != ifract.size)
        or (ion_vel.shape[0] != ifract.size)
        or (Ti.size != ifract.size)
    ):
        raise ValueError(
            f"Inconsistent number of species in ifract ({ifract}), "
            f"ion_species ({len(ion_species)}), Ti ({Ti.size}), "
            f"and/or ion_vel ({ion_vel.shape[0]})."
        )

    # Condition Te
    if Te.size == 1:
        # If a single quantity is given, put it in an array so it's iterable
        # If Te.size != len(efract), assume same temp. for all species
        Te = [Te.value] * len(efract) * Te.unit
    elif Te.size != len(efract):
        raise ValueError(
            f"Got {Te.size} electron temperatures and expected {len(efract)}."
        )

    # Make sure the sizes of efract, electron_vel, and Te all match
    if (electron_vel.shape[0] != efract.size) or (Te.size != efract.size):
        raise ValueError(
            f"Inconsistent number of electron populations in efract ({efract.size}), "
            f"Te ({Te.size}), or electron velocity ({electron_vel.shape[0]})."
        )

    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(len(ion_species))
    ion_m = np.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number

    probe_vec = probe_vec / np.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / np.linalg.norm(scatter_vec)

    # Apply the insturment function
    if inst_fcn is not None and callable(inst_fcn):
        # Create an array of wavelengths of the same size as wavelengths
        # but centered on zero
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        inst_fcn_arr = inst_fcn(eval_w)
        inst_fcn_arr *= 1 / np.sum(inst_fcn_arr)
    else:
        inst_fcn_arr = None

    alpha, Skw = arbitrary_fast_spectral_density_maxwellian(
        wavelengths.to(u.m).value,
        probe_wavelength.to(u.m).value,
        n.to(u.m ** -3).value,
        Te.to(u.K).value,
        Ti.to(u.K).value,
        notches = notches.to(u.m).value,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_m = ion_m,
        ion_vel=ion_vel.to(u.m / u.s).value,
        electron_vel=electron_vel.to(u.m / u.s).value,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        inst_fcn_arr=inst_fcn_arr,
        scattered_power=scattered_power,
    )

    return alpha, Skw * u.s / u.rad


# ***************************************************************************
# These functions are necessary to interface scalar Parameter objects with
# the array inputs of spectral_density
# ***************************************************************************


def arbitrary__count_populations_in_params(params, prefix, allow_empty = False):
    """
    Counts the number of entries matching the pattern prefix_i in a
    list of keys
    """
    
    keys = list(params.keys())
    prefixLength = len(prefix)
    
    if allow_empty:
        nParams = 0
        for myKey in keys:
            if myKey[:prefixLength] == prefix:
                nParams = max(nParams, int(myKey[prefixLength + 1:]) + 1)
        
        return nParams
    
    else:
        return len(re.findall(prefix, ",".join(keys)))


def arbitrary__params_to_array(params, prefix, vector=False, allow_empty = False):
    """
    Takes a list of parameters and returns an array of the values corresponding
    to a key, based on the following naming convention:

    Each parameter should be named prefix_i
    Where i is an integer (starting at 0)

    This function allows lmfit.Parameter inputs to be converted into the
    array-type inputs required by the spectral density function

    """

    if vector:
        npop = arbitrary__count_populations_in_params(params, prefix + "_x", allow_empty = allow_empty)
        output = np.zeros([npop, 3])
        for i in range(npop):
            for j, ax in enumerate(["x", "y", "z"]):
                if (prefix + f"_{ax}_{i}") in params:
                    output[i, j] = params[prefix + f"_{ax}_{i}"].value
                else:
                    output[i, j] = None

    else:
        npop = arbitrary__count_populations_in_params(params, prefix, allow_empty = allow_empty)
        output = np.zeros([npop])
        for i in range(npop):
            if prefix + f"_{i}" in params:
                output[i] = params[prefix + f"_{i}"]

    return output


# ***************************************************************************
# Fitting functions
# ***************************************************************************


def arbitrary__scattered_power_model_arbdist(wavelengths, settings=None, **params):
    """
    Non user-facing function for the lmfit model
    wavelengths: list of wavelengths over which scattered power is computed over
    settings: settings for the scattered power function
    eparams: parameters to put into emodel to generate a VDF
    iparams: parameters to put into imodel to generate a VDF
    """
    
    
    # check number of ion species
    
    if "ion_m" in settings:
        nSpecies = len(settings["ion_m"])
    else:
        nSpecies = 1
    
    # Separate params into electron params and ion params
    # Electron params must take the form e_paramName, where paramName is the name of the param in emodel
    # Ion params must take the form i_paramName, where paramName is the name of the param in imodel
    # The electron density n is just passed in as "n" and is treated separately from the other params
    # Velocity array is passed into settings
    eparams = {}
    iparams = [{} for _ in range(nSpecies)]

    # Extract crucial settings of emodel, imodel first
    emodel = settings["emodel"]
    imodel = settings["imodel"]

    # print("emodel:", emodel)
    # print("imodel:", imodel)
    
    ifract = arbitrary__params_to_array(params, "ifract")
    
    
    #ion charges follow params if given, otherwise they are fixed at default values
    ion_z = np.zeros(nSpecies)
    for i in range(nSpecies):
        if "q_" + str(i) in params:
            ion_z[i] = params["q_" + str(i)]
        else:
            ion_z[i] = settings["ion_z"][i]
    
    for myParam in params.keys():
        
        myParam_split = myParam.split("_")
        
        if myParam_split[0] == "e":
            eparams[myParam_split[1]] = params[myParam]        
        elif (len(myParam_split[0])>0) and (myParam_split[0][0] == "i"):
            if myParam_split[0][1:].isnumeric():
                iparams[int(myParam_split[0][1:])][myParam_split[1]] = params[myParam]
        elif myParam_split[0] == "n":
            n = params[myParam]
        
    # Create VDFs from model functions
    ve = settings["e_velocity_axes"]
    vi = settings["i_velocity_axes"]

    fe = emodel(ve, **eparams)
    fi = [None] * nSpecies
    for i in range(nSpecies):
        fi[i] = imodel[i](vi[i], **(iparams[i]))

    # Remove emodel, imodel temporarily to put settings into the scattered power
    settings.pop("emodel")
    settings.pop("imodel")
    settings.pop("ion_z")

    # Call scattered power function
    alpha, model_Pw = arbitrary_fast_spectral_density_arbdist(
        wavelengths=wavelengths,
        n=n * 1e6, #this is so it accepts cm^-3 values by default
        efn=fe,
        ifn=fi,
        scattered_power=True,
        ifract = ifract,
        ion_z = ion_z,
        **settings,
    )

    # print("alpha:", alpha)
    # print("S(k,w):", model_Pw)

    # Put settings back now
    # this is necessary to avoid changing the settings array globally
    settings["emodel"] = emodel
    settings["imodel"] = imodel
    settings["ion_z"] = ion_z 

    return model_Pw


def arbitrary__scattered_power_model_maxwellian(wavelengths, settings=None, **params):
    """
    lmfit Model function for fitting Thomson spectra

    For descriptions of arguments, see the `thomson_model` function.

    """

    wavelengths_unitless = wavelengths.to(u.m).value

    # LOAD FROM SETTINGS
    notches = settings["notches"]
    ion_z = settings["ion_z"]
    ion_m = settings["ion_m"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    electron_vdir = settings["electron_vdir"]
    ion_vdir = settings["ion_vdir"]
    probe_wavelength = settings["probe_wavelength"]
    inst_fcn_arr = settings["inst_fcn_arr"]

    # LOAD FROM PARAMS
    n = params["n"]
    Te = arbitrary__params_to_array(params, "Te")
    Ti = arbitrary__params_to_array(params, "Ti")
    efract = arbitrary__params_to_array(params, "efract")
    ifract = arbitrary__params_to_array(params, "ifract")

    electron_speed = arbitrary__params_to_array(params, "electron_speed")
    ion_speed = arbitrary__params_to_array(params, "ion_speed")

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Convert temperatures from eV to Kelvin (required by fast_spectral_density)
    Te *= 11605
    Ti *= 11605
    
    #Convert density from cm^-3 to m^-3
    n *= 1e6

    alpha, model_Pw = arbitrary_fast_spectral_density_maxwellian(
        wavelengths_unitless,
        probe_wavelength,
        n,
        Te,
        Ti,
        notches = notches,
        efract=efract,
        ifract=ifract,
        ion_z=ion_z,
        ion_m=ion_m,
        electron_vel=electron_vel,
        ion_vel=ion_vel,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        inst_fcn_arr=inst_fcn_arr,
        scattered_power=True,
    )

    # print("alpha:", alpha)
    # print("S(k,w):", model_Pw)

    return model_Pw


def arbitrary_scattered_power_model_arbdist(wavelengths, settings, params):
    """
    User facing fitting function, calls arbitrary__scattered_power_model_arbdist to obtain lmfit model
    """

    

    # Extract crucial settings of emodel, imodel first

    if "emodel" in settings:
        emodel = settings["emodel"]
    else:
        raise ValueError("Missing electron VDF model in settings")

    if "imodel" in settings:
        imodel = settings["imodel"]
    else:
        raise ValueError("Missing ion VDF model in settings")
    
    if "ion_species" in settings:
        nSpecies = len(settings["ion_species"])
    else:
        settings["ion species"] = ["p"]
        nSpecies = 1
        
    if "ifract_0" not in list(params.keys()):
        params.add("ifract_0", value=1.0, vary=False)
        
    
    num_i = arbitrary__count_populations_in_params(params, "ifract")
    
    if num_i > 1:
        nums = ["ifract_" + str(i) for i in range(num_i - 1)]
        nums.insert(0, "1.0")
        params["ifract_" + str(num_i - 1)].expr = " - ".join(nums)
        
    
    
    
    # Separate params into electron params and ion params
    # Electron params must take the form e_paramName, where paramName is the name of the param in emodel
    # Ion params must take the form i_paramName, where paramName is the name of the param in imodel
    # The electron density n is just passed in as "n" and is treated separately from the other params
    # Velocity array is passed into settings
    eparams = {}
    iparams = [{} for _ in range(nSpecies)]
        
    
    
    for myParam in params.keys():
        
        myParam_split = myParam.split("_")
        
        if myParam_split[0] == "e":
            eparams[myParam_split[1]] = params[myParam]
        elif myParam_split[0][0] == "i":
            if myParam_split[0][1:].isnumeric():
                iparams[int(myParam_split[0][1:])][myParam_split[1]] = params[myParam]
        elif myParam_split[0] == "n":
            n = params[myParam]
        elif myParam_split[0] == "z":
            z = params[myParam] #this line does nothing and is just to not have the next else trigger
        else:
            raise ValueError("Param name " + myParam + " invalid, must start with e or i")
        
    # Check that models have correct params as inputs
    
    

    # Param names from the model functions
    emodel_param_names = set(inspect.getfullargspec(emodel)[0])

    # Check if models take in velocity as an input -- this is ignored as a param
    if not ("v" in emodel_param_names):
        raise ValueError("Electron VDF model does not take velocity as input")
    
    emodel_param_names.remove("v")
    eparam_names = set(eparams.keys())
    # Raise errors if params are wrong
    if emodel_param_names != eparam_names:
        raise ValueError("Electron parameters do not match")
    
      
    for i in range(nSpecies):
        imodel_param_names = set(inspect.getfullargspec(imodel[i])[0])
        # print("imodel_param_names:", imodel_param_names)     
        if not ("v" in imodel_param_names):
            raise ValueError("Ion VDF model does not take velocity as input")
        
        imodel_param_names.remove("v")
        iparam_names = set(iparams[i].keys())
        
        if imodel_param_names != iparam_names:
            raise ValueError("Ion species " + str(i) + " parameters do not match")

    
    ion_species = settings["ion_species"]
    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(nSpecies)
    ion_m = np.zeros(nSpecies)
    for i, species in enumerate(ion_species):
        particle = Particle(species)
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number
    settings["ion_z"] = ion_z
    settings["ion_m"] = ion_m
    
    # Remove the ion_species from settings
    
    settings.pop("ion_species")

    model = Model(
        arbitrary__scattered_power_model_arbdist,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings.copy(),
    )
    
    settings["ion_species"] = ion_species

    return model


def arbitrary_scattered_power_model_maxwellian(wavelengths, settings, params):
    """
    Returns a `lmfit.Model` function for Thomson spectral density function


    Parameters
    ----------


    wavelengths : u.Quantity
        Wavelength array


    settings : dict
        A dictionary of non-variable inputs to the spectral density function
        which must include the following:

            - probe_wavelength: Probe wavelength in nm
            - probe_vec : (3,) unit vector in the probe direction
            - scatter_vec: (3,) unit vector in the scattering direction
            - ion_species : list of Particle strings describing each ion species

        and may contain the following optional variables
            - electron_vdir : (e#, 3) array of electron velocity unit vectors
            - ion_vdir : (e#, 3) array of ion velocity unit vectors
            - inst_fcn : A function that takes a wavelength array and represents
                    a spectrometer insturment function.

        These quantities cannot be varied during the fit.


    params : `lmfit.Parameters` object
        A Parameters object that must contains the following variables
            - n: 0th order density in cm^-3
            - Te_e# : Temperature in eV
            - Ti_i# : Temperature in eV

        and may contain the following optional variables
            - efract_e# : Fraction of each electron population (must sum to 1) (optional)
            - ifract_i# : Fraction of each ion population (must sum to 1) (optional)
            - electron_speed_e# : Electron speed in m/s (optional)
            - ion_speed_i# : Ion speed in m/s (optional)

        where i# and e# are the number of electron and ion populations,
        zero-indexed, respectively (eg. 0,1,2...).

        These quantities can be either fixed or varying.


    Returns
    -------

    Spectral density (optimization function)


    """

    # **********************
    # Required settings and parameters
    # **********************
    req_settings = ["probe_wavelength", "probe_vec", "scatter_vec", "ion_species"]
    for k in req_settings:
        if k not in list(settings.keys()):
            raise KeyError(f"{k} was not provided in settings, but is required.")

    req_params = ["n"]
    for k in req_params:
        if k not in list(params.keys()):
            raise KeyError(f"{k} was not provided in parameters, but is required.")

    # **********************
    # Count number of populations
    # **********************
    if "efract_0" not in list(params.keys()):
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in list(params.keys()):
        params.add("ifract_0", value=1.0, vary=False)

    num_e = arbitrary__count_populations_in_params(params, "efract")
    num_i = arbitrary__count_populations_in_params(params, "ifract")

    # **********************
    # Required settings and parameters per population
    # **********************
    req_params = ["Te"]
    for p in req_params:
        for e in range(num_e):
            k = p + "_" + str(e)
            if k not in list(params.keys()):
                raise KeyError(f"{p} was not provided in parameters, but is required.")

    req_params = ["Ti"]
    for p in req_params:
        for i in range(num_i):
            k = p + "_" + str(i)
            if k not in list(params.keys()):
                raise KeyError(f"{p} was not provided in parameters, but is required.")


    # Create arrays of ion Z and mass from particles given
    ion_z = np.zeros(num_i)
    ion_m= np.zeros(num_i)
    for i, species in enumerate(settings["ion_species"]):
        particle = Particle(species)
        ion_z[i] = particle.charge_number
        ion_m[i] = particle.mass_number
    settings["ion_z"] = ion_z
    settings["ion_m"] = ion_m
    

    # Automatically add an expression to the last efract parameter to
    # indicate that it depends on the others (so they sum to 1.0)
    # The resulting expression for the last of three will look like
    # efract_2.expr = "1.0 - efract_0 - efract_1"
    if num_e > 1:
        nums = ["efract_" + str(i) for i in range(num_e - 1)]
        nums.insert(0, "1.0")
        params["efract_" + str(num_e - 1)].expr = " - ".join(nums)

    if num_i > 1:
        nums = ["ifract_" + str(i) for i in range(num_i - 1)]
        nums.insert(0, "1.0")
        params["ifract_" + str(num_i - 1)].expr = " - ".join(nums)

    # **************
    # Electron velocity
    # **************
    electron_speed = np.zeros([num_e])
    for e in range(num_e):
        k = "electron_speed_" + str(e)
        if k in list(params.keys()):
            electron_speed[e] = params[k].value
        else:
            # electron_speed[e] = 0 already
            params.add(k, value=0, vary=False)

    if "electron_vdir" not in list(settings.keys()):
        if np.all(electron_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["electron_vdir"] = np.ones([num_e, 3])
        else:
            raise ValueError(
                "electron_vdir must be set if electron_speeds " "are not all zero."
            )
    # Normalize vdir
    norm = np.linalg.norm(settings["electron_vdir"], axis=-1)
    settings["electron_vdir"] = settings["electron_vdir"] / norm[:, np.newaxis]

    # **************
    # Ion velocity
    # **************
    ion_speed = np.zeros([num_i])
    for i in range(num_i):
        k = "ion_speed_" + str(i)
        if k in list(params.keys()):
            ion_speed[i] = params[k].value
        else:
            # ion_speed[i] = 0 already
            params.add(k, value=0, vary=False)

    if "ion_vdir" not in list(settings.keys()):
        if np.all(ion_speed == 0):
            # vdir is arbitrary in this case because vel is zero
            settings["ion_vdir"] = np.ones([num_i, 3])
        else:
            raise ValueError("ion_vdir must be set if ion_speeds " "are not all zero.")
    # Normalize vdir
    norm = np.linalg.norm(settings["ion_vdir"], axis=-1)
    settings["ion_vdir"] = settings["ion_vdir"] / norm[:, np.newaxis]

    if "inst_fcn" not in list(settings.keys()):
        settings["inst_fcn_arr"] = None
    else:
        # Create inst fcn array from inst_fcn
        inst_fcn = settings["inst_fcn"]
        wspan = (np.max(wavelengths) - np.min(wavelengths)) / 2
        eval_w = np.linspace(-wspan, wspan, num=wavelengths.size)
        inst_fcn_arr = inst_fcn(eval_w)
        inst_fcn_arr *= 1 / np.sum(inst_fcn_arr)
        settings["inst_fcn_arr"] = inst_fcn_arr

    # Convert and strip units from settings if necessary
    val = {"probe_wavelength": u.m}
    for k, unit in val.items():
        if hasattr(settings[k], "unit"):
            settings[k] = settings[k].to(unit).value

    # TODO: raise an exception if the number of any of the ion or electron
    # quantities isn't consistent with the number of that species defined
    # by ifract or efract.

    # Create a lmfit.Model
    # nan_policy='omit' automatically ignores NaN values in data, allowing those
    # to be used to represnt regions of missing data
    # the "settings" dict is an additional kwarg that will be passed to the model function on every call
    model = Model(
        arbitrary__scattered_power_model_maxwellian,
        independent_vars=["wavelengths"],
        nan_policy="omit",
        settings=settings,
    )

    return model

# ----------------------------------------------------------------------------
# Autodiff implementation (from cpu_autodiff_thomson.py)
# ----------------------------------------------------------------------------
# Install torch dependencies
import torch
import torch.nn.functional as F

import astropy.constants as const
import astropy.units as u
import inspect
import numpy as np
import re
import warnings

from lmfit import Model
from typing import List, Tuple, Union, Optional    # Imported Optional

from plasmapy.formulary.dielectric_fast import fast_permittivity_1D_Maxwellian
from plasmapy.formulary.parameters import fast_plasma_frequency, fast_thermal_speed
from plasmapy.particles import Particle, particle_mass
from plasmapy.utils.decorators import validate_quantities

# Make default torch tensor type
torch.set_default_dtype(torch.float64)

_c = const.c.si.value  # Make sure C is in SI units
_e = const.e.si.value
_m_p = const.m_p.si.value
_m_e = const.m_e.si.value

@torch.jit.script
def autodiff_derivative(f: torch.Tensor, x: torch.Tensor, derivative_matrices: Tuple[torch.Tensor, torch.Tensor], order: int):
    dx = x[1]-x[0]

    order1_mat = derivative_matrices[0]
    order2_mat = derivative_matrices[1]

    if order == 1:
        f = (1./dx)*torch.matmul(order1_mat, f) # Use matrix for 1st order derivatives
        return f
    elif order == 2:
        f = (1./dx**2)*torch.matmul(order2_mat, f) # Use matrix for 1st order derivatives
        return f
    else:
        print("You can only choose an order of 1 or 2...")

@torch.jit.script
# Original interpolation function from Lars Du (end of thread): https://github.com/pytorch/pytorch/issues/1552
def autodiff_torch_1d_interp(
    x: torch.Tensor,
    xp: torch.Tensor,
    fp: torch.Tensor,
    # left: Optional[float] = None, #| None = None,
    # right: Optional[float] = None #| None = None,
) -> torch.Tensor:

    """
    One-dimensional linear interpolation for monotonically increasing sample points.

    Returns the one-dimensional piecewise linear interpolant to a function with given discrete data points (xp, fp), evaluated at x.

    Args:
        x: The x-coordinates at which to evaluate the interpolated values.
        xp: 1d sequence of floats. x-coordinates. Must be increasing
        fp: 1d sequence of floats. y-coordinates. Must be same length as xp
        left: Value to return for x < xp[0], default is fp[0]
        right: Value to return for x > xp[-1], default is fp[-1]

    Returns:
        The interpolated values, same shape as x.
    """
    
    left = fp[0]
    right = fp[-1]

    i = torch.clip(torch.searchsorted(xp, x, right=True), 1, len(xp) - 1)

    answer = torch.where(
        x < xp[0],
        left,
        (fp[i - 1] * (xp[i] - x) + fp[i] * (x - xp[i - 1])) / (xp[i] - xp[i - 1]),
    )

    answer = torch.where(x > xp[-1], right, answer)
    return answer

def autodiff_chi(
    f,
    derivative_matrices,
    u_axis,
    k,
    xi,
    v_th,
    n,
    particle_m,
    particle_q,
    phi=1e-5,
    nPoints=1e3,
    inner_range=0.3,
    inner_frac=0.8,
):
    """
    f: array, distribution function of velocities
    u_axis: normalized velocity axis
    k: wavenumber
    xi: normalized phase velocities
    v_th: thermal velocity of the distribution, used to normalize the velocity axis
    n: ion density
    m: particle mass in atomic mass units
    q: particle charge in fundamental charges
    phi: standoff variable used to avoid singularities
    nPoints: number of points used in integration
    deltauMax: maximum distance on the u axis to integrate to
    """

    # Take f' = df/du and f" = d^2f/d^2u
    fPrime = autodiff_derivative(f=f, x=u_axis, derivative_matrices=derivative_matrices, order=1)
    fDoublePrime = autodiff_derivative(f=f, x=u_axis, derivative_matrices=derivative_matrices, order=2)

    # Interpolate f' and f" onto xi
    g = autodiff_torch_1d_interp(xi, u_axis, fPrime)
    gPrime = autodiff_torch_1d_interp(xi, u_axis, fDoublePrime)

    # Set up integration ranges and spacing
    # We need fine divisions near the asymtorchote, but not at infinity

    """
    the fractional range of the inner fine divisions near the asymtorchote
    inner_range = 0.1
    the fraction of total divisions used in the inner range; should be > inner_range
    inner_frac = 0.8
    """
    
    with torch.no_grad():
        outer_frac = torch.tensor([1.]) - inner_frac

        m_inner = torch.linspace(0, inner_range, int(torch.floor(torch.tensor([nPoints / 2 * inner_frac]))))
        p_inner = torch.linspace(0, inner_range, int(torch.ceil(torch.tensor([nPoints / 2 * inner_frac]))))
        m_outer = torch.linspace(inner_range, 1, int(torch.floor(torch.tensor([nPoints / 2 * outer_frac]))))
        p_outer = torch.linspace(inner_range, 1, int(torch.ceil(torch.tensor([nPoints / 2 * outer_frac]))))

        m = torch.cat((m_inner, m_outer))
        p = torch.cat((p_inner, p_outer))

        # Generate integration sample points that avoid the singularity
        # Create empty arrays of the correct size
        zm = torch.zeros((len(xi), len(m)))
        zp = torch.zeros((len(xi), len(p)))
    
        # Compute maximum width of integration range based on the size of the input array of normalized velocities
        deltauMax = max(u_axis) - min(u_axis)
        # print("deltauMax:", deltauMax)

        # Compute arrays of offsets to add to the central points in xi
        m_point_array = phi + m * deltauMax
        p_point_array = phi + p * deltauMax

        m_deltas = torch.cat((m_point_array[1:] - m_point_array[:-1], torch.tensor([0.])))
        p_deltas = torch.cat((p_point_array[1:] - p_point_array[:-1], torch.tensor([0.])))

        # The integration points on u
        for i in range(len(xi)):
            zm[i, :] = xi[i] + m_point_array
            zp[i, :] = xi[i] - p_point_array

    gm = autodiff_torch_1d_interp(zm, u_axis, fPrime)
    gp = autodiff_torch_1d_interp(zp, u_axis, fPrime)
    
    # Evaluate integral (df/du / (u - xi)) du
    M_array = m_deltas * gm / m_point_array
    P_array = p_deltas * gp / p_point_array

    integral = (
        torch.sum(M_array, axis=1)
        - torch.sum(P_array, axis=1)
        + 1j * torch.pi * g
        + 2 * phi * gPrime
    )

    # Convert mass and charge to SI units
    m_SI = torch.tensor([particle_m * 1.6605e-27])
    q_SI = torch.tensor([particle_q * 1.6022e-19])
    
    # Compute plasma frequency squared
    wpl2 = n * torch.square(q_SI) / (m_SI * 8.8541878e-12)

    # Coefficient
    v_th = torch.tensor([v_th])
    coefficient = -1. * wpl2 / k ** 2 / (torch.sqrt(torch.tensor([2])) * v_th)
    
    return coefficient * integral

def autodiff_fast_spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,
    ifn,
    derivative_matrices,
    n,
    notches = None,
    efract = torch.tensor([1.0], dtype=torch.float64),
    ifract = torch.tensor([1.0], dtype=torch.float64),
    ion_z=torch.tensor([1], dtype=torch.float64),
    ion_m=torch.tensor([1], dtype=torch.float64),
    probe_vec=torch.tensor([1, 0, 0]),
    scatter_vec=torch.tensor([0, 1, 0]),
    scattered_power=True,
    inner_range=0.1,
    inner_frac=0.8,
    return_chi = False
):

    # Ensure unit vectors are normalized
    probe_vec = probe_vec / torch.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / torch.linalg.norm(scatter_vec)

    # Normal vector along k, assume all velocities lie in this direction
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / torch.linalg.norm(k_vec)  # normalization

    # Compute drift velocities and thermal speeds for all electrons and ion species
    electron_vel = torch.tensor([])  # drift velocities (vector)
    electron_vel_1d = torch.tensor([]) # 1D drift velocities (scalar)
    vTe = torch.tensor([])  # thermal speeds (scalar)

    # Note that we convert to SI, strip units, then reintroduce them outside the loop to get the correct objects
    for i, fn in enumerate(efn):
        v_axis = e_velocity_axes[i]
        moment1_integrand = torch.multiply(fn, v_axis)
        bulk_velocity = torch.trapz(moment1_integrand, v_axis)
        moment2_integrand = torch.multiply(fn, torch.square(v_axis - bulk_velocity))

        electron_vel = torch.cat((electron_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        electron_vel_1d = torch.cat((electron_vel_1d, torch.tensor([bulk_velocity])))
        vTe = torch.cat((vTe, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    electron_vel = torch.reshape(electron_vel, (len(efn), 3))

    ion_vel = torch.tensor([])
    ion_vel_1d = torch.tensor([])
    vTi = torch.tensor([])

    for i, fn in enumerate(ifn):
        v_axis = i_velocity_axes[i]
        moment1_integrand = torch.multiply(fn, v_axis)
        bulk_velocity = torch.trapz(moment1_integrand, v_axis)
        moment2_integrand = torch.multiply(fn, torch.square(v_axis - bulk_velocity))

        ion_vel = torch.cat((ion_vel, bulk_velocity * k_vec / torch.linalg.norm(k_vec)))
        ion_vel_1d = torch.cat((ion_vel_1d, torch.tensor([bulk_velocity])))
        vTi = torch.cat((vTi, torch.tensor([torch.sqrt(torch.trapz(moment2_integrand, v_axis))])))

    ion_vel = torch.reshape(ion_vel, (len(ifn), 3))

    # Define some constants
    C = torch.tensor([299792458], dtype = torch.float64)  # speed of light

    # Calculate plasma parameters
    zbar = torch.sum(torch.as_tensor(ifract) * ion_z)
    ne = efract * n
    ni = torch.as_tensor(ifract) * n / zbar  # ne/zbar = sum(ni)

    # wpe is calculated for the entire plasma (all electron populations combined)
    # wpe = plasma_frequency(n=n, particle="e-").to(u.rad / u.s).value
    n = n * 3182.60735
    wpe = torch.sqrt(n)

    # Convert wavelengths to angular frequencies (electromagnetic waves, so
    # phase speed is c)
    ws = 2 * torch.pi * C / wavelengths
    wl = 2 * torch.pi * C / probe_wavelength

    # Compute the frequency shift (required by energy conservation)
    w = ws - wl
    
    # Compute the wavenumbers in the plasma
    # See Sheffield Sec. 1.8.1 and Eqs. 5.4.1 and 5.4.2
    ks = torch.sqrt((torch.square(ws) - torch.square(wpe))) / C
    kl = torch.sqrt((torch.square(wl) - torch.square(wpe))) / C

    # Compute the wavenumber shift (required by momentum conservation)
    scattering_angle = torch.arccos(torch.dot(probe_vec, scatter_vec))
    # Eq. 1.7.10 in Sheffield
    k = torch.sqrt((torch.square(ks) + torch.square(kl) - 2 * ks * kl * torch.cos(scattering_angle)))

    # Compute Doppler-shifted frequencies for both the ions and electrons
    # Matmul is simultaneously conducting dot product over all wavelengths
    # and ion components

    w_e = w - torch.matmul(electron_vel, torch.outer(k, k_vec).T)
    w_i = w - torch.matmul(ion_vel, torch.outer(k, k_vec).T)

    # Compute the scattering parameter alpha
    # expressed here using the fact that v_th/w_p = root(2) * Debye length
    alpha = torch.sqrt(torch.tensor([2])) * wpe / torch.outer(k, vTe)

    # Calculate the normalized phase velocities (Sec. 3.4.2 in Sheffield)
    xie = (torch.outer(1 / vTe, 1 / k) * w_e) / torch.sqrt(torch.tensor([2]))
    xii = (torch.outer(1 / vTi, 1 / k) * w_i) / torch.sqrt(torch.tensor([2]))

    # Calculate the susceptibilities
    # Apply Sheffield (3.3.9) with the following substitutions
    # xi = w / (sqrt2 k v_th), u = v / (sqrt2 v_th)
    # Then autodiff_chi = -w_pl ** 2 / (2 v_th ** 2 k ** 2) integral (df/du / (u - xi)) du

    # Electron susceptibilities
    chiE = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for i in range(len(efract)):
        chiE[i, :] = autodiff_chi(
            f=efn[i],
            derivative_matrices=derivative_matrices,
            u_axis=(
                e_velocity_axes[i] - electron_vel_1d[i]
            )
            / (torch.sqrt(torch.tensor(2)) * vTe[i]),
            k=k,
            xi=xie[i],
            v_th=vTe[i],
            n=ne[i],
            particle_m=5.4858e-4,
            particle_q=-1,
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # Ion susceptibilities
    chiI = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for i in range(len(ifract)):
        chiI[i, :] = autodiff_chi(
            f=ifn[i],
            derivative_matrices=derivative_matrices,
            u_axis=(i_velocity_axes[i] - ion_vel_1d[i])
            / (torch.sqrt(torch.tensor([2])) * vTi[i]),
            k=k,
            xi=xii[i],
            v_th=vTi[i],
            n=ni[i],
            particle_m=ion_m[i],
            particle_q=ion_z[i],
            inner_range = inner_range,
            inner_frac = inner_frac
        )

    # Calculate the longitudinal dielectric function
    epsilon = 1 + torch.sum(chiE, axis=0) + torch.sum(chiI, axis=0)

    # Make a for loop to calculate and interpolate necessary arguments ahead of time
    eInterp = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for m in range(len(efract)):
        longArgE = (e_velocity_axes[m] - electron_vel_1d[m]) / (torch.sqrt(torch.tensor(2)) * vTe[m])
        eInterp[m] = autodiff_torch_1d_interp(xie[m], longArgE, efn[m])

    # Electron component of Skw from Sheffield 5.1.2
    econtr = torch.zeros((len(efract), len(w)), dtype=torch.complex128)
    for m in range(len(efract)):
        econtr[m] = efract[m] * (
            2
            * torch.pi
            / k
            * torch.pow(torch.abs(1 - torch.sum(chiE, axis=0) / epsilon), 2)
            * eInterp[m]
        )

    iInterp = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for m in range(len(ifract)):
        longArgI = (i_velocity_axes[m] - ion_vel_1d[m]) / (torch.sqrt(torch.tensor(2)) * vTi[m])
        iInterp[m] = autodiff_torch_1d_interp(xii[m], longArgI, ifn[m])

    # ion component
    icontr = torch.zeros((len(ifract), len(w)), dtype=torch.complex128)
    for m in range(len(ifract)):
        icontr[m] = ifract[m] * (
            2
            * torch.pi
            * ion_z[m]
            / k
            * torch.pow(torch.abs(torch.sum(chiE, axis=0) / epsilon), 2)
            * iInterp[m]
        )

    # Recast as real: imaginary part is already zero
    Skw = torch.real(torch.sum(econtr, axis=0) + torch.sum(icontr, axis=0))

    # Convert to power spectrum if otorchion is enabled
    if scattered_power:
        # Conversion factor
        Skw = Skw * (1 + 2 * w / wl) * 2 / (torch.square(wavelengths))
        #this is to convert from S(frequency) to S(wavelength), there is an
        #extra 2 * pi * c here but that should be removed by normalization

    # Work under assumption only EPW wavelengths require notch(es)
    if notches != None:
        # Account for notch(es) in differentiable manner
        bools = torch.ones(len(Skw), dtype = torch.bool)
        for i, j in enumerate(notches):
            if len(j) != 2:
                raise ValueError("Notches must be pairs of values")
            x0 = torch.argmin(torch.abs(wavelengths - j[0]))
            x1 = torch.argmin(torch.abs(wavelengths - j[-1]))
            bools[x0:x1] = False
        Skw = torch.mul(Skw, bools)

    # Normalize result to have integral 1
    Skw = Skw / torch.trapz(Skw, wavelengths)

    if return_chi:
        return torch.mean(alpha), Skw, chiE, chiI
    return torch.mean(alpha), Skw 

def autodiff_spectral_density_arbdist(
    wavelengths,
    probe_wavelength,
    e_velocity_axes,
    i_velocity_axes,
    efn,
    ifn,
    derivative_matrices,
    n,
    notches = None,
    efract = None,
    ifract = None,
    ion_species: Union[str, List[str], Particle, List[Particle]] = "p",
    probe_vec=torch.tensor([1, 0, 0]),
    scatter_vec=torch.tensor([0, 1, 0]),
    scattered_power=False,
    inner_range=0.1,
    inner_frac=0.8,
    return_chi = False
):

    from plasmapy.particles import Particle
    from typing import List, Union

    # --- Condition ion_species ---
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]

    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")

    # Convert all entries to Particle instances
    ion_species = [
        Particle(ion) if isinstance(ion, str) else ion
        for ion in ion_species
    ]

    ion_species: List[Particle] 
    
    if efract is None:
        efract = torch.ones(1)

    if ifract is None:
        ifract = torch.ones(1)
        
    #Check for notches
    if notches is None:
        notches = torch.tensor([[520, 540]]) # * u.nm
    
    # Condition ion_species
    if isinstance(ion_species, (str, Particle)):
        ion_species = [ion_species]
    if len(ion_species) == 0:
        raise ValueError("At least one ion species needs to be defined.")
    for ii, ion in enumerate(ion_species):
        if isinstance(ion, Particle):
            continue
        ion_species[ii] = Particle(ion)
    
    # Create arrays of ion Z and mass from particles given
    ion_z = torch.zeros(len(ion_species))
    ion_m = torch.zeros(len(ion_species))
    for i, particle in enumerate(ion_species):
        ion_z[i] = particle.charge_number
        ion_m[i] = ion_species[i].mass_number
        
    probe_vec = probe_vec / torch.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / torch.linalg.norm(scatter_vec)
    
    return autodiff_fast_spectral_density_arbdist(
        wavelengths, 
        probe_wavelength, 
        e_velocity_axes, 
        i_velocity_axes, 
        efn, 
        ifn,
        derivative_matrices,
        n,
        notches,
        efract,
        ifract,
        ion_z,
        ion_m,
        probe_vec,
        scatter_vec,
        scattered_power,
        inner_range,
        inner_frac,
        return_chi = return_chi
        )


# =============================================================================
# Public wrappers / aliases
# =============================================================================

# --- Recommended names you asked for ---
def arbitrary_forwardmodel(*args, **kwargs):
    """
    Forward model for arbitrary VDFs (NumPy/Astropy-units implementation).

    This is an alias for :func:`arbitrary_spectral_density_arbdist`.
    """
    return arbitrary_spectral_density_arbdist(*args, **kwargs)


def autodiff_forwardmodel(*args, **kwargs):
    """
    Forward model for arbitrary VDFs with PyTorch autodiff support.

    This is an alias for :func:`autodiff_spectral_density_arbdist`.
    """
    return autodiff_spectral_density_arbdist(*args, **kwargs)


# --- Backwards-compatible defaults (keep thomsonVDF behavior) ---
spectral_density_arbdist = arbitrary_spectral_density_arbdist
fast_spectral_density_arbdist = arbitrary_fast_spectral_density_arbdist
spectral_density_maxwellian = arbitrary_spectral_density_maxwellian
fast_spectral_density_maxwellian = arbitrary_fast_spectral_density_maxwellian

scattered_power_model_arbdist = arbitrary_scattered_power_model_arbdist
scattered_power_model_maxwellian = arbitrary_scattered_power_model_maxwellian

# --- Explicit autodiff aliases (opt-in) ---
spectral_density_arbdist_autodiff = autodiff_spectral_density_arbdist
fast_spectral_density_arbdist_autodiff = autodiff_fast_spectral_density_arbdist

# --- Helper aliases in the naming convention you suggested ---
chi_arbitrary_forwardmodel = arbitrary_chi
chi_autodiff_forwardmodel = autodiff_chi


# ----------------------------------------------------------------------------
# Experimental PlasmaPy-style API (from thomson_plasmampy.py)
# ----------------------------------------------------------------------------
# Everything in this section is suffixed with `_experimental` to make it clear
# that these are alternative entry points / API surfaces.

try:
    """
    Defines the Thomson scattering analysis module as part of
    `plasmapy.diagnostics`.
    """

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

    c_si_unitless = const.c.si.value
    e_si_unitless = const.e.si.value
    m_p_si_unitless = const.m_p.si.value
    m_e_si_unitless = const.m_e.si.value


    # TODO: interface for inputting a multi-species configuration could be
    #     simplified using the plasmapy.classes.plasma_base class if that class
    #     included ion and electron drift velocities and information about the ion
    #     atomic species.


    @preserve_signature
    def spectral_density_lite_experimental(
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
        `~plasmapy.diagnostics.thomson.spectral_density_experimental`.  Performs the same
        thermal speed calculations as
        `~plasmapy.diagnostics.thomson.spectral_density_experimental`, but is intended
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
                    ]
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
    @bind_lite_func(spectral_density_lite_experimental)
    def spectral_density_experimental(  # noqa: C901, PLR0912, PLR0915
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
            ions = ParticleList([Particle(ions)])
        # If a list is provided, ensure all values are Particles, then convert
        # to a ParticleList
        elif isinstance(ions, list):
            for ii, ion in enumerate(ions):
                if isinstance(ion, Particle):
                    continue
                ions[ii] = Particle(ion)
            ions = ParticleList(ions)
        else:
            raise TypeError(
                "The type of object provided to the ``ions`` keyword "
                f"is not supported: {type(ions)}"
            )

        # Validate ions
        if len(ions) == 0:
            raise ValueError("At least one ion species needs to be defined.")

        try:
            if sum(ion.charge_number <= 0 for ion in ions):
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
            (len(ions) != ifract.size)
            or (ion_vel.shape[0] != ifract.size)
            or (T_i.size != ifract.size)
        ):
            raise ValueError(
                f"Inconsistent number of ion species in ifract ({ifract}), "
                f"ions ({len(ions)}), T_i ({T_i.size}), "
                f"and/or ion_vel ({ion_vel.shape[0]})."
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
                f"T_e ({T_e.size}), or electron velocity ({electron_vel.shape[0]})."
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
                    f" a {type(instr_func_arr)}"
                )

            if wavelengths.shape != instr_func_arr.shape:
                raise ValueError(
                    "The shape of the array returned from the "
                    f"instr_func ({instr_func_arr.shape}) "
                    "does not match the shape of the wavelengths "
                    f"array ({wavelengths.shape})."
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
                        "the second element."
                    )
        else:
            notch_unitless = None

        alpha, Skw = spectral_density_lite_experimental(
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


    # ***************************************************************************
    # These functions are necessary to interface scalar Parameter objects with
    # the array inputs of spectral_density_experimental
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
        params: dict[str, Any], prefix: str, vector: bool = False
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
        probe_vec = settings["probe_vec"]
        scatter_vec = settings["scatter_vec"]
        electron_vdir = settings["electron_vdir"]
        ion_vdir = settings["ion_vdir"]
        probe_wavelength = settings["probe_wavelength"]
        instr_func_arr = settings["instr_func_arr"]
        notch = settings["notch"]

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

        _alpha, model_Skw = spectral_density_lite_experimental(
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


    def spectral_density_model_experimental(  # noqa: C901, PLR0912, PLR0915
        wavelengths, settings, params
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
                f"'settings' argument: {missing_settings}"
            )

        required_params = {"n"}
        if missing_params := required_params - set(params):
            raise ValueError(
                f"The following required parameters were not provided in the "
                f"'params': {missing_params}"
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
            ["T_e", "T_i", "ion_mu", "ion_z"], [num_e, num_i, num_i, num_i], strict=False
        ):
            for num in range(nums):
                key = f"{p}_{num!s}"
                if key not in params:
                    raise ValueError(
                        f"{p} was not provided in kwarg 'parameters', but is required."
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
                    "params['electron_speed_<#>'] is non-zero)."
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
                    "params['ion_speed_<#>'] is non-zero)."
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
                    f" a {type(instr_func_arr)}"
                )

            if wavelengths.shape != instr_func_arr.shape:
                raise ValueError(
                    "The shape of the array returned from the "
                    f"instr_func ({instr_func_arr.shape}) "
                    "does not match the shape of the wavelengths "
                    f"array ({wavelengths.shape})."
                )

            instr_func_arr *= 1 / np.sum(instr_func_arr)
            settings["instr_func_arr"] = instr_func_arr

            warnings.warn(
                "If an instrument function is included, the data "
                "should not include any `numpy.nan` values. "
                "Instead regions with no data should be removed from "
                "both the data and wavelength arrays using "
                "`numpy.delete`."
            )

        if "notch" not in settings:
            settings["notch"] = None

        # TODO: raise an exception if the number of any of the ion or electron
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


    # Forward-model alias for the experimental API
    experimental_forwardmodel = spectral_density_experimental

except Exception as _thomson_experimental_import_error:  # pragma: no cover
    def _experimental_unavailable(*args, **kwargs):
        raise ImportError(
            "Experimental Thomson functions could not be imported/initialized. "
            "Original error: "
            f"{_thomson_experimental_import_error!r}"
        )

    spectral_density_experimental = _experimental_unavailable
    spectral_density_lite_experimental = _experimental_unavailable
    spectral_density_model_experimental = _experimental_unavailable
    experimental_forwardmodel = _experimental_unavailable


# ----------------------------------------------------------------------------
# Unified forward-model dispatcher
# ----------------------------------------------------------------------------
def thomson_forwardmodel(*args, backend: str = "arbitrary", **kwargs):
    """Dispatch to the requested Thomson forward model.

    Parameters
    ----------
    backend : {{'arbitrary', 'autodiff', 'experimental'}}
        Selects which forward model to run.

    Notes
    -----
    This is a thin convenience wrapper. It does not modify inputs.
    """
    backend_l = (backend or "arbitrary").lower()
    if backend_l in {"autodiff", "torch"}:
        return autodiff_forwardmodel(*args, **kwargs)
    if backend_l in {"experimental", "plasmapy"}:
        return experimental_forwardmodel(*args, **kwargs)
    return arbitrary_forwardmodel(*args, **kwargs)


# ----------------------------------------------------------------------------
# Backwards-compatibility aliases (deprecated)
# ----------------------------------------------------------------------------
def _deprecated_alias(name: str, new_name: str):
    def _wrapper(*args, **kwargs):
        warnings.warn(
            f"`{name}` is deprecated in this unified module. Use `{new_name}` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return globals()[new_name](*args, **kwargs)
    _wrapper.__name__ = name
    _wrapper.__doc__ = f"DEPRECATED. Use `{new_name}`."
    return _wrapper

# Old naming scheme -> new forwardmodel conventions
spectral_density_arbdist = _deprecated_alias("spectral_density_arbdist", "arbitrary_forwardmodel")
fast_spectral_density_arbdist = _deprecated_alias("fast_spectral_density_arbdist", "arbitrary_forwardmodel")
scattered_power_model_arbdist = _deprecated_alias("scattered_power_model_arbdist", "arbitrary_forwardmodel")

spectral_density_arbdist_autodiff = _deprecated_alias("spectral_density_arbdist_autodiff", "autodiff_forwardmodel")
fast_spectral_density_arbdist_autodiff = _deprecated_alias("fast_spectral_density_arbdist_autodiff", "autodiff_forwardmodel")

# Keep these two legacy Maxwellian helpers available, but discourage their use.
spectral_density_maxwellian = _deprecated_alias("spectral_density_maxwellian", "arbitrary_forwardmodel")
fast_spectral_density_maxwellian = _deprecated_alias("fast_spectral_density_maxwellian", "arbitrary_forwardmodel")
scattered_power_model_maxwellian = _deprecated_alias("scattered_power_model_maxwellian", "arbitrary_forwardmodel")
