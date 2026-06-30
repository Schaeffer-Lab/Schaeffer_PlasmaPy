r"""
Autodiff fitting of a non-Maxwellian distribution to a Thomson spectrum.
========================================================================

This standalone example demonstrates a *gradient-based* fit of an arbitrary
(super-Gaussian) ion velocity distribution function to a synthetic Thomson
scattering spectrum, using a fully differentiable PyTorch reimplementation of
:func:`plasmapy.diagnostics.thomson.spectral_density_arbitrary`.

It is the counterpart to the ``scipy.optimize`` fit demonstrated in
``tests/diagnostics/test_thomson.py``
(``test_spectral_density_arbitrary_supergaussian_fit``).  Both recover the same
non-Maxwellian distribution; this version obtains the parameter gradients by
automatic differentiation rather than finite differences, which can be faster
and more robust for high-dimensional or free-form distribution fits.

PyTorch is an *optional* dependency.  Install it with::

    pip install plasmapy[thomson]

and run this script directly::

    python examples/diagnostics/thomson_arbitrary_vdf_autodiff_fit.py

The script

1. builds a differentiable torch forward model (``spectral_density_arbitrary_torch``),
2. verifies it agrees with the NumPy ``spectral_density_arbitrary`` for a
   Maxwellian, then
3. fits a super-Gaussian ion distribution to a synthetic spectrum with the
   Adam optimizer and reports the recovered parameters.

This script is intentionally kept out of the installed package so that the core
library does not depend on PyTorch.  It exists to evaluate whether an autodiff
backend is worth adding to PlasmaPy proper.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "This example requires PyTorch. Install it with "
        "`pip install plasmapy[thomson]` or `pip install torch`."
    ) from exc

import astropy.constants as const
import astropy.units as u

# SI constants as plain floats
_c = const.c.si.value
_e = const.e.si.value
_eps0 = const.eps0.si.value
_m_e = const.m_e.si.value


# ---------------------------------------------------------------------------
# Differentiable building blocks
# ---------------------------------------------------------------------------
def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Differentiable 1D linear interpolation (``xp`` strictly increasing)."""
    shape = x.shape
    xf = x.reshape(-1)
    i = torch.clip(torch.searchsorted(xp, xf, right=True), 1, len(xp) - 1)
    slope = (fp[i] - fp[i - 1]) / (xp[i] - xp[i - 1])
    out = fp[i - 1] + slope * (xf - xp[i - 1])
    out = torch.where(xf < xp[0], fp[0], out)
    out = torch.where(xf > xp[-1], fp[-1], out)
    return out.reshape(shape)


def _derivative(f: torch.Tensor, dx: torch.Tensor, order: int) -> torch.Tensor:
    """Functional (autograd-safe) 4th-order finite-difference derivative."""
    if order == 1:
        interior = (-f[4:] + 8 * f[3:-1] - 8 * f[1:-3] + f[:-4]) / (12 * dx)
        e0 = (-25 / 12 * f[0] + 4 * f[1] - 3 * f[2] + 4 / 3 * f[3] - f[4] / 4) / dx
        e1 = (-25 / 12 * f[1] + 4 * f[2] - 3 * f[3] + 4 / 3 * f[4] - f[5] / 4) / dx
        en1 = (f[-5] / 4 - 4 / 3 * f[-4] + 3 * f[-3] - 4 * f[-2] + 25 / 12 * f[-1]) / dx
        en2 = (f[-6] / 4 - 4 / 3 * f[-5] + 3 * f[-4] - 4 * f[-3] + 25 / 12 * f[-2]) / dx
    elif order == 2:
        interior = (
            -f[:-4] / 12 + 4 / 3 * f[1:-3] - 5 / 2 * f[2:-2] + 4 / 3 * f[3:-1] - f[4:] / 12
        ) / dx**2
        e0 = (15 / 4 * f[0] - 77 / 6 * f[1] + 107 / 6 * f[2] - 13 * f[3] + 61 / 12 * f[4] - 5 / 6 * f[5]) / dx**2
        e1 = (15 / 4 * f[1] - 77 / 6 * f[2] + 107 / 6 * f[3] - 13 * f[4] + 61 / 12 * f[5] - 5 / 6 * f[6]) / dx**2
        en1 = (15 / 4 * f[-1] - 77 / 6 * f[-2] + 107 / 6 * f[-3] - 13 * f[-4] + 61 / 12 * f[-5] - 5 / 6 * f[-6]) / dx**2
        en2 = (15 / 4 * f[-2] - 77 / 6 * f[-3] + 107 / 6 * f[-4] - 13 * f[-5] + 61 / 12 * f[-6] - 5 / 6 * f[-7]) / dx**2
    else:
        raise ValueError("order must be 1 or 2")
    return torch.cat([e0.reshape(1), e1.reshape(1), interior, en2.reshape(1), en1.reshape(1)])


def _chi_torch(f, u_axis, k, xi, v_th, n, mass, Z, phi=1e-5, n_points=1000, inner_range=0.1, inner_frac=0.8):
    """Differentiable arbitrary-distribution susceptibility (cf. ``_chi_arbitrary``)."""
    du = u_axis[1] - u_axis[0]
    f_prime = _derivative(f, du, 1)
    f_dprime = _derivative(f, du, 2)
    g = _interp(xi, u_axis, f_prime).to(torch.complex128)
    g_prime = _interp(xi, u_axis, f_dprime)

    outer_frac = 1 - inner_frac
    m_inner = torch.linspace(0, inner_range, int(np.floor(n_points / 2 * inner_frac)), dtype=torch.float64)
    p_inner = torch.linspace(0, inner_range, int(np.ceil(n_points / 2 * inner_frac)), dtype=torch.float64)
    m_outer = torch.linspace(inner_range, 1, int(np.floor(n_points / 2 * outer_frac)), dtype=torch.float64)
    p_outer = torch.linspace(inner_range, 1, int(np.ceil(n_points / 2 * outer_frac)), dtype=torch.float64)
    m = torch.cat((m_inner, m_outer))
    p = torch.cat((p_inner, p_outer))

    delta_u_max = torch.max(u_axis) - torch.min(u_axis)
    m_points = phi + m * delta_u_max
    p_points = phi + p * delta_u_max
    m_deltas = torch.cat((m_points[1:] - m_points[:-1], torch.zeros(1, dtype=torch.float64)))
    p_deltas = torch.cat((p_points[1:] - p_points[:-1], torch.zeros(1, dtype=torch.float64)))

    zm = xi[:, None] + m_points[None, :]
    zp = xi[:, None] - p_points[None, :]
    gm = _interp(zm, u_axis, f_prime)
    gp = _interp(zp, u_axis, f_prime)

    integral = (
        torch.sum(m_deltas * gm / m_points, dim=1)
        - torch.sum(p_deltas * gp / p_points, dim=1)
        + 1j * np.pi * g
        + 2 * phi * g_prime
    )
    wpl_sq = Z**2 * _e**2 * n / (_eps0 * mass)
    return -wpl_sq / k**2 / (np.sqrt(2) * v_th) * integral


def spectral_density_arbitrary_torch(
    wavelengths, probe_wavelength, n, e_velocity_axes, i_velocity_axes,
    efn, ifn, efract, ifract, ion_z, ion_mass, probe_vec, scatter_vec,
):
    """Differentiable torch port of ``spectral_density_arbitrary_lite`` (no notch)."""
    probe_vec = probe_vec / torch.linalg.norm(probe_vec)
    scatter_vec = scatter_vec / torch.linalg.norm(scatter_vec)
    k_vec = scatter_vec - probe_vec
    k_vec = k_vec / torch.linalg.norm(k_vec)

    def moments(axes, fns):
        drift, vth = [], []
        for i, fn in enumerate(fns):
            v = axes[i]
            bulk = torch.trapezoid(fn * v, v)
            drift.append(bulk)
            vth.append(torch.sqrt(torch.trapezoid(fn * (v - bulk) ** 2, v)))
        drift = torch.stack(drift)
        vth = torch.stack(vth)
        return torch.outer(drift, k_vec), drift, vth

    electron_vel, _e1d, vTe = moments(e_velocity_axes, efn)
    ion_vel, _i1d, vTi = moments(i_velocity_axes, ifn)

    zbar = torch.sum(ifract * ion_z)
    ne = efract * n
    ni = ifract * n / zbar
    wpe = torch.sqrt(n * _e**2 / (_eps0 * _m_e))

    ws = 2 * np.pi * _c / wavelengths
    wl = 2 * np.pi * _c / probe_wavelength
    w = ws - wl
    ks = torch.sqrt(ws**2 - wpe**2) / _c
    kl = torch.sqrt(wl**2 - wpe**2) / _c
    scattering_angle = torch.arccos(torch.dot(probe_vec, scatter_vec))
    k = torch.sqrt(ks**2 + kl**2 - 2 * ks * kl * torch.cos(scattering_angle))

    w_e = w - torch.matmul(electron_vel, torch.outer(k, k_vec).T)
    w_i = w - torch.matmul(ion_vel, torch.outer(k, k_vec).T)
    alpha = wpe / torch.outer(k, vTe)
    xie = (torch.outer(1 / vTe, 1 / k) * w_e) / np.sqrt(2)
    xii = (torch.outer(1 / vTi, 1 / k) * w_i) / np.sqrt(2)

    chiE = []
    for i in range(len(efract)):
        u_axis = (e_velocity_axes[i] - _e1d[i]) / (np.sqrt(2) * vTe[i])
        chiE.append(_chi_torch(efn[i], u_axis, k, xie[i], vTe[i], ne[i], _m_e, 1))
    chiE = torch.stack(chiE)

    chiI = []
    for i in range(len(ifract)):
        u_axis = (i_velocity_axes[i] - _i1d[i]) / (np.sqrt(2) * vTi[i])
        chiI.append(_chi_torch(ifn[i], u_axis, k, xii[i], vTi[i], ni[i], ion_mass[i], ion_z[i]))
    chiI = torch.stack(chiI)

    epsilon = 1 + torch.sum(chiE, dim=0) + torch.sum(chiI, dim=0)

    econtr = torch.zeros(w.shape, dtype=torch.complex128)
    for j in range(len(efract)):
        u_axis = (e_velocity_axes[j] - _e1d[j]) / (np.sqrt(2) * vTe[j])
        econtr = econtr + efract[j] * (
            2 * np.pi / k
            * torch.abs(1 - torch.sum(chiE, dim=0) / epsilon) ** 2
            * _interp(xie[j], u_axis, efn[j])
        )
    icontr = torch.zeros(w.shape, dtype=torch.complex128)
    for j in range(len(ifract)):
        u_axis = (i_velocity_axes[j] - _i1d[j]) / (np.sqrt(2) * vTi[j])
        icontr = icontr + ifract[j] * (
            2 * np.pi * ion_z[j] / k
            * torch.abs(torch.sum(chiE, dim=0) / epsilon) ** 2
            * _interp(xii[j], u_axis, ifn[j])
        )

    Skw = torch.real(econtr + icontr)
    Skw = Skw / torch.trapezoid(Skw, wavelengths)
    return torch.mean(alpha), Skw


def super_gaussian(v, width, drift, p):
    """Normalized super-Gaussian ``exp(-|(v-drift)/width|**p)``."""
    f = torch.exp(-torch.abs((v - drift) / width) ** p)
    return f / torch.trapezoid(f, v)


def main() -> None:
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(0)

    probe = (532e-9)
    wavelengths = torch.linspace(531e-9, 533e-9, 250)
    n = torch.tensor(1e24)  # m^-3
    ion_z = torch.tensor([1.0])
    ion_mass = torch.tensor([const.m_p.si.value])
    efract = torch.tensor([1.0])
    ifract = torch.tensor([1.0])
    probe_vec = torch.tensor([1.0, 0.0, 0.0])
    scatter_vec = torch.tensor([0.0, 1.0, 0.0])

    Te = (200 * u.eV).to(u.K, equivalencies=u.temperature_energy())
    Ti = (50 * u.eV).to(u.K, equivalencies=u.temperature_energy())
    sig_e = float(np.sqrt(const.k_B * Te / const.m_e).to(u.m / u.s).value)
    sig_i = float(np.sqrt(const.k_B * Ti / const.m_p).to(u.m / u.s).value)
    ve = torch.linspace(-8 * sig_e, 8 * sig_e, 1201)
    vi = torch.linspace(-8 * sig_i, 8 * sig_i, 1201)

    efn = [super_gaussian(ve, np.sqrt(2) * sig_e, 0.0, 2.0)]

    def forward(ifn_list):
        return spectral_density_arbitrary_torch(
            wavelengths, probe, n, [ve], [vi], efn, ifn_list,
            efract, ifract, ion_z, ion_mass, probe_vec, scatter_vec,
        )

    # --- sanity check: the torch model agrees with the NumPy library model ---
    from plasmapy.diagnostics.thomson import spectral_density_arbitrary

    ifn_maxwell = super_gaussian(vi, np.sqrt(2) * sig_i, 0.0, 2.0)
    with torch.no_grad():
        _, skw_torch = forward([ifn_maxwell])
    _, skw_numpy = spectral_density_arbitrary(
        wavelengths.numpy() * u.m,
        probe * u.m,
        n.item() * u.m**-3,
        e_velocity_axes=ve.numpy()[None, :] * u.m / u.s,
        i_velocity_axes=vi.numpy()[None, :] * u.m / u.s,
        efn=efn[0].numpy()[None, :] * u.s / u.m,
        ifn=ifn_maxwell.numpy()[None, :] * u.s / u.m,
        ions=["H+"],
    )
    skw_numpy_v = skw_numpy.to(u.m**-1).value
    rel_diff = np.max(np.abs(skw_torch.numpy() - skw_numpy_v)) / np.max(skw_numpy_v)
    print(f"torch vs NumPy agreement (Maxwellian): max |ΔSkw| / max(Skw) = {rel_diff:.2e}\n")

    # --- synthetic target: a non-Maxwellian (p = 3.2) ion distribution ---
    truth = {"width": np.sqrt(2) * sig_i, "drift": 4e4, "p": 3.2}
    with torch.no_grad():
        ifn_true = super_gaussian(vi, truth["width"], truth["drift"], truth["p"])
        _, target = forward([ifn_true])

    # --- free parameters with autodiff, started from a Maxwellian guess ---
    log_width = torch.tensor(np.log(1.3 * np.sqrt(2) * sig_i), requires_grad=True)
    drift = torch.tensor(0.0, requires_grad=True)
    p = torch.tensor(2.5, requires_grad=True)

    opt = torch.optim.Adam(
        [{"params": [log_width, p], "lr": 3e-2}, {"params": [drift], "lr": 5e3}]
    )

    for step in range(400):
        opt.zero_grad()
        ifn = super_gaussian(vi, torch.exp(log_width), drift, p)
        _, model = forward([ifn])
        loss = torch.nn.functional.mse_loss(model, target) / torch.var(target)
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == 399:
            print(f"step {step:4d}  loss={loss.item():.3e}")

    width_fit = float(torch.exp(log_width))
    print("\nrecovered vs truth")
    print(f"  width: {width_fit:.4e}  (truth {truth['width']:.4e})")
    print(f"  drift: {float(drift):.4e}  (truth {truth['drift']:.4e})")
    print(f"  p    : {float(p):.4f}      (truth {truth['p']:.4f})")


if __name__ == "__main__":
    main()
