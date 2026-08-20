r"""
End-to-end validation of `plasmapy.diagnostics.pic_thomson` against a known answer.

The shock runs this pipeline is aimed at have no analytic spectrum to check
against, so a disagreement there cannot be attributed. This tool removes that
ambiguity: it runs WarpX on a **uniform, stationary, Maxwellian** plasma whose
density and temperatures are set in the deck, pushes the resulting
macroparticles through the whole pipeline, and compares the result against
`~plasmapy.diagnostics.thomson.spectral_density`, which solves the same problem
analytically for exactly that plasma.

Everything between the deck and the spectrum is then under test at once: the
reader's units and weighting, the velocity projection, the conditioning, the
tail model, and the forward model. A failure here is a pipeline bug, not a
physics question.

Two comparisons are made, and they fail differently:

``moments``
    Does the reader recover the deck's :math:`n_e`, :math:`T_e`, :math:`T_i` and
    drift from the macroparticles? This isolates the reader.
``spectrum``
    Does the forward-modelled spectrum match the analytic one for a Maxwellian
    at the **measured** moments? Comparing at the measured rather than the
    nominal moments keeps numerical heating in the PIC run from being charged to
    the pipeline; the drift between the two is reported separately.

Usage::

    python tools/pic_thomson_warpx_validation.py ~/thomson_validation/collective
    python tools/pic_thomson_warpx_validation.py <run> --probe 532 --theta 90
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt
from plasmapy.diagnostics.thomson import spectral_density

MEDIA = Path(__file__).resolve().parent.parent / "media"


def deck_constants(run: Path) -> dict[str, float]:
    """The plasma the deck asked for, read back out of the deck itself."""
    text = (run / "inputs").read_text()
    wanted = {}
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line.startswith("my_constants."):
            continue
        name, _, value = line[len("my_constants.") :].partition("=")
        try:
            wanted[name.strip()] = float(value.strip())
        except ValueError:
            continue
    return wanted


def moments_of(phase_space: pt.PICPhaseSpace, mass: float) -> dict[str, float]:
    """Density, drift and temperature of a phase space, per timestep."""
    f = phase_space.f[:, :, 0]
    v = phase_space.v
    density = np.trapezoid(f, v, axis=1)
    drift = np.trapezoid(f * v, v, axis=1) / density
    width = np.trapezoid(f * (v - drift[:, None]) ** 2, v, axis=1) / density
    return {
        "density": density,
        "drift": drift,
        "T_eV": mass * width / const.e.si.value,
        "sigma": np.sqrt(width),
    }


def probe_geometry(theta_deg: float) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Incident and scattered unit vectors whose scattering vector is along
    :math:`\hat{z}`.

    With :math:`\hat{i} = \cos(\theta/2)\hat{x} - \sin(\theta/2)\hat{z}`
    and :math:`\hat{s} = \cos(\theta/2)\hat{x} + \sin(\theta/2)\hat{z}`
    the difference is :math:`2\sin(\theta/2)\hat{z}`, so **k is along the
    simulation's resolved axis** whatever the scattering angle. That is the
    geometry a shock is analysed in, and it puts any bulk drift where the
    Doppler shift can be checked against it.
    """
    half = np.deg2rad(theta_deg) / 2.0
    return (
        np.array([np.cos(half), 0.0, -np.sin(half)]),
        np.array([np.cos(half), 0.0, +np.sin(half)]),
    )


def read_species(run: Path, args) -> tuple[pt.PICPhaseSpace, pt.PICPhaseSpace]:
    """Bin both species along the scattering vector."""
    common = {
        "path": run / "diags",
        "scatter_direction": k_hat(args),
        # One spatial bin: the plasma is uniform, so every macroparticle in the
        # box is a sample of the same distribution and the statistics of the
        # tail are what the whole test turns on.
        "n_position_bins": 1,
        "n_velocity_bins": args.velocity_bins,
        "progress": False,
    }
    electrons = pt.read_warpx_phase_space(
        species="electrons", mass=const.m_e, is_electron=True, **common
    )
    ions = pt.read_warpx_phase_space(
        species="ions", mass=const.m_p, label="p+", **common
    )
    return electrons, ions


def k_hat(args) -> np.ndarray:
    """Unit scattering vector for this geometry."""
    probe_vec, scatter_vec = probe_geometry(args.theta)
    direction = scatter_vec - probe_vec
    return direction / np.linalg.norm(direction)


def analytic(wavelengths, n_e, T_e, T_i, drift, *, args):
    """`spectral_density` for a Maxwellian plasma at these moments."""
    probe_vec, scatter_vec = probe_geometry(args.theta)
    alpha, skw = spectral_density(
        wavelengths,
        args.probe * u.nm,
        n_e * u.m**-3,
        T_e=T_e * u.eV,
        T_i=np.array([T_i]) * u.eV,
        ions=["p+"],
        # The reader measures the drift along k, so put it back along k.
        electron_vel=np.array([drift * k_hat(args)]) * u.m / u.s,
        ion_vel=np.array([drift * k_hat(args)]) * u.m / u.s,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
    )
    skw = np.asarray(skw, dtype=np.float64)
    if args.scattered_power:
        skw = skw * scattered_power_factor(wavelengths, args.probe * u.nm)
    return float(np.mean(np.asarray(alpha))), skw


def scattered_power_factor(wavelengths, probe_wavelength):
    r"""
    Convert :math:`S(k, \omega)` to scattered power per unit wavelength.

    `~plasmapy.diagnostics.thomson.spectral_density` returns
    :math:`S(k, \omega)` sampled at a set of wavelengths;
    `~plasmapy.diagnostics.thomson.arbitrary_forwardmodel` with
    ``scattered_power`` returns the power a spectrometer would collect per unit
    wavelength, which is that times :math:`(1 + 2\omega/\omega_0)` for the
    scattered power and :math:`2/\lambda^2` for the Jacobian. They are
    different quantities and comparing them directly puts a factor of three of
    tilt across a 280 nm window -- which looks exactly like a pipeline error and
    is not one.
    """
    lam = wavelengths.to_value(u.m)
    lam0 = probe_wavelength.to_value(u.m)
    two_pi_c = 2.0 * np.pi * const.c.si.value
    omega = two_pi_c / lam - two_pi_c / lam0
    return (1.0 + 2.0 * omega / (two_pi_c / lam0)) * 2.0 / lam**2


def normalised(spectrum, wavelengths):
    """Unit area, so shapes are compared rather than absolute intensities."""
    area = np.trapezoid(spectrum, wavelengths)
    return spectrum / area if area > 0 else spectrum


def compare(
    name, recovered, reference, wavelengths, *, probe_nm, ignore_nm=0.0
) -> dict[str, float]:
    """
    Agreement between two normalised spectra.

    The features come in pairs about the probe line and a symmetric plasma makes
    them equal to within shot noise, so a single global peak flips between them
    at random and says nothing. Each side is located separately instead.

    *ignore_nm* blanks a band about the probe line. The EPW window contains the
    central feature as well as the satellites, and the central feature is orders
    of magnitude brighter, so without it every "peak" is the same central spike.
    """
    outside = np.abs(wavelengths - probe_nm) > ignore_nm
    a = normalised(np.asarray(recovered, dtype=np.float64), wavelengths)
    b = normalised(np.asarray(reference, dtype=np.float64), wavelengths)
    result = {"l1": float(np.trapezoid(np.abs(a - b), wavelengths))}
    print(f"  {name:<5} L1 {result['l1']:.4f}")
    for label, side in (
        ("blue", outside & (wavelengths < probe_nm)),
        ("red", outside & (wavelengths > probe_nm)),
    ):
        if not side.any():
            continue
        got = float(wavelengths[side][np.argmax(a[side])])
        want = float(wavelengths[side][np.argmax(b[side])])
        result[f"peak_{label}"] = got
        result[f"peak_{label}_reference"] = want
        print(
            f"        {label:<4} peak {got:9.4f} vs {want:9.4f} nm"
            f"   ({got - want:+.4f} nm)"
        )
    # Integrated power in each band, not a pointwise ratio. Above alpha ~ 3 the
    # satellites are narrow, so a half-nanometre offset between two otherwise
    # correct resonances makes a pointwise ratio read as a factor of 100 while
    # the power under them agrees to a few percent. Bands measure the physics.
    bands = {
        "blue wing": wavelengths < probe_nm - ignore_nm,
        "centre": np.abs(wavelengths - probe_nm) <= ignore_nm,
        "red wing": wavelengths > probe_nm + ignore_nm,
    }
    for band, mask in bands.items():
        if mask.sum() < 2:
            continue
        got = float(np.trapezoid(a[mask], wavelengths[mask]))
        want = float(np.trapezoid(b[mask], wavelengths[mask]))
        result[f"power_{band.replace(' ', '_')}"] = got
        result[f"power_{band.replace(' ', '_')}_reference"] = want
        share = f"{got * 100:5.1f}% vs {want * 100:5.1f}%"
        error = abs(got - want) / want * 100 if want > 0 else float("nan")
        print(f"        {band:<10} power {share}   ({error:.1f}% off)")
    return result


def report_moments(run: Path, electrons, ions) -> dict:
    """Deck versus what the reader got back out of the macroparticles."""
    constants = deck_constants(run)
    c = const.c.si.value
    wanted = {
        "n_e": constants.get("ne", float("nan")),
        "T_e": const.m_e.si.value
        * (constants.get("uth_e", float("nan")) * c) ** 2
        / const.e.si.value,
        "T_i": const.m_p.si.value
        * (constants.get("uth_i", float("nan")) * c) ** 2
        / const.e.si.value,
        "drift": constants.get("udrift_e", 0.0) * c,
    }
    got_e = moments_of(electrons, const.m_e.si.value)
    got_i = moments_of(ions, const.m_p.si.value)

    print(
        "\nmoments: the deck against what the reader recovered "
        "(drift error is relative to v_th)"
    )
    print(f"  {'':<10} {'deck':>12} {'first dump':>12} {'last dump':>12} {'error':>9}")
    rows = {}
    for label, key, got, target in (
        ("n_e (m^-3)", "density", got_e, wanted["n_e"]),
        ("T_e (eV)", "T_eV", got_e, wanted["T_e"]),
        ("T_i (eV)", "T_eV", got_i, wanted["T_i"]),
        ("u_e (m/s)", "drift", got_e, wanted["drift"]),
    ):
        first, last = float(got[key][0]), float(got[key][-1])
        # A zero target has no relative error; for the drift, judge it against
        # the thermal speed, which is the scale shot noise moves it on.
        error = (
            abs(first / target - 1.0)
            if target
            else abs(first) / float(got_e["sigma"][0])
        )
        rows[label] = {"deck": target, "first": first, "last": last, "error": error}
        print(
            f"  {label:<10} {target:12.4e} {first:12.4e} {last:12.4e} "
            f"{error * 100:8.2f}%"
        )
    return rows


def report_spectra(electrons, ions, args) -> dict:
    """Forward-model the read phase space and check it against the analytic one."""
    probe_vec, scatter_vec = probe_geometry(args.theta)
    epw = np.linspace(*args.epw, args.spectral_bins) * u.nm
    iaw = np.linspace(*args.iaw, args.spectral_bins) * u.nm

    spectra = pt.spectra_from_phase_spaces(
        electrons,
        [ions],
        position=float(np.mean(electrons.x)),
        probe_wavelength=args.probe * u.nm,
        epw_wavelengths=epw,
        iaw_wavelengths=iaw,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        electron_conditioning={"smoothing_width": 0.25, "smoothing_iterations": 2},
        ion_conditioning={"smoothing_width": 0.25, "smoothing_iterations": 2},
        scattered_power=args.scattered_power,
        progress=False,
    )

    step = -1
    e_moments = moments_of(electrons, const.m_e.si.value)
    i_moments = moments_of(ions, const.m_p.si.value)
    n_e = float(e_moments["density"][step])
    T_e = float(e_moments["T_eV"][step])
    T_i = float(i_moments["T_eV"][step])
    drift = float(e_moments["drift"][step])

    print(
        f"\nspectra, against spectral_density at the MEASURED moments\n"
        f"  n_e = {n_e:.4e} m^-3, T_e = {T_e:.2f} eV, T_i = {T_i:.2f} eV"
    )
    # A bulk drift moves the whole spectrum, so the bands have to be split
    # about where the feature actually is, not about the probe line. The shift
    # is lambda_0^2 k u / 2 pi c along k.
    lam0 = args.probe * 1e-9
    k = 4.0 * np.pi / lam0 * np.sin(np.deg2rad(args.theta) / 2.0)
    shift = -(lam0**2) * k * drift / (2.0 * np.pi * const.c.si.value) * 1e9
    centre_nm = args.probe + shift
    if abs(shift) > 1e-3:
        print(
            f"  bulk drift shifts the feature by {shift:+.3f} nm, to {centre_nm:.3f} nm"
        )

    alpha_ref, epw_ref = analytic(epw, n_e, T_e, T_i, drift, args=args)
    _, iaw_ref = analytic(iaw, n_e, T_e, T_i, drift, args=args)
    alpha_got = float(spectra.alpha_epw[step]) / np.sqrt(2)
    print(
        f"  alpha (1/k lambda_De): pipeline {alpha_got:.4f} vs analytic "
        f"{alpha_ref:.4f}   error {abs(alpha_got / alpha_ref - 1) * 100:.2f}%"
    )
    results = {
        "alpha_pipeline": alpha_got,
        "alpha_analytic": float(alpha_ref),
        "epw": compare(
            "EPW",
            spectra.epw[step],
            epw_ref,
            epw.to_value(u.nm),
            probe_nm=centre_nm,
            ignore_nm=args.epw_ignore,
        ),
        "iaw": compare(
            "IAW", spectra.iaw[step], iaw_ref, iaw.to_value(u.nm), probe_nm=centre_nm
        ),
    }
    return results, spectra, (epw, epw_ref), (iaw, iaw_ref), step


def figure(run, spectra, epw_pair, iaw_pair, step, *, args) -> Path:
    """Recovered against analytic, on a log scale, plus the residual."""
    epw, epw_ref = epw_pair
    iaw, iaw_ref = iaw_pair
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for column, (grid, reference, got, name) in enumerate(
        (
            (epw.to_value(u.nm), epw_ref, spectra.epw[step], "EPW"),
            (iaw.to_value(u.nm), iaw_ref, spectra.iaw[step], "IAW"),
        )
    ):
        a = normalised(np.asarray(got, dtype=np.float64), grid)
        b = normalised(np.asarray(reference, dtype=np.float64), grid)
        top = axes[0][column]
        plot = top.semilogy if args.log_scale else top.plot
        plot(grid, b, lw=2.0, color="0.55", label="spectral_density (analytic)")
        plot(grid, a, lw=1.0, color="tab:red", label="pic_thomson (from WarpX)")
        top.set(
            xlabel="wavelength (nm)",
            ylabel="S, unit area",
            title=f"{name}: recovered vs known answer",
            ylim=(
                (max(b.max() * 1e-8, 1e-30), b.max() * 3)
                if args.log_scale
                else (0.0, b.max() * 1.15)
            ),
        )
        top.legend(fontsize=8)

        bottom = axes[1][column]
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(b > b.max() * 1e-6, a / b, np.nan)
        bottom.plot(grid, ratio, lw=1.0, color="tab:blue")
        bottom.axhline(1.0, ls=":", color="0.4")
        bottom.set(
            xlabel="wavelength (nm)",
            ylabel="recovered / analytic",
            title=f"{name}: ratio where the analytic spectrum is above 1e-6 of peak",
            ylim=(0.0, 2.0),
        )

    fig.suptitle(
        f"pic_thomson validation on a known WarpX plasma: {run.name}", fontsize=12
    )
    fig.tight_layout()
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / f"12_warpx_validation_{run.name}.png"
    fig.savefig(path, dpi=130)
    return path


def main() -> None:
    """Run the validation and write the report."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="the WarpX run directory")
    parser.add_argument("--probe", type=float, default=532.0, help="probe [nm]")
    parser.add_argument("--theta", type=float, default=90.0, help="scattering angle")
    parser.add_argument("--epw", nargs=2, type=float, default=[440.0, 640.0])
    parser.add_argument("--iaw", nargs=2, type=float, default=[531.0, 533.0])
    parser.add_argument(
        "--epw-ignore",
        type=float,
        default=10.0,
        help="half-width [nm] of the band about the probe left out of the EPW "
        "comparison, so the satellites are measured and not the central feature",
    )
    parser.add_argument(
        "--scattered-power",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compare scattered power per wavelength (the pipeline's default) "
        "rather than bare S(k, omega)",
    )
    parser.add_argument(
        "--log-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log y-axis on the spectra. Off by default: a linear scale is what "
        "a measured spectrum is read on",
    )
    parser.add_argument("--spectral-bins", type=int, default=600)
    parser.add_argument("--velocity-bins", type=int, default=512)
    args = parser.parse_args()

    run = args.run.expanduser().resolve()
    print(f"validating against {run}")
    electrons, ions = read_species(run, args)
    print(f"  {electrons.shape[0]} dump(s), {electrons.v.size} velocity bins")

    moments = report_moments(run, electrons, ions)
    spectra_results, spectra, epw_pair, iaw_pair, step = report_spectra(
        electrons, ions, args
    )
    path = figure(run, spectra, epw_pair, iaw_pair, step, args=args)
    print(f"\n  wrote {path}")
    (run / "validation.json").write_text(
        json.dumps({"moments": moments, "spectra": spectra_results}, indent=2)
    )


if __name__ == "__main__":
    main()
