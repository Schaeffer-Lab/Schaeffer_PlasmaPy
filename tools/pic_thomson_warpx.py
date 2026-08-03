r"""
End-to-end run of `plasmapy.diagnostics.pic_thomson` on a WarpX simulation.

Reads every particle plotfile of a run, forward-models Thomson spectra at one
point, and writes figures into ``media/``.

Whether the run scatters collectively depends entirely on its density, which
for a scaled shock simulation can sit anywhere: the same deck run at
``n0 = 1e18`` m^-3 gives ``alpha ~ 1e-5`` at 532 nm, and at ``n0 = 6e26`` gives
``alpha ~ 0.2``. The script reports alpha rather than assuming a regime, and
sizes its wavelength window from the electron distribution it actually finds.

Where ``alpha << 1`` the electron susceptibility vanishes, the ion feature
disappears, and ``S(k, omega)`` reduces to ``(2 pi / k) f_e(omega / k)``. That
gives a validation the collective case cannot: the computed spectrum is
compared against the conditioned electron VDF read through the Doppler map. The
check is reported at any alpha, and is expected to *stop* holding once
collective effects appear -- which is itself informative.

Usage::

    python tools/pic_thomson_warpx.py --diags ~/KinShock2020/runs/R1_paper/diags

    # with the reduced-mass-ratio velocity correction applied
    python tools/pic_thomson_warpx.py --diags ... --velocity-scale-factor 18.36
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt

MEDIA = Path(__file__).resolve().parent.parent / "media"

PROBE_VEC = np.array([1.0, 0.0, 0.0])
SCATTER_VEC = np.array([0.0, 1.0, 0.0])
SCATTERING_ANGLE = np.deg2rad(90.0)
PROBE_WAVELENGTH_M = 532e-9  # overwritten from the command line


def conditioning(args) -> dict:
    """
    Conditioning settings, shared by the driver and the validation check.

    The collective regime needs markedly more smoothing than the
    non-collective one. Where alpha > 1 the spectrum carries a factor
    ``|1 - chi_e/epsilon|^2`` that diverges as ``epsilon -> 0`` at the
    electron-plasma-wave resonance, so shot noise in the VDF -- which enters
    through the derivative of f inside chi -- is amplified into speckle. With
    alpha << 1 the same noise passes through untouched.
    """
    return {
        "smoothing_window": args.smoothing_window,
        "smoothing_iterations": args.smoothing_iterations,
        "max_taper_bins": 20,
    }


def _time_unit(times) -> tuple[float, str]:
    """Pick a readable time unit for an axis spanning *times*."""
    span = float(np.max(times)) if np.size(times) else 0.0
    for scale, unit in ((1e9, "ns"), (1e12, "ps"), (1e15, "fs")):
        if span * scale >= 1.0:
            return scale, unit
    return 1.0, "s"


def save(fig, name: str) -> None:
    """Write a figure into the media directory and report where it went."""
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(MEDIA.parent)}")


def scattering_wavenumber(wavelengths_nm, density) -> np.ndarray:
    r"""
    Wavenumber :math:`k = |\mathbf{k}_s - \mathbf{k}_0|` at each wavelength.

    Both the probe and the scattered light propagate in the plasma, so their
    wavenumbers carry the :math:`\sqrt{\omega^2 - \omega_{pe}^2}` correction.
    Over a window this wide :math:`k` varies by more than an order of magnitude,
    which a constant-:math:`k` Doppler map would get badly wrong.
    """
    speed_of_light = const.c.si.value
    omega_s = 2 * np.pi * speed_of_light / (wavelengths_nm * 1e-9)
    omega_0 = 2 * np.pi * speed_of_light / (PROBE_WAVELENGTH_M)
    omega_pe = np.sqrt(
        density * const.e.si.value**2 / (const.eps0.si.value * const.m_e.si.value)
    )
    k_s = np.sqrt(np.maximum(omega_s**2 - omega_pe**2, 0.0)) / speed_of_light
    k_0 = np.sqrt(max(omega_0**2 - omega_pe**2, 0.0)) / speed_of_light
    k = np.sqrt(k_s**2 + k_0**2 - 2 * k_s * k_0 * np.cos(SCATTERING_ANGLE))
    return omega_s - omega_0, k


def read_species(args) -> tuple[list, list]:
    """Read every electron and ion population of the run."""
    electron_mass = const.m_e.si.value
    timesteps = None
    if args.stride > 1:
        available = sorted(p for p in args.diags.glob("diag1*") if p.is_dir())
        timesteps = list(range(0, len(available), args.stride))

    cache_dir = args.cache_dir
    common = {
        "n_velocity_bins": args.velocity_bins,
        "n_position_bins": args.position_bins,
        "timesteps": timesteps,
    }

    def read(species, mass, label, *, is_electron):
        cache = None if cache_dir is None else cache_dir / f"{species}.npz"
        return pt.read_warpx_phase_space(
            args.diags,
            species,
            mass=mass,
            label=label,
            is_electron=is_electron,
            cache=cache,
            **common,
        )

    electrons = [
        read(name, electron_mass, None, is_electron=True)
        for name in args.electron_species
    ]
    ions = [
        read(
            name,
            args.ion_mass_ratio * electron_mass,
            args.ion_label,
            is_electron=False,
        )
        for name in args.ion_species
    ]
    return electrons, ions


def choose_window(electrons, args) -> u.Quantity:
    """
    Size the wavelength window from the electron distribution itself.

    The Doppler width of a non-collective spectrum is set by the electron
    thermal speed, which here varies by more than an order of magnitude between
    the cold ambient and the heated piston, so a fixed window is no use.
    """
    if args.window is not None:
        return np.linspace(args.window[0], args.window[1], args.wavelength_bins) * u.nm

    widest = 0.0
    for phase_space in electrons:
        density = pt.number_density(phase_space.f, phase_space.v)
        step, cell = np.unravel_index(np.argmax(density), density.shape)
        f = phase_space.f[step, :, cell]
        total = np.trapezoid(f, phase_space.v)
        mean = np.trapezoid(f * phase_space.v, phase_space.v) / total
        width = np.sqrt(
            np.trapezoid(f * (phase_space.v - mean) ** 2, phase_space.v) / total
        )
        widest = max(widest, float(width))

    spread = args.probe_wavelength * 2 * np.sin(SCATTERING_ANGLE / 2) * widest
    spread /= const.c.si.value
    half = min(2.0 * spread, args.probe_wavelength - 40.0)
    print(
        f"  electron sigma up to {widest:.3e} m/s ({widest / const.c.si.value:.3f} c)"
        f" -> 1-sigma Doppler {spread:.0f} nm"
    )
    return (
        np.linspace(
            args.probe_wavelength - half,
            args.probe_wavelength + half,
            args.wavelength_bins,
        )
        * u.nm
    )


def check_non_collective(spectrogram, electrons, args, wavelengths) -> dict:
    r"""
    Compare the spectrum against the Doppler-mapped electron distribution.

    For :math:`\alpha \ll 1` the electron susceptibility vanishes, the shielding
    factor :math:`|1 - \chi_e/\epsilon|^2` goes to one, and the ion feature
    disappears entirely, leaving

    .. math::

       S(k, \omega) \propto \frac{2\pi}{k}\, f_e\!\left(\frac{\omega}{k}\right).

    Reproducing that end to end exercises the reader, the conditioning and the
    forward model on a plasma whose answer is known independently. The
    :math:`2\pi/k` prefactor and the wavelength dependence of :math:`k` both
    matter here, because the window spans most of an octave.
    """
    conditioned = [
        pt.condition_phase_space(
            phase_space.at_position(args.position), **conditioning(args)
        )
        for phase_space in electrons
    ]
    nm = wavelengths.to_value(u.nm)

    predictions = []
    for step in range(spectrogram.n_time):
        if not np.isfinite(spectrogram.epw[step]).all():
            predictions.append(np.full(nm.size, np.nan))
            continue
        shift, k = scattering_wavenumber(nm, spectrogram.electron_density[step])
        velocity = np.divide(shift, k, out=np.full(nm.size, np.inf), where=k > 0)

        total = np.zeros(nm.size)
        for weight, phase_space in zip(
            spectrogram.efract[:, step], conditioned, strict=True
        ):
            if weight <= 0:
                continue
            total += weight * np.interp(
                velocity, phase_space.v, phase_space.f[step, :, 0], left=0.0, right=0.0
            )
        total = np.divide(total, k, out=np.zeros_like(total), where=k > 0)
        area = np.trapezoid(total, nm)
        predictions.append(total / area if area > 0 else np.full(nm.size, np.nan))

    prediction = np.stack(predictions)
    measured = np.stack(
        [
            row / np.trapezoid(row, nm) if np.isfinite(row).all() else row
            for row in spectrogram.epw
        ]
    )

    both = np.isfinite(prediction).all(axis=1) & np.isfinite(measured).all(axis=1)
    l1 = np.full(spectrogram.n_time, np.nan)
    l1[both] = np.trapezoid(np.abs(prediction[both] - measured[both]), nm, axis=1)

    print("\nnon-collective limit: spectrum vs Doppler-mapped electron VDF")
    print(f"  comparable timesteps:       {int(both.sum())} of {spectrogram.n_time}")
    if both.any():
        print(
            f"  normalised L1 difference:   median {np.nanmedian(l1):.4f}"
            f"   90th pct {np.nanpercentile(l1, 90):.4f}"
        )
    return {"prediction": prediction, "measured": measured, "l1": l1}


def figure_phase_space(electrons, ions, args) -> None:
    """Phase space and density profiles for every population."""
    species = electrons + ions
    fig, axes = plt.subplots(
        2, len(species), figsize=(4.6 * len(species), 7.5), squeeze=False
    )
    step = species[0].f.shape[0] - 1

    for column, phase_space in enumerate(species):
        density = pt.number_density(phase_space.f, phase_space.v)
        image = axes[0][column].imshow(
            phase_space.f[step],
            origin="lower",
            aspect="auto",
            extent=[
                phase_space.x[0],
                phase_space.x[-1],
                phase_space.v[0] / const.c.si.value,
                phase_space.v[-1] / const.c.si.value,
            ],
            cmap="inferno",
            vmax=np.percentile(phase_space.f[step], 99.5),
        )
        fig.colorbar(image, ax=axes[0][column])
        axes[0][column].axvline(args.position, color="cyan", lw=0.8, ls="--")
        axes[0][column].set_title(
            f"{phase_space.meta['species']}  t = {phase_space.t[step] * 1e9:.0f} ns",
            fontsize=9,
        )
        axes[0][column].set_xlabel("x (m)")
        axes[0][column].set_ylabel("v / c")

        for index in (0, step // 2, step):
            axes[1][column].semilogy(
                phase_space.x,
                np.clip(density[index], 1e10, None),
                lw=1.0,
                label=f"t = {phase_space.t[index] * 1e9:.0f} ns",
            )
        axes[1][column].axvline(args.position, color="cyan", lw=0.8, ls="--")
        axes[1][column].set_xlabel("x (m)")
        axes[1][column].set_ylabel(r"n (m$^{-3}$)")
        axes[1][column].legend(fontsize=7)

    fig.suptitle("pic_thomson: WarpX reader output (R1_paper)", y=1.0)
    save(fig, f"08_warpx_phase_space{args.tag}.png")


def figure_spectra(spectrogram, wavelengths, comparison, args) -> None:
    """The spectrogram, the scalar tracks, and the non-collective check."""
    nm = wavelengths.to_value(u.nm)
    # These runs span microseconds to picoseconds, so pick the time unit from
    # the data rather than assuming nanoseconds.
    scale, unit = _time_unit(spectrogram.t)
    time_ns = spectrogram.t * scale

    fig = plt.figure(figsize=(15, 9))
    grid = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    ax = fig.add_subplot(grid[0, :2])
    data = spectrogram.epw
    finite = data[np.isfinite(data).all(axis=1)]
    image = ax.imshow(
        data.T,
        origin="lower",
        aspect="auto",
        extent=[time_ns[0], time_ns[-1], nm[0], nm[-1]],
        cmap="inferno",
        vmax=np.percentile(finite, 99) if finite.size else None,
    )
    fig.colorbar(image, ax=ax, label="S(k, w), area-normalised")
    ax.axhline(args.probe_wavelength, color="cyan", lw=0.7, ls="--")
    ax.set_xlabel(f"time ({unit})")
    ax.set_ylabel("wavelength (nm)")
    alpha = np.nanmedian(spectrogram.alpha_epw)
    regime = "non-collective" if alpha < 0.3 else "collective"
    ax.set_title(
        f"Thomson spectrogram at x = {args.position:.4g} m "
        f"({regime}, median alpha = {alpha:.2f})"
    )

    ax = fig.add_subplot(grid[0, 2])
    ax.semilogy(time_ns, spectrogram.electron_density, lw=1.3)
    ax.set_xlabel(f"time ({unit})")
    ax.set_ylabel(r"$n_e$ (m$^{-3}$)")
    ax.set_title("electron density", fontsize=10)

    ax = fig.add_subplot(grid[1, 0])
    for row, label in enumerate(args.electron_species):
        ax.plot(time_ns, spectrogram.efract[row], lw=1.2, label=f"e: {label}")
    for row, label in enumerate(args.ion_species):
        ax.plot(time_ns, spectrogram.ifract[row], lw=1.0, ls="--", label=f"i: {label}")
    ax.set_xlabel(f"time ({unit})")
    ax.set_ylabel("population fraction")
    ax.set_title("piston arrival", fontsize=10)
    ax.legend(fontsize=7)

    ax = fig.add_subplot(grid[1, 1])
    ax.semilogy(time_ns, spectrogram.alpha_epw, lw=1.3)
    ax.axhline(1.0, color="k", lw=0.6, ls=":")
    ax.set_xlabel(f"time ({unit})")
    ax.set_ylabel(r"$\alpha$")
    ax.set_title(r"scattering parameter $\alpha$", fontsize=10)

    ax = fig.add_subplot(grid[1, 2])
    usable = np.nonzero(np.isfinite(comparison["l1"]))[0]
    for step in usable[:: max(1, len(usable) // 3)][:3]:
        line = ax.plot(
            nm,
            comparison["measured"][step],
            lw=1.4,
            alpha=0.7,
            label=f"spectrum, t = {spectrogram.t[step]:.3g} s",
        )[0]
        ax.plot(
            nm,
            comparison["prediction"][step],
            lw=1.0,
            ls="--",
            color=line.get_color(),
            label="Doppler-mapped VDF",
        )
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("normalised")
    ax.set_title(
        f"non-collective check (L1 = {np.nanmedian(comparison['l1']):.3f})",
        fontsize=10,
    )
    ax.legend(fontsize=6)

    fig.suptitle("pic_thomson: synthetic Thomson spectra from WarpX (R1_paper)", y=0.97)
    save(fig, f"09_warpx_spectra{args.tag}.png")


def main() -> None:
    """Read the run, forward-model it, validate, and draw."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--diags",
        type=Path,
        required=True,
        help="WarpX diagnostics directory holding diag1* plotfiles",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=None,
        help="sampling position in m; defaults to the centre of the domain, "
        "since a hard-coded value goes stale the moment a run is resized",
    )
    parser.add_argument(
        "--electron-species", nargs="+", default=["amb_electrons", "piston_electrons"]
    )
    parser.add_argument("--ion-species", nargs="+", default=["amb_ions", "piston_ions"])
    parser.add_argument(
        "--ion-mass-ratio",
        type=float,
        default=100.0,
        help="simulation ion mass in units of the electron mass",
    )
    parser.add_argument(
        "--ion-label", default="p+", help="physical species the ions represent"
    )
    parser.add_argument("--probe-wavelength", type=float, default=532.0)
    parser.add_argument(
        "--window",
        nargs=2,
        type=float,
        default=None,
        help="wavelength window in nm; derived from the electron "
        "thermal speed when omitted",
    )
    parser.add_argument("--wavelength-bins", type=int, default=400)
    parser.add_argument("--velocity-bins", type=int, default=512)
    parser.add_argument("--position-bins", type=int, default=256)
    parser.add_argument(
        "--velocity-scale-factor",
        type=float,
        default=None,
        help="mass-ratio reduction factor R; velocities are divided "
        "by sqrt(R). Left off by default, so the spectra are "
        "in the simulation's own velocity units",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=9,
        help="boxcar width in velocity bins. Raise it for collective spectra; "
        "see the note on conditioning()",
    )
    parser.add_argument("--smoothing-iterations", type=int, default=1)
    parser.add_argument(
        "--tag",
        default="",
        help="suffix for the output filenames, so runs at different probe "
        "wavelengths do not overwrite each other",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="directory to memoise the binned phase spaces in",
    )
    args = parser.parse_args()

    global PROBE_WAVELENGTH_M  # noqa: PLW0603
    PROBE_WAVELENGTH_M = args.probe_wavelength * 1e-9

    if args.cache_dir is not None:
        args.cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"reading {args.diags} ...")
    electrons, ions = read_species(args)
    if args.position is None:
        args.position = float(np.mean(electrons[0].x))
        print(
            f"  no --position given; sampling the domain centre, {args.position:.5g} m"
        )
    # Scale-agnostic formatting: these runs span metres to millimetres and
    # microseconds to picoseconds depending on how the deck is normalised.
    print(
        f"  {electrons[0].shape[0]} timesteps, "
        f"{electrons[0].t[-1]:.3g} s, "
        f"domain {electrons[0].x[-1]:.4g} m"
    )

    wavelengths = choose_window(electrons, args)
    print(f"  window {wavelengths[0].value:.0f} - {wavelengths[-1].value:.0f} nm")

    spectrogram = pt.spectra_from_phase_spaces(
        electrons,
        ions,
        position=args.position,
        # The reader already scales f so that its zeroth moment is a density
        # in m^-3, so there is no further normalisation to undo here.
        reference_density=1 * u.m**-3,
        probe_wavelength=args.probe_wavelength * u.nm,
        epw_wavelengths=wavelengths,
        probe_vec=PROBE_VEC,
        scatter_vec=SCATTER_VEC,
        electron_conditioning=conditioning(args),
        ion_conditioning=conditioning(args),
        velocity_scale_factor=args.velocity_scale_factor,
        # S(k, omega) rather than power per unit wavelength, so the
        # non-collective comparison below is like for like.
        scattered_power=False,
        progress=True,
    )

    print(
        f"\n  n_e:    {np.nanmin(spectrogram.electron_density):.3e} to "
        f"{np.nanmax(spectrogram.electron_density):.3e} m^-3"
    )
    print(
        f"  alpha:  {np.nanmin(spectrogram.alpha_epw):.3e} to "
        f"{np.nanmax(spectrogram.alpha_epw):.3e}"
    )

    comparison = check_non_collective(spectrogram, electrons, args, wavelengths)

    figure_phase_space(electrons, ions, args)
    figure_spectra(spectrogram, wavelengths, comparison, args)


if __name__ == "__main__":
    main()
