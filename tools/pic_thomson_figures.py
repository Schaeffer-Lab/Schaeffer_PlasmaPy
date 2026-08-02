"""
Visual checks for `plasmapy.diagnostics.pic_thomson`.

Renders the behaviour the test suite asserts numerically, so it can be eyeballed
as well. Writes PNGs into ``media/`` (git-ignored) at the repository root.

Usage::

    python tools/pic_thomson_figures.py                 # synthetic figures only
    python tools/pic_thomson_figures.py --osiris PATH   # also read a real run
    python tools/pic_thomson_figures.py --osiris PATH --stride 16

``PATH`` is an OSIRIS output directory, conventionally named ``MS``.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt
from plasmapy.diagnostics import thomson

MEDIA = Path(__file__).resolve().parent.parent / "media"

PROBE_WAVELENGTH = 532 * u.nm
PROBE_VEC = np.array([1.0, 0.0, 0.0])
SCATTER_VEC = np.array([0.0, 1.0, 0.0])


def save(fig, name: str) -> None:
    """Write a figure into the media directory and report where it went."""
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(MEDIA.parent)}")


def maxwellian(v, sigma, drift=0.0):
    """Normalised Maxwellian with standard deviation *sigma*."""
    return np.exp(-((v - drift) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def block(profile, n_time=1, n_x=1):
    """Broadcast a 1-D velocity profile into a ``(n_time, n_v, n_x)`` block."""
    return np.broadcast_to(
        profile[np.newaxis, :, np.newaxis], (n_time, profile.size, n_x)
    ).copy()


# ---------------------------------------------------------------------------
# 1. Conditioning, stage by stage
# ---------------------------------------------------------------------------


def figure_conditioning_stages() -> None:
    """Each conditioning step applied to a noisy, truncated Maxwellian."""
    rng = np.random.default_rng(3)
    sigma = 1.0
    v = np.linspace(-6 * sigma, 6 * sigma, 601)
    clean = maxwellian(v, sigma)
    noisy = np.clip(clean + 0.01 * rng.standard_normal(v.size), -np.inf, None)
    noisy[np.abs(v) > 4.0] = 0.0  # hard truncation, as a PIC diagnostic has

    raw = block(noisy)
    smoothed = pt.smooth_vdf(raw, window=15, iterations=3)
    tapered = pt.taper_vdf_edges(
        smoothed, threshold_frac=0.005, max_taper_bins=40, pedestal_warning=None
    )
    normalised = np.clip(pt.normalize_vdf(tapered, v), pt.DEFAULT_FLOOR, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, log in zip(axes, (False, True), strict=True):
        ax.plot(v, raw[0, :, 0], lw=0.8, alpha=0.55, label="raw (noisy, truncated)")
        ax.plot(v, smoothed[0, :, 0], lw=1.4, label="after smooth_vdf")
        ax.plot(v, tapered[0, :, 0], lw=1.4, ls="--", label="after taper_vdf_edges")
        ax.plot(
            v,
            normalised[0, :, 0] * np.trapezoid(tapered[0, :, 0], v),
            lw=1.0,
            ls=":",
            label="after normalize_vdf (rescaled to overlay)",
        )
        ax.set_xlabel("v / sigma")
        ax.set_ylabel("f")
        if log:
            ax.set_yscale("log")
            ax.set_ylim(1e-8, 1)
            ax.set_title("log scale -- the tails are where the taper acts")
        else:
            ax.set_title("conditioning stages")
            ax.legend(fontsize=8)
    fig.suptitle("pic_thomson: conditioning a noisy PIC distribution", y=1.02)
    save(fig, "01_conditioning_stages.png")


# ---------------------------------------------------------------------------
# 2. The taper pedestal
# ---------------------------------------------------------------------------


def figure_taper_pedestal() -> None:
    """
    Why the unbounded taper is dangerous on a wide velocity grid: the rolloff
    fabricates a pedestal at large |v|, where the v^2 weighting of the second
    moment is largest.
    """
    sigma = 1.0
    half_widths = [4, 6, 8, 12]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

    v = np.linspace(-12 * sigma, 12 * sigma, 2001)
    f = block(maxwellian(v, sigma))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        unbounded = pt.taper_vdf_edges(f, threshold_frac=0.005)
        bounded = pt.taper_vdf_edges(f, threshold_frac=0.005, max_taper_bins=30)
    axes[0].plot(v, f[0, :, 0], lw=1.2, label="input Maxwellian")
    axes[0].plot(v, unbounded[0, :, 0], lw=1.4, label="taper, max_taper_bins=None")
    axes[0].plot(v, bounded[0, :, 0], lw=1.4, ls="--", label="taper, max_taper_bins=30")
    axes[0].set_yscale("log")
    axes[0].set_ylim(1e-10, 1)
    axes[0].set_xlabel("v / sigma")
    axes[0].set_ylabel("f")
    axes[0].set_title("the fabricated pedestal (12 sigma grid)")
    axes[0].legend(fontsize=8)

    widths_unbounded, widths_bounded = [], []
    for half_width in half_widths:
        v = np.linspace(-half_width * sigma, half_width * sigma, 2001)
        f = block(maxwellian(v, sigma))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for store, bins in (
                (widths_unbounded, None),
                (widths_bounded, 30),
            ):
                tapered = pt.normalize_vdf(
                    pt.taper_vdf_edges(f, threshold_frac=0.005, max_taper_bins=bins), v
                )
                store.append(np.sqrt(np.trapezoid(tapered[0, :, 0] * v**2, v)) - 1)

    axes[1].plot(half_widths, 100 * np.array(widths_unbounded), "o-", label="unbounded")
    axes[1].plot(
        half_widths, 100 * np.array(widths_bounded), "s--", label="max_taper_bins=30"
    )
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("grid half-width / sigma")
    axes[1].set_ylabel("error in recovered width (%)")
    axes[1].set_title("the forward model reads vTe straight off this")
    axes[1].legend(fontsize=8)
    fig.suptitle("pic_thomson: taper_vdf_edges on a grid wider than the data", y=1.02)
    save(fig, "02_taper_pedestal.png")


# ---------------------------------------------------------------------------
# 3. Driver vs the analytic Maxwellian model
# ---------------------------------------------------------------------------


def figure_driver_vs_analytic() -> None:
    """The end-to-end physics check, drawn."""
    n_e = 5e18 * u.cm**-3
    T_e, T_i = 100 * u.eV, 50 * u.eV
    wavelengths = np.linspace(480, 590, 351) * u.nm

    def sigma_of(T, mass):
        energy = T.to(u.J, equivalencies=u.temperature_energy())
        return float(np.sqrt(energy / mass).to_value(u.m / u.s))

    sigma_e, sigma_i = sigma_of(T_e, const.m_e), sigma_of(T_i, const.m_p)
    v_e = np.linspace(-8 * sigma_e, 8 * sigma_e, 4001)
    v_i = np.linspace(-8 * sigma_i, 8 * sigma_i, 4001)

    electrons = pt.from_arrays(block(maxwellian(v_e, sigma_e)), v_e, [0.0], [0.0], "e-")
    ions = pt.from_arrays(block(maxwellian(v_i, sigma_i)), v_i, [0.0], [0.0], "p+")

    conditioning = {"taper_threshold": 1e-4, "max_taper_bins": 50}
    spectrogram = pt.spectra_from_phase_spaces(
        electrons,
        [ions],
        position=0.0,
        reference_density=n_e,
        probe_wavelength=PROBE_WAVELENGTH,
        epw_wavelengths=wavelengths,
        probe_vec=PROBE_VEC,
        scatter_vec=SCATTER_VEC,
        electron_conditioning=conditioning,
        ion_conditioning=conditioning,
        scattered_power=False,
        progress=False,
    )
    _, analytic = thomson.spectral_density(
        wavelengths,
        PROBE_WAVELENGTH,
        n_e,
        T_e=T_e,
        T_i=T_i,
        ions=["p+"],
        probe_vec=PROBE_VEC,
        scatter_vec=SCATTER_VEC,
    )

    axis = wavelengths.to_value(u.m)
    driver = spectrogram.epw[0] / np.trapezoid(spectrogram.epw[0], axis)
    analytic = np.asarray(
        analytic.value if isinstance(analytic, u.Quantity) else analytic, dtype=float
    )
    analytic = analytic / np.trapezoid(analytic, axis)
    l1 = np.trapezoid(np.abs(driver - analytic), axis)

    nm = wavelengths.to_value(u.nm)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True, height_ratios=[3, 1])
    axes[0].plot(nm, analytic, lw=2.2, alpha=0.65, label="thomson.spectral_density")
    axes[0].plot(nm, driver, lw=1.2, ls="--", label="pic_thomson driver (VDF model)")
    axes[0].set_ylabel("S(k, w), normalised")
    axes[0].set_title(
        f"Maxwellian plasma: n={n_e:.0e}, Te={T_e:.0f}, Ti={T_i:.0f}  "
        f"(normalised L1 difference {l1:.3f})"
    )
    axes[0].legend()
    axes[1].plot(nm, driver - analytic, lw=1.0, color="crimson")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_xlabel("wavelength (nm)")
    axes[1].set_ylabel("residual")
    save(fig, "03_driver_vs_analytic.png")


# ---------------------------------------------------------------------------
# 4-6. Real OSIRIS data
# ---------------------------------------------------------------------------


def read_osiris(
    ms_path: Path, stride: int, reference_density, args
) -> tuple[pt.PICPhaseSpace, list[pt.PICPhaseSpace]]:
    """Read the electron and ion phase spaces from an OSIRIS run."""
    available = sorted((ms_path / "PHA" / "p1x1" / "e").glob("p1x1-e-*.h5"))
    dumps = list(range(0, len(available), stride))
    print(f"  reading {len(dumps)} of {len(available)} dumps (stride {stride})")

    electrons = pt.read_osiris_phase_space(
        ms_path,
        "p1x1",
        "e",
        reference_density=reference_density,
        is_electron=True,
        timesteps=dumps,
    )
    ions = []
    for species, label in (("cham", args.ion_label), ("targ", args.ion_label)):
        if (ms_path / "PHA" / "p1x1" / species).is_dir():
            ions.append(
                pt.read_osiris_phase_space(
                    ms_path,
                    "p1x1",
                    species,
                    reference_density=reference_density,
                    label=label,
                    timesteps=dumps,
                )
            )
    return electrons, ions


def figure_osiris_phase_space(electrons, ions) -> None:
    """The raw phase space the reader produced, and the density profiles."""
    species = [electrons, *ions]
    fig, axes = plt.subplots(
        2, len(species), figsize=(5.2 * len(species), 7.5), squeeze=False
    )
    step = electrons.f.shape[0] - 1

    for column, phase_space in enumerate(species):
        density = pt.number_density(phase_space.f, phase_space.v)
        image = axes[0][column].imshow(
            phase_space.f[step],
            origin="lower",
            aspect="auto",
            extent=[
                phase_space.x[0] * 1e3,
                phase_space.x[-1] * 1e3,
                phase_space.v[0] * 1e-6,
                phase_space.v[-1] * 1e-6,
            ],
            cmap="inferno",
            vmax=np.percentile(phase_space.f[step], 99.5),
        )
        fig.colorbar(image, ax=axes[0][column], label="f (arb.)")
        axes[0][column].set_title(
            f"{phase_space.meta['species']} ({phase_space.label}) "
            f"at t = {phase_space.t[step] * 1e9:.2f} ns"
        )
        axes[0][column].set_xlabel("x (mm)")
        axes[0][column].set_ylabel("v (1000 km/s)")

        for index in (0, step // 2, step):
            axes[1][column].plot(
                phase_space.x * 1e3,
                density[index],
                lw=1.1,
                label=f"t = {phase_space.t[index] * 1e9:.2f} ns",
            )
        axes[1][column].set_yscale("log")
        axes[1][column].set_xlabel("x (mm)")
        axes[1][column].set_ylabel("n / n0")
        axes[1][column].legend(fontsize=8)
    fig.suptitle("pic_thomson: OSIRIS reader output", y=1.0)
    save(fig, "04_osiris_phase_space.png")


def figure_osiris_taper(electrons, ions) -> None:
    """
    Answer the taper question on real data.

    How much does the unbounded rolloff inflate the distribution width of an
    actual PIC run, and how far does bounding it help?
    """
    species = [electrons, *ions]
    fig, axes = plt.subplots(
        1, len(species), figsize=(5.6 * len(species), 4.2), squeeze=False
    )

    for column, phase_space in enumerate(species):
        smoothed = pt.smooth_vdf(
            phase_space.f,
            window=40,
            iterations=3 if phase_space.is_electron else 0,
        )
        density = pt.number_density(smoothed, phase_space.v)
        step, cell = np.unravel_index(np.argmax(density), density.shape)

        def width(g, step=step, cell=cell, v=phase_space.v):
            mean = np.trapezoid(g[step, :, cell] * v, v)
            return np.sqrt(np.trapezoid(g[step, :, cell] * (v - mean) ** 2, v))

        baseline = width(pt.normalize_vdf(smoothed, phase_space.v))
        options = [None, 100, 50, 20, 10]
        errors = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for bins in options:
                tapered = pt.normalize_vdf(
                    pt.taper_vdf_edges(
                        smoothed, threshold_frac=0.005, max_taper_bins=bins
                    ),
                    phase_space.v,
                )
                errors.append(100 * (width(tapered) / baseline - 1))

        labels = ["none" if b is None else str(b) for b in options]
        bars = axes[0][column].bar(labels, errors, color="steelblue")
        bars[0].set_color("crimson")
        axes[0][column].axhline(0, color="k", lw=0.6)
        axes[0][column].set_xlabel("max_taper_bins")
        axes[0][column].set_ylabel("error in recovered width (%)")
        axes[0][column].set_title(
            f"{phase_space.meta['species']}: grid half-width "
            f"= {phase_space.v[-1] / baseline:.0f} sigma"
        )
    fig.suptitle(
        "pic_thomson: the unbounded taper on real OSIRIS data (densest cell)", y=1.02
    )
    save(fig, "05_osiris_taper.png")


def figure_osiris_spectrogram(electrons, ions, reference_density, position) -> None:
    """A synthetic Thomson spectrogram from the OSIRIS run."""
    # Wide enough to hold the EPW satellites once the density jumps, and a
    # notch narrow enough not to swallow them at the upstream density: at
    # 9e17 cm^-3 they sit only ~8 nm from the probe line.
    epw_wavelengths = np.linspace(440, 640, 500) * u.nm
    iaw_wavelengths = np.linspace(527, 537, 400) * u.nm
    # The taper rolloff has to be bounded, and much more tightly for the ions:
    # their momentum grid is far wider than the actual thermal spread, so an
    # unbounded rolloff inflates the ion width by a factor of tens. See
    # figure 05 and the notes on `taper_vdf_edges`.
    spectrogram = pt.spectra_from_phase_spaces(
        electrons,
        ions,
        position=position,
        reference_density=reference_density,
        probe_wavelength=PROBE_WAVELENGTH,
        epw_wavelengths=epw_wavelengths,
        iaw_wavelengths=iaw_wavelengths,
        epw_notches=[530, 534] * u.nm,
        scatter_vec=[np.cos(np.deg2rad(63)), np.sin(np.deg2rad(63)), 0.0],
        electron_conditioning={
            "smoothing_window": 40,
            "smoothing_iterations": 3,
            "max_taper_bins": 20,
        },
        ion_conditioning={"smoothing_iterations": 0, "max_taper_bins": 3},
        velocity_scale_factor=50,
    )

    time_ns = spectrogram.t * 1e9
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    for ax, (data, axis, name) in zip(
        axes[0],
        (
            (spectrogram.epw, spectrogram.epw_wavelengths, "EPW"),
            (spectrogram.iaw, spectrogram.iaw_wavelengths, "IAW"),
        ),
        strict=True,
    ):
        finite = data[np.isfinite(data).all(axis=1)]
        image = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=[time_ns[0], time_ns[-1], axis[0] * 1e9, axis[-1] * 1e9],
            cmap="inferno",
            vmax=np.percentile(finite, 99) if finite.size else None,
        )
        fig.colorbar(image, ax=ax, label="scattered power (arb., area-normalised)")
        ax.set_xlabel("time (ns)")
        ax.set_ylabel("wavelength (nm)")
        ax.set_title(f"{name} spectrogram at x = {spectrogram.position * 1e3:.2f} mm")

    axes[1][0].semilogy(time_ns, spectrogram.electron_density * 1e-6, lw=1.3)
    axes[1][0].set_xlabel("time (ns)")
    axes[1][0].set_ylabel("n_e (cm^-3)")
    axes[1][0].set_title("electron density at the sampled point")

    axes[1][1].plot(time_ns, spectrogram.alpha_epw, lw=1.3, label="alpha")
    for row, phase_space in enumerate(ions):
        axes[1][1].plot(
            time_ns,
            spectrogram.ifract[row],
            lw=1.0,
            ls="--",
            label=f"ifract, {phase_space.meta['species']}",
        )
    axes[1][1].set_xlabel("time (ns)")
    axes[1][1].set_title("scattering parameter and ion fractions")
    axes[1][1].legend(fontsize=8)

    fig.suptitle("pic_thomson: synthetic Thomson spectra from an OSIRIS run", y=1.0)
    save(fig, "06_osiris_spectrogram.png")


def main() -> None:
    """Render every figure, optionally including the real-data ones."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--osiris", type=Path, default=None, help="OSIRIS output directory (MS)"
    )
    parser.add_argument(
        "--reference-density",
        type=float,
        default=9e17,
        help="simulation reference density in cm^-3 (default: 9e17)",
    )
    parser.add_argument(
        "--stride", type=int, default=32, help="read every Nth dump (default: 32)"
    )
    parser.add_argument(
        "--ion-label",
        default="C 6+",
        help="physical species the ion populations represent",
    )
    parser.add_argument(
        "--position",
        type=float,
        default=None,
        help="sampling position in mm (default: the middle of the domain)",
    )
    args = parser.parse_args()

    print("Synthetic figures:")
    figure_conditioning_stages()
    figure_taper_pedestal()
    figure_driver_vs_analytic()

    if args.osiris is None:
        print("\nNo --osiris path given; skipping the real-data figures.")
        return

    reference_density = args.reference_density * u.cm**-3
    print(f"\nOSIRIS run at {args.osiris}:")
    electrons, ions = read_osiris(args.osiris, args.stride, reference_density, args)
    figure_osiris_phase_space(electrons, ions)
    figure_osiris_taper(electrons, ions)

    position = (
        float(np.mean(electrons.x)) if args.position is None else args.position * 1e-3
    )
    figure_osiris_spectrogram(electrons, ions, reference_density, position)


if __name__ == "__main__":
    main()
