r"""
End-to-end run of `plasmapy.diagnostics.pic_thomson` on an OSIRIS simulation.

Runs the full pipeline over every timestep of a run, in two configurations:

``legacy``
    Every knob set to reproduce the ``osiris2thomson`` pipeline this module
    replaces, including its unbounded taper rolloff.
``corrected``
    The same, but with the taper bounded, which is how the new pipeline is
    meant to be run. See the notes on `~plasmapy.diagnostics.pic_thomson.
    taper_vdf_edges`.

Comparing the two isolates what the taper fix changes on real data. Pass
``--reference`` as well, and the run is also compared against a ``spectra.hdf5``
written by the old pipeline.

Writes a figure into ``media/`` and prints a numerical report.

Usage::

    python tools/pic_thomson_osiris_comparison.py \\
        --ms ~/OmegaShock/runs/omegashock_w3.5e11_exp/MS \\
        --reference-density 9e17 --velocity-scale-factor 50 --position 5.0

    # with the old pipeline's output to check against
    python tools/pic_thomson_osiris_comparison.py \\
        --ms ~/osiris2thomson/MS --reference ~/osiris2thomson/spectra.hdf5 \\
        --reference-density 1.83e18 --velocity-scale-factor 74 \\
        --position 3.0 --stride 10
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt
from plasmapy.particles import Particle

MEDIA = Path(__file__).resolve().parent.parent / "media"

PROBE_WAVELENGTH = 532 * u.nm
EPW_WAVELENGTHS = np.linspace(432, 632, 500) * u.nm
IAW_WAVELENGTHS = np.linspace(522, 542, 500) * u.nm
SCATTER_VEC = [np.cos(np.deg2rad(63)), np.sin(np.deg2rad(63)), 0.0]

PROTON_RQM = float((const.m_p / const.m_e).decompose().value)


def normalised(spectrum, wavelengths):
    """Rescale each row to unit area, leaving all-NaN rows alone."""
    spectrum = np.asarray(spectrum, dtype=float)
    out = np.full_like(spectrum, np.nan)
    good = np.isfinite(spectrum).all(axis=1)
    areas = np.trapezoid(spectrum[good], wavelengths, axis=1)
    positive = areas > 0
    rows = np.nonzero(good)[0][positive]
    out[rows] = spectrum[rows] / areas[positive][:, np.newaxis]
    return out


def l1_per_row(a, b, wavelengths):
    """Per-timestep L1 difference between two area-normalised spectrograms."""
    errors = np.full(a.shape[0], np.nan)
    both = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    errors[both] = np.trapezoid(np.abs(a[both] - b[both]), wavelengths, axis=1)
    return errors


def peak_wavelength(spectrum, wavelengths, mask):
    """Wavelength of the largest value inside *mask*, per timestep."""
    peaks = np.full(spectrum.shape[0], np.nan)
    good = np.isfinite(spectrum).all(axis=1)
    peaks[good] = wavelengths[mask][np.argmax(spectrum[good][:, mask], axis=1)]
    return peaks


def read_run(
    ms_path: Path, args, reference_density
) -> tuple[pt.PICPhaseSpace, list[pt.PICPhaseSpace]]:
    """Read the electron and ion phase spaces for the run."""
    dumps = None
    if args.stride > 1:
        available = sorted((ms_path / "PHA" / "p1x1" / args.electron).glob("*.h5"))
        dumps = list(range(0, len(available), args.stride))

    read = {"reference_density": reference_density, "timesteps": dumps}
    electrons = pt.read_osiris_phase_space(
        ms_path, "p1x1", args.electron, is_electron=True, **read
    )
    ions = [
        pt.read_osiris_phase_space(ms_path, "p1x1", name, label=label, **read)
        for name, label in zip(args.ions, args.ion_labels, strict=True)
    ]
    return electrons, ions


def run_pipeline(electrons, ions, args, reference_density, position, *, bounded):
    """Run the driver in the legacy or the corrected configuration."""
    return pt.spectra_from_phase_spaces(
        electrons,
        ions,
        position=position,
        reference_density=reference_density,
        probe_wavelength=PROBE_WAVELENGTH,
        epw_wavelengths=EPW_WAVELENGTHS,
        iaw_wavelengths=IAW_WAVELENGTHS,
        epw_notches=args.notch * u.nm,
        scatter_vec=SCATTER_VEC,
        electron_conditioning={
            "smoothing_window": 40,
            "smoothing_iterations": args.smoothing_iterations,
            "max_taper_bins": 20 if bounded else None,
            "pedestal_warning": None,
        },
        ion_conditioning={
            "smoothing_iterations": 0,
            "max_taper_bins": 3 if bounded else None,
            "pedestal_warning": None,
        },
        velocity_scale_factor=args.velocity_scale_factor,
        progress=True,
    )


def load_reference(path: Path) -> dict:
    """Read a spectrogram written by the osiris2thomson pipeline."""
    with h5py.File(path, "r") as handle:
        fractions = handle["ION FRACTIONS"]
        return {
            "epw": handle["SPECTRA/EPW/epws"][()],
            "iaw": handle["SPECTRA/IAW/iaws"][()],
            "alpha": handle["SPECTRA/SCATTERING_PARAMETERS/alpha"][()],
            "density": handle["DENSITY/dens"][()],
            "time": handle["AXES/TIME_AXES/time"][()],
            "ifract": np.stack(
                [fractions[key][()] for key in fractions if key != "AXIS"]
            ),
        }


def describe_species(args) -> None:
    """
    Warn if the assumed ion labels disagree with the deck's mass-to-charge ratio.

    The forward model takes each ion's charge and mass from its label, so an
    assumed species with the wrong A/Z misplaces the ion-acoustic feature.
    """
    if not args.ion_rqm:
        return
    print("\nion species check:")
    print(
        "  deck rqm x rqm_factor / 1836 is the A/Z the simulation implies; the "
        "label fixes\n  the A/Z the forward model uses. The IAW width scales as "
        "sqrt(A/Z)."
    )
    for name, label, rqm in zip(args.ions, args.ion_labels, args.ion_rqm, strict=True):
        implied = rqm * args.velocity_scale_factor / PROTON_RQM
        particle = Particle(label)
        assumed = float(
            (particle.mass / const.m_p).decompose().value / particle.charge_number
        )
        # Inverting the relation says what rqm_factor the label would need,
        # which is the more actionable number when the two disagree slightly.
        needed = assumed * PROTON_RQM / rqm
        flag = "  <-- MISMATCH" if abs(implied / assumed - 1) > 0.1 else ""
        print(
            f"  {name:<6} rqm {rqm:<5g} x {args.velocity_scale_factor:<5g} "
            f"-> A/Z = {implied:5.2f}   |   label {label!r} has A/Z = {assumed:5.2f}, "
            f"which needs rqm_factor = {needed:.1f}{flag}"
        )


def report(name, spectrogram, reference) -> dict:
    """Print and return metrics for one configuration."""
    epw_nm = EPW_WAVELENGTHS.to_value(u.nm)
    iaw_nm = IAW_WAVELENGTHS.to_value(u.nm)
    result = {
        "epw": normalised(spectrogram.epw, epw_nm),
        "iaw": normalised(spectrogram.iaw, iaw_nm),
    }

    print(f"\n--- {name} ---")
    finite = int(np.isfinite(spectrogram.epw).all(axis=1).sum())
    print(f"  timesteps with a spectrum:  {finite} of {spectrogram.n_time}")
    print(
        f"  alpha:                      median {np.nanmedian(spectrogram.alpha_epw):.3f}"
        f"   range {np.nanmin(spectrogram.alpha_epw):.2f}"
        f" to {np.nanmax(spectrogram.alpha_epw):.2f}"
    )

    if reference is None:
        return result

    ref_epw = normalised(reference["epw"], epw_nm)
    ref_iaw = normalised(reference["iaw"], iaw_nm)
    result["epw_l1"] = l1_per_row(result["epw"], ref_epw, epw_nm)
    result["iaw_l1"] = l1_per_row(result["iaw"], ref_iaw, iaw_nm)
    result["ref_epw"], result["ref_iaw"] = ref_epw, ref_iaw

    red = epw_nm > 545
    shift = peak_wavelength(result["epw"], epw_nm, red) - peak_wavelength(
        ref_epw, epw_nm, red
    )
    density_ratio = spectrogram.electron_density * 1e-6 / reference["density"]

    print(f"  density ratio (new/old):    median {np.nanmedian(density_ratio):.4f}")
    print(
        f"  alpha ratio (new/old):      median "
        f"{np.nanmedian(spectrogram.alpha_epw / reference['alpha']):.4f}"
    )
    print(
        f"  EPW L1 vs reference:        median {np.nanmedian(result['epw_l1']):.4f}"
        f"   90th pct {np.nanpercentile(result['epw_l1'], 90):.4f}"
    )
    print(
        f"  IAW L1 vs reference:        median {np.nanmedian(result['iaw_l1']):.4f}"
        f"   90th pct {np.nanpercentile(result['iaw_l1'], 90):.4f}"
    )
    print(f"  EPW red-peak shift (nm):    median {np.nanmedian(shift):+.2f}")
    return result


def check_epw_tracks_density(spectrogram, notch) -> None:
    r"""
    Check the EPW satellite against the plasma-frequency shift.

    The satellite sits at the plasma frequency plus a Bohm-Gross excess, so the
    observed-to-predicted ratio should be :math:`\sqrt{1 + 3 k^2 \lambda_{De}^2}`
    -- a little above one. Agreement validates the whole chain: reader units,
    conditioning, and forward model.

    The notch has to be kept out of it. Where the satellite would fall inside
    the notch, the largest surviving value sits on the notch edge and the ratio
    is meaningless, so those timesteps are excluded rather than reported.
    """
    epw_nm = EPW_WAVELENGTHS.to_value(u.nm)
    probe_nm = PROBE_WAVELENGTH.to_value(u.nm)
    notch_nm = u.Quantity(notch, u.nm).to_value(u.nm)
    clearance = 2.0  # nm the satellite must clear the notch edge by
    red = epw_nm > notch_nm[1] + clearance
    peaks = peak_wavelength(spectrogram.epw, epw_nm, red)

    omega_pe = np.sqrt(
        spectrogram.electron_density
        * const.e.si.value**2
        / (const.eps0.si.value * const.m_e.si.value)
    )
    predicted = (
        PROBE_WAVELENGTH.to_value(u.m) ** 2
        * omega_pe
        / (2 * np.pi * const.c.si.value)
        * 1e9
    )
    observed = peaks - probe_nm
    ratio = observed / predicted

    # Only meaningful where the satellite clears the notch and still fits inside
    # the window.
    usable = (
        np.isfinite(ratio)
        & (predicted > notch_nm[1] - probe_nm + clearance)
        & (peaks < epw_nm[-1] - 5)
    )
    # Bohm-Gross expectation from the reported alpha, converting the forward
    # model's sqrt(2)*wpe/(k*sigma) convention into 1/(k*lambda_De).
    k_lambda_de = np.sqrt(2) / spectrogram.alpha_epw
    expected = np.sqrt(1 + 3 * k_lambda_de**2)

    print("\nEPW satellite vs the plasma-frequency shift:")
    print(
        f"  usable timesteps:           {int(usable.sum())} of {spectrogram.n_time}"
        f"   (satellite must clear the notch)"
    )
    if usable.any():
        print(f"  observed / predicted:       median {np.nanmedian(ratio[usable]):.3f}")
        print(
            f"  Bohm-Gross expectation:     median "
            f"{np.nanmedian(expected[usable]):.3f}"
            f"   (= sqrt(1 + 3 k^2 lambda_De^2) from alpha)"
        )


def figure(spectrograms, results, reference, args) -> None:
    """Draw the two configurations, and the reference if there is one."""
    epw_nm = EPW_WAVELENGTHS.to_value(u.nm)
    iaw_nm = IAW_WAVELENGTHS.to_value(u.nm)
    legacy, corrected = results["legacy"], results["corrected"]
    time_ns = spectrograms["legacy"].t * 1e9

    columns = 3 if reference is not None else 2
    fig, axes = plt.subplots(3, columns, figsize=(5.4 * columns, 12), squeeze=False)

    for row, (name, axis, key) in enumerate(
        (("EPW", epw_nm, "epw"), ("IAW", iaw_nm, "iaw"))
    ):
        panels = [
            (legacy[key], f"{name}: legacy-matched (unbounded taper)"),
            (corrected[key], f"{name}: corrected (bounded taper)"),
        ]
        if reference is not None:
            panels.insert(0, (legacy[f"ref_{key}"], f"{name}: osiris2thomson"))
        for column, (data, title) in enumerate(panels):
            finite = data[np.isfinite(data).all(axis=1)]
            image = axes[row][column].imshow(
                data.T,
                origin="lower",
                aspect="auto",
                extent=[time_ns[0], time_ns[-1], axis[0], axis[-1]],
                cmap="inferno",
                vmax=np.percentile(finite, 99) if finite.size else None,
            )
            fig.colorbar(image, ax=axes[row][column])
            axes[row][column].set_title(title, fontsize=9)
            axes[row][column].set_xlabel("time (ns)")
            axes[row][column].set_ylabel("wavelength (nm)")

    axes[2][0].semilogy(time_ns, spectrograms["legacy"].electron_density * 1e-6, lw=1.2)
    axes[2][0].set_ylabel(r"$n_e$ (cm$^{-3}$)")
    axes[2][0].set_title("electron density at the sampled point", fontsize=9)

    axes[2][1].plot(
        time_ns, spectrograms["legacy"].alpha_epw, lw=1.2, label="legacy-matched"
    )
    axes[2][1].plot(
        time_ns, spectrograms["corrected"].alpha_epw, lw=1.2, ls="--", label="corrected"
    )
    axes[2][1].set_ylabel(r"$\alpha$")
    axes[2][1].set_title("scattering parameter", fontsize=9)
    axes[2][1].legend(fontsize=8)

    if reference is not None:
        axes[2][2].semilogy(time_ns, legacy["epw_l1"], lw=1.1, label="EPW, legacy")
        axes[2][2].semilogy(time_ns, legacy["iaw_l1"], lw=1.1, label="IAW, legacy")
        axes[2][2].semilogy(
            time_ns, corrected["epw_l1"], lw=1.0, ls="--", label="EPW, corrected"
        )
        axes[2][2].semilogy(
            time_ns, corrected["iaw_l1"], lw=1.0, ls="--", label="IAW, corrected"
        )
        axes[2][2].set_ylabel("L1 vs osiris2thomson")
        axes[2][2].set_title("per-timestep spectrum difference", fontsize=9)
        axes[2][2].legend(fontsize=8)

    for ax in axes[2]:
        ax.set_xlabel("time (ns)")

    fig.suptitle(
        f"pic_thomson end-to-end: {args.ms.parent.name}, x = {args.position:.2f} mm",
        y=1.0,
    )
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / "07_osiris_end_to_end.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {path.relative_to(MEDIA.parent)}")


def main() -> None:
    """Run both configurations over the simulation and report the outcome."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms", type=Path, required=True, help="OSIRIS MS directory")
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="optional osiris2thomson spectra.hdf5 to compare against",
    )
    parser.add_argument(
        "--reference-density",
        type=float,
        default=9e17,
        help="simulation reference density in cm^-3",
    )
    parser.add_argument(
        "--velocity-scale-factor",
        type=float,
        default=50.0,
        help="mass-ratio reduction factor R; velocities are divided by sqrt(R)",
    )
    parser.add_argument(
        "--position", type=float, default=5.0, help="sampling position in mm"
    )
    parser.add_argument("--electron", default="e", help="electron species name")
    parser.add_argument("--ions", nargs="+", default=["cham", "targ"])
    parser.add_argument(
        "--ion-labels",
        nargs="+",
        default=["p+", "p+"],
        help="PlasmaPy species for each ion population",
    )
    parser.add_argument(
        "--ion-rqm",
        nargs="+",
        type=float,
        default=None,
        help="deck rqm of each ion, used only to check the labels",
    )
    parser.add_argument("--smoothing-iterations", type=int, default=4)
    parser.add_argument("--stride", type=int, default=1, help="read every Nth dump")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the corrected spectrogram to this HDF5 file",
    )
    parser.add_argument(
        "--instrument-time-fwhm",
        type=float,
        default=None,
        help="streak-camera temporal resolution in ps; renders a second figure",
    )
    parser.add_argument(
        "--instrument-wavelength-fwhm",
        type=float,
        default=None,
        help="spectrometer resolution in nm",
    )
    parser.add_argument(
        "--notch",
        nargs=2,
        type=float,
        default=[530.0, 534.0],
        help="stray-light notch in nm. Must be narrower than the EPW satellite "
        "separation, which is only ~8 nm at 1e18 cm^-3",
    )
    args = parser.parse_args()

    reference_density = args.reference_density * u.cm**-3
    describe_species(args)

    print(f"\nreading {args.ms} ...")
    electrons, ions = read_run(args.ms, args, reference_density)
    print(
        f"  {electrons.shape[0]} timesteps, "
        f"{electrons.t[-1] * 1e9:.2f} ns, "
        f"domain {electrons.x[-1] * 1e3:.2f} mm"
    )

    reference = load_reference(args.reference) if args.reference else None
    if reference is not None and reference["epw"].shape[0] != electrons.shape[0]:
        raise SystemExit(
            f"reference has {reference['epw'].shape[0]} timesteps but the run was "
            f"read with {electrons.shape[0]}; adjust --stride."
        )

    spectrograms, results = {}, {}
    for name, bounded in (("legacy", False), ("corrected", True)):
        print(f"\nrunning {name} configuration ...")
        spectrograms[name] = run_pipeline(
            electrons,
            ions,
            args,
            reference_density,
            args.position * 1e-3,
            bounded=bounded,
        )
        results[name] = report(name, spectrograms[name], reference)

    check_epw_tracks_density(spectrograms["corrected"], args.notch * u.nm)

    if args.output is not None:
        spectrograms["corrected"].to_hdf5(args.output)
        print(f"\n  wrote {args.output}")

    if args.instrument_time_fwhm or args.instrument_wavelength_fwhm:
        degraded = spectrograms["corrected"].apply_instrument_response(
            time_fwhm=(args.instrument_time_fwhm or 0.0) * u.ps,
            epw_wavelength_fwhm=(args.instrument_wavelength_fwhm or 0.0) * u.nm,
            iaw_wavelength_fwhm=(args.instrument_wavelength_fwhm or 0.0) * u.nm,
        )
        MEDIA.mkdir(exist_ok=True)
        # Not named `figure`: that is the module-level plotting function.
        rendered, _ = degraded.plot(save=MEDIA / "10_osiris_instrument_response.png")
        plt.close(rendered)
        print("  wrote media/10_osiris_instrument_response.png")

    epw_nm = EPW_WAVELENGTHS.to_value(u.nm)
    iaw_nm = IAW_WAVELENGTHS.to_value(u.nm)
    print("\nlegacy vs corrected (what bounding the taper changes):")
    print(
        f"  EPW L1 difference:          median "
        f"{np.nanmedian(l1_per_row(results['legacy']['epw'], results['corrected']['epw'], epw_nm)):.4f}"
    )
    print(
        f"  IAW L1 difference:          median "
        f"{np.nanmedian(l1_per_row(results['legacy']['iaw'], results['corrected']['iaw'], iaw_nm)):.4f}"
    )
    ratio = spectrograms["corrected"].alpha_epw / spectrograms["legacy"].alpha_epw
    print(f"  alpha ratio (corrected/legacy): median {np.nanmedian(ratio):.3f}")

    figure(spectrograms, results, reference, args)


if __name__ == "__main__":
    main()
