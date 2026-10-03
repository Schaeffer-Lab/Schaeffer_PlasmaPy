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
from dataclasses import replace
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


def probe_volume(phase_space, position, n_cells):
    """
    Average the phase space over the cells a real collection volume spans.

    A Thomson collection volume at 532 nm is tens to a hundred microns; a cell
    in this run is 13.1. Averaging over the cells the volume covers is what the
    measurement does anyway, and it buys sqrt(N) in macroparticle statistics
    where the forward model is most sensitive to them. Averaged, never summed:
    summing multiplies the density by the cell count.
    """
    if n_cells <= 1:
        return phase_space
    index = int(np.argmin(np.abs(phase_space.x - position)))
    half = n_cells // 2
    lo, hi = max(index - half, 0), min(index + half + 1, phase_space.x.size)
    return replace(
        phase_space,
        f=np.ascontiguousarray(phase_space.f[:, :, lo:hi].mean(axis=2, keepdims=True)),
        x=phase_space.x[index : index + 1],
        meta={**phase_space.meta, "probe_cells": hi - lo},
    )


def run_pipeline(electrons, ions, args, reference_density, position, *, bounded):
    """Run the driver in the legacy or the corrected configuration."""
    # The legacy pipeline smoothed with a 40-bin boxcar. On this run the
    # electrons occupy 31 of the 1024 bins the momentum diagnostic spans, so
    # that window is wider than the distribution and adds several hundred eV of
    # variance to it. The corrected configuration asks for a width in thermal
    # speeds instead and lets the module size the window.
    electron_smoothing = (
        {"smoothing_width": 0.25} if bounded else {"smoothing_window": 40}
    )
    electrons = probe_volume(electrons, position, args.probe_cells)
    ions = [probe_volume(ion, position, args.probe_cells) for ion in ions]
    return pt.spectra_from_phase_spaces(
        electrons,
        ions,
        position=position,
        reference_density=reference_density,
        probe_wavelength=PROBE_WAVELENGTH,
        epw_wavelengths=EPW_WAVELENGTHS,
        iaw_wavelengths=IAW_WAVELENGTHS,
        # Legacy: the fixed notch the old pipeline used. Corrected: sized from
        # the central feature every frame, because the flow shifts it out of a
        # fixed one and the bin just outside then takes most of the window.
        epw_notches=("auto" if (bounded and args.auto_notch) else args.notch * u.nm),
        scatter_vec=SCATTER_VEC,
        electron_conditioning={
            **electron_smoothing,
            "smoothing_iterations": args.smoothing_iterations,
            # Legacy: the half-cosine taper, run to the grid edge. Corrected:
            # the fitted-Maxwellian tail, which is the default.
            "tail_model": None if not bounded else "maxwellian",
            "max_taper_bins": None,
            "pedestal_warning": None,
            "smoothing_variance_warning": None if bounded else 1e9,
        },
        ion_conditioning={
            "smoothing_iterations": 0,
            "tail_model": None if not bounded else "maxwellian",
            "max_taper_bins": None,
            "pedestal_warning": None,
        },
        # The ions only, plus a translation of the electron bulk velocity.
        # The deck settles this: `species { name = "e", rqm = -1.0 }` is a real
        # electron, so T_e, v_te, lambda_De and therefore alpha are already
        # physical and the electron velocity axis must not be rescaled. Only
        # the flow the electrons share with the ions is sqrt(R) too fast, and a
        # translation is what moves a mean while leaving every central moment
        # alone. See plan.md 16 and 17.8.
        #
        # This used to pass velocity_scale_factor, i.e. the whole axis for every
        # species. That is treatment C of 16.5, and on this run it multiplied
        # alpha by sqrt(50) = 7.07 -- from 1.64 to 11.58 -- carrying it past the
        # point where the satellite is computable in double precision at all.
        # It is why the EPW panel read as empty.
        ion_velocity_scale_factor=args.velocity_scale_factor,
        electron_drift_scale_factor=args.velocity_scale_factor,
        # Keep every row in both: with the tail fitted, a resonance past the
        # sampled data is an extrapolation priced by epw_tail_uncertainty rather
        # than something to hide.
        mask_unresolved_epw=False,
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
    Check the EPW satellites against Bohm-Gross, on the frames that have them.

    The resonance sits at :math:`\omega^2 = \omega_{pe}^2 (1 + 3/\alpha^2)`,
    with :math:`\alpha = 1/k\lambda_{De}`, which on each side of the probe
    maps to :math:`\lambda_\pm = 2\pi c / (\omega_0 \mp \omega)`. Agreement
    validates the whole chain: reader units, conditioning, and forward model.

    Two exclusions, each of which an earlier version of this check got wrong.
    Only frames with :math:`\alpha > 1.2` count: below that there is no
    resonance, and the peak of the electron feature is not a satellite. And the
    notch test uses the Bohm-Gross offset, not the bare plasma-frequency shift,
    which is smaller -- testing that against the notch threw away every
    collective frame of ``omegashock_w3.5e11_exp`` and left only the
    non-collective ones, where the comparison means nothing.
    """
    epw_nm = EPW_WAVELENGTHS.to_value(u.nm)
    notch_nm = u.Quantity(notch, u.nm).to_value(u.nm)
    c = const.c.si.value
    omega_0 = 2 * np.pi * c / PROBE_WAVELENGTH.to_value(u.m)
    omega_pe = np.sqrt(
        spectrogram.electron_density
        * const.e.si.value**2
        / (const.eps0.si.value * const.m_e.si.value)
    )
    # The forward model reports sqrt(2) * wpe / (k sigma); 1/(k lambda_De) is
    # that over sqrt(2).
    alpha = spectrogram.alpha_epw / np.sqrt(2)
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = omega_pe * np.sqrt(1 + 3 / alpha**2)
    expected = {
        "blue": 2 * np.pi * c / (omega_0 + omega) * 1e9,
        "red": 2 * np.pi * c / (omega_0 - omega) * 1e9,
    }
    outside = {"blue": epw_nm < notch_nm[0], "red": epw_nm > notch_nm[1]}
    search = 6.0  # nm either side of Bohm-Gross to look for the peak

    offsets = {"blue": [], "red": []}
    for step in np.nonzero(np.isfinite(alpha) & (alpha > 1.2))[0]:
        row = np.nan_to_num(np.asarray(spectrogram.epw[step], dtype=float))
        found = {}
        for side in ("blue", "red"):
            near = outside[side] & (np.abs(epw_nm - expected[side][step]) < search)
            if near.sum() < 3 or row[near].max() <= 0:
                break
            found[side] = epw_nm[near][np.argmax(row[near])] - expected[side][step]
        if len(found) == 2:
            for side, offset in found.items():
                offsets[side].append(offset)

    print("\nEPW satellites vs Bohm-Gross (frames with alpha > 1.2):")
    print(
        f"  frames compared:            {len(offsets['red'])} of {spectrogram.n_time}"
    )
    for side in ("blue", "red"):
        if offsets[side]:
            values = np.asarray(offsets[side])
            print(
                f"  {side:4s} observed - expected:  median {np.median(values):+.2f} nm"
                f"   (largest {values[np.argmax(np.abs(values))]:+.2f} nm)"
            )


def figure(spectrograms, results, reference, args) -> None:  # noqa: PLR0915
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
        # The pipeline's own spectra, n_e S(k, w) on one scale, not the
        # per-row normalised copies the L1 comparison uses: normalising each
        # row hid the brightness history and, where the notch had removed the
        # central feature, promoted its residue into a rail along each edge.
        panels = [
            (
                getattr(spectrograms["legacy"], key),
                f"{name}: legacy (half-cosine taper to zero)",
            ),
            (
                getattr(spectrograms["corrected"], key),
                f"{name}: corrected (fitted Maxwellian tail)",
            ),
        ]
        if reference is not None:
            panels.insert(0, (legacy[f"ref_{key}"], f"{name}: osiris2thomson"))
        for column, (data, title) in enumerate(panels):
            away = np.abs(axis - PROBE_WAVELENGTH.to_value(u.nm)) > args.colour_guard
            shown_data = np.asarray(data, dtype=float)
            finite = shown_data[np.isfinite(shown_data).any(axis=1)]
            scale_from = finite[:, away] if away.sum() > 4 else finite
            if args.log_scale:
                # The features that matter span
                # many decades: the notch skirt sits orders of magnitude above
                # the satellites, so on a linear scale everything else is one
                # colour. Zeros -- the notch itself -- are masked rather than
                # clipped, so the mask reads as blank instead of as signal.
                shown = np.ma.masked_invalid(
                    np.ma.masked_where(~(shown_data > 0), shown_data)
                )
                positive = scale_from[np.isfinite(scale_from) & (scale_from > 0)]
                top = np.percentile(positive, 99.9) if positive.size else 1.0
                norm = mpl.colors.LogNorm(vmin=top * 10.0**-args.log_decades, vmax=top)
                kwargs = {"norm": norm}
            else:
                shown = np.ma.masked_invalid(shown_data)
                kwargs = {
                    "vmin": 0.0,
                    "vmax": (
                        np.nanpercentile(scale_from, 99)
                        if np.isfinite(scale_from).any()
                        else None
                    ),
                }
            image = axes[row][column].imshow(
                shown.T,
                origin="lower",
                aspect="auto",
                extent=[time_ns[0], time_ns[-1], axis[0], axis[-1]],
                cmap="inferno",
                **kwargs,
            )
            fig.colorbar(image, ax=axes[row][column], extend="max")
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


def main() -> None:  # noqa: PLR0915
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
        help="mass-ratio reduction factor R. The ion velocity axes are divided "
        "by sqrt(R), and the electron bulk velocity is divided by sqrt(R) as a "
        "translation that leaves T_e alone. The electron axis is NOT rescaled: "
        "the deck runs real-mass electrons, so their thermal scale is already "
        "physical",
    )
    parser.add_argument(
        "--position", type=float, default=5.0, help="sampling position in mm"
    )
    parser.add_argument(
        "--auto-notch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="size the corrected run's stray-light mask from the central "
        "feature each frame, instead of using --notch. Off by default: a "
        "stray-light notch is a piece of glass and does not change width "
        "between frames. The automatic sizing measures a 99.9%% containment "
        "interval, which is set by whatever is in the wings rather than by "
        "the line, and is then capped at a fraction of the Bohm-Gross offset, "
        "which makes it a function of density. On omegashock_w3.5e11_exp it "
        "ranges over 0.52 to 9.92 nm, a factor of 19, with a median of 0.64. "
        "See plan.md 17.5",
    )
    parser.add_argument(
        "--log-scale",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="log colour scale on the spectrograms. Off by default: a linear "
        "scale is what a measured spectrum is read on. Worth turning on to "
        "check whether a feature is absent or merely faint",
    )
    parser.add_argument(
        "--colour-guard",
        type=float,
        default=8.0,
        help="nm either side of the probe line excluded when choosing the "
        "colour limit, so the central feature does not crush the satellites. "
        "Keep it just outside --notch",
    )
    parser.add_argument(
        "--log-decades",
        type=float,
        default=8.0,
        help="decades below the 99.9th percentile to show with --log-scale",
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
    parser.add_argument(
        "--probe-cells",
        type=int,
        default=5,
        help="cells the collection volume spans, averaged. Cells here are "
        "19.9 um on omegashock_w3.5e11_exp, so 5 is a 100 um volume -- the "
        "scale of a real 532 nm Thomson collection volume. 1 disables it",
    )
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
        default=[525.0, 539.0],
        help="fixed stray-light notch in nm. Sized from the skirt on "
        "omegashock_w3.5e11_exp: measured out from the probe line, the "
        "central feature falls from 1e9 to ~1e4 at 4 nm, ~1e1 at 6 nm and "
        "~1e-5 by 7 nm, so +-7 leaves nothing above the satellites for the "
        "colour scale to lock onto. A narrower mask leaves the first "
        "surviving bin carrying the skirt, which draws a saturated rail "
        "along each notch edge across the whole spectrogram. The real "
        "satellites in this run are 11-20 nm out (512-520 and 545 nm, in the "
        "alpha = 1.9-4.5 window at 0.35-0.50 ns), so +-7 clears them",
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
    print("\nlegacy vs corrected (what the fitted tail changes):")
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
