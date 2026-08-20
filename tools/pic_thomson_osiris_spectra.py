r"""
Thomson spectra from an OSIRIS run, with the reduced-mass-ratio corrections.

The question this answers is what the forward model gives for a real run once
the velocity corrections are applied -- nothing else. There is no legacy
configuration here: the half-cosine taper is settled (see ``plan.md`` 8 and the
notes on ``taper_vdf_edges``), and comparing against it only crowds the figure.

Two treatments are drawn, one per row:

``all velocities / sqrt(R)``
    What a similarity-scaled run wants. Every species' velocity axis is divided
    by :math:`\sqrt{R}`, electrons included, because the scaling is a property
    of the model rather than of a species' mass.
``ions / sqrt(R) + electron drift / sqrt(R)``
    What a run with kinetic electrons at the real :math:`m_e` and reduced-mass
    ions wants: the ion axis rescaled, and the electrons *translated* so their
    bulk velocity is divided while their temperature is left alone. Drawn here
    so the difference between the two is visible on the same run.

**Both spectrograms use one absolute colour scale across all timesteps.** Each
row is *not* rescaled to its own maximum: doing that makes a frame carrying no
signal look identical to one carrying a strong feature, and where a notch has
removed the central feature it promotes the residue at the notch edge into a
saturated rail. The scale here is shared, so brightness means intensity.

Usage::

    python tools/pic_thomson_osiris_spectra.py \
        --ms ~/OmegaShock/runs/omegashock_w3.5e11_exp/MS
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path

import astropy.units as u
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt

MEDIA = Path(__file__).resolve().parent.parent / "media"


def probe_volume(phase_space, position, n_cells):
    """Average the phase space over the cells a real collection volume spans."""
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


def read(args):
    """Read the electron and ion phase spaces, reduced to the probe volume."""
    dumps = None
    if args.stride > 1:
        available = sorted((args.ms / "PHA" / "p1x1" / args.electron).glob("*.h5"))
        dumps = list(range(0, len(available), args.stride))
    density = args.reference_density * u.cm**-3
    common = {"reference_density": density, "timesteps": dumps}
    electrons = pt.read_osiris_phase_space(
        args.ms, "p1x1", args.electron, is_electron=True, **common
    )
    ions = [
        pt.read_osiris_phase_space(args.ms, "p1x1", name, label=label, **common)
        for name, label in zip(args.ions, args.ion_labels, strict=True)
    ]
    position = args.position * 1e-3
    electrons = probe_volume(electrons, position, args.probe_cells)
    ions = [probe_volume(ion, position, args.probe_cells) for ion in ions]
    return electrons, ions, density, position


def spectra(electrons, ions, density, position, args, **treatment):
    """Run the forward model over every frame for one treatment."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pt.spectra_from_phase_spaces(
            electrons,
            ions,
            position=position,
            reference_density=density,
            probe_wavelength=args.probe_wavelength * u.nm,
            epw_wavelengths=np.linspace(*args.epw, args.spectral_bins) * u.nm,
            iaw_wavelengths=np.linspace(*args.iaw, args.spectral_bins) * u.nm,
            epw_notches=np.array(args.notch) * u.nm,
            scatter_vec=[
                np.cos(np.deg2rad(args.theta)),
                np.sin(np.deg2rad(args.theta)),
                0.0,
            ],
            electron_conditioning={
                "smoothing_width": args.smoothing_width,
                "smoothing_iterations": args.smoothing_iterations,
                "max_taper_bins": None,
                "pedestal_warning": None,
                "smoothing_variance_warning": None,
            },
            ion_conditioning={
                "smoothing_iterations": 0,
                "max_taper_bins": None,
                "pedestal_warning": None,
            },
            mask_unresolved_epw=False,
            n_quadrature_points=args.quadrature_points,
            progress=False,
            **treatment,
        )


def figure(runs, args) -> None:
    """Two treatments by two features, on one absolute colour scale per column."""
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4), squeeze=False)
    columns = (("epw", "epw_wavelengths", "EPW"), ("iaw", "iaw_wavelengths", "IAW"))

    # One scale per feature, over every treatment and every timestep, so that
    # brightness is comparable everywhere in the figure.
    limits = {}
    for key, _, _ in columns:
        stacked = np.concatenate(
            [
                np.asarray(getattr(run, key), dtype=float).ravel()
                for run in runs.values()
            ]
        )
        stacked = stacked[np.isfinite(stacked) & (stacked > 0)]
        limits[key] = np.percentile(stacked, args.percentile) if stacked.size else None

    for row, (label, run) in enumerate(runs.items()):
        t = run.t * 1e9
        for column, (key, axis_name, name) in enumerate(columns):
            grid = getattr(run, axis_name) * 1e9
            data = np.ma.masked_invalid(np.asarray(getattr(run, key), dtype=float).T)
            ax = axes[row][column]
            image = ax.imshow(
                data,
                origin="lower",
                aspect="auto",
                cmap="inferno",
                extent=[t[0], t[-1], grid[0], grid[-1]],
                vmin=0.0,
                vmax=limits[key],
            )
            fig.colorbar(image, ax=ax, extend="max")
            ax.set(
                xlabel="time (ns)",
                ylabel="wavelength (nm)",
                title=f"{name} — {label}",
            )

    fig.suptitle(
        f"pic_thomson on {args.ms.parent.name}, x = {args.position:.2f} mm  —  "
        rf"$R = {args.velocity_scale_factor:.0f}$, "
        f"{args.probe_cells}-cell probe volume, "
        f"fixed {args.notch[0]:.0f}–{args.notch[1]:.0f} nm notch"
        "\none absolute colour scale per feature: no per-timestep renormalisation",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / "14_osiris_spectra.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    """Read the run, apply each treatment, and draw the spectra."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--ms", type=Path, required=True, help="OSIRIS MS directory")
    parser.add_argument("--reference-density", type=float, default=9e17, help="cm^-3")
    parser.add_argument("--velocity-scale-factor", type=float, default=50.0)
    parser.add_argument("--position", type=float, default=5.0, help="probe x [mm]")
    parser.add_argument("--probe-wavelength", type=float, default=532.0)
    parser.add_argument("--theta", type=float, default=63.0)
    parser.add_argument("--epw", nargs=2, type=float, default=[432.0, 632.0])
    parser.add_argument("--iaw", nargs=2, type=float, default=[522.0, 542.0])
    parser.add_argument("--notch", nargs=2, type=float, default=[525.0, 539.0])
    parser.add_argument("--spectral-bins", type=int, default=500)
    parser.add_argument("--probe-cells", type=int, default=5)
    parser.add_argument("--quadrature-points", type=float, default=1e4)
    parser.add_argument("--smoothing-width", type=float, default=0.25)
    parser.add_argument("--smoothing-iterations", type=int, default=4)
    parser.add_argument("--electron", default="e")
    parser.add_argument("--ions", nargs="+", default=["cham", "targ"])
    parser.add_argument("--ion-labels", nargs="+", default=["p+", "p+"])
    parser.add_argument("--stride", type=int, default=10, help="read every Nth dump")
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.5,
        help="percentile of the whole spectrogram set used for the colour limit",
    )
    args = parser.parse_args()

    print(f"reading {args.ms} ...", flush=True)
    electrons, ions, density, position = read(args)
    print(f"  {electrons.shape[0]} frames, {electrons.t[-1] * 1e9:.2f} ns", flush=True)

    ratio = args.velocity_scale_factor
    treatments = {
        r"all velocities /$\sqrt{R}$": {"velocity_scale_factor": ratio},
        r"ions /$\sqrt{R}$ + electron drift /$\sqrt{R}$": {
            "ion_velocity_scale_factor": ratio,
            "electron_drift_scale_factor": ratio,
        },
    }
    runs = {}
    for label, treatment in treatments.items():
        start = time.time()
        runs[label] = spectra(electrons, ions, density, position, args, **treatment)
        alpha = runs[label].alpha_epw / np.sqrt(2)
        print(
            f"  {label}: alpha median {np.nanmedian(alpha):6.2f} "
            f"(range {np.nanmin(alpha):.2f}-{np.nanmax(alpha):.2f})"
            f"   ({time.time() - start:.0f}s)",
            flush=True,
        )
    figure(runs, args)


if __name__ == "__main__":
    main()
