r"""
`plasmapy.diagnostics.pic_thomson` on a reduced-ion-mass KinShock run.

The run has **kinetic electrons at the real electron mass** and ions at
``m_i = 100 m_e``, so :math:`R = m_p/m_\mathrm{sim} = 18.36`. That splits the
quantities: :math:`n_e`, :math:`T_e`, :math:`\lambda_{De}`, :math:`v_{te}` and
therefore :math:`\alpha` are already physical, while every ion velocity and the
bulk flow the electrons share with the ions are :math:`\sqrt{R}` too fast. Four
treatments are compared; see ``plan.md`` sections 16 and 17.

Three settings here are not the module defaults, and each of them was arrived
at by a measurement rather than a preference:

``--quadrature-points 1e4``
    The forward model's principal-value integral for :math:`\chi` is sampled on
    a grid anchored at :math:`\xi`. Its default resolution is converged for a
    smooth distribution and is not for a PIC histogram: the grid slides under a
    shot-noise-roughened :math:`f'` as :math:`\xi` sweeps the wavelength axis,
    putting ~0.5 dex of spurious ringing into the wings of both features.

``--probe-cells 11``
    A Thomson collection volume at 532 nm is tens to a hundred microns; a cell
    in this run is 7.6. Averaging over the cells the real volume spans is what
    the measurement does anyway and buys :math:`\sqrt{N}` in the tail, which is
    where the electron feature reads. Average, never sum -- summing multiplies
    the density by the cell count.

``--notch 526 538``
    A fixed notch, because a stray-light filter is a piece of glass. Sized once
    from the central feature's full excursion over the run; the nearest
    Bohm-Gross satellite is at :math:`\pm 20` nm, so it clears by 14 nm.

Usage::

    python tools/pic_thomson_kinshock.py ~/KinShock2020/runs/R1_phase/R1_paper_470eV
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import replace
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics import pic_thomson as pt

MEDIA = Path(__file__).resolve().parent.parent / "media"

QE = const.e.si.value
ME = const.m_e.si.value
EPS0 = const.eps0.si.value
C = const.c.si.value

SPECIES = (
    ("amb_electrons", "e-", True),
    ("piston_electrons", "e-", True),
    ("amb_ions", "p+", False),
    ("piston_ions", "p+", False),
)


def read(run: Path, args) -> dict[str, pt.PICPhaseSpace]:
    """Read every species, caching the binned phase space beside the run."""
    cache = run / "thomson_cache"
    cache.mkdir(exist_ok=True)
    mass_sim = args.mass_ratio * ME
    out = {}
    for name, label, is_electron in SPECIES:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[name] = pt.read_warpx_phase_space(
                run / "diags",
                species=name,
                mass=ME if is_electron else mass_sim,
                label=label,
                is_electron=is_electron,
                scatter_direction=(0.0, 0.0, 1.0),
                n_position_bins=args.position_bins,
                n_velocity_bins=args.velocity_bins,
                cache=cache / f"{name}.npz",
                progress=False,
            )
        print(f"  {name:18} {out[name].shape}  {time.time() - t0:.0f}s", flush=True)
    return out


def probe_volume(
    phase_space: pt.PICPhaseSpace, position: float, n_cells: int
) -> pt.PICPhaseSpace:
    """
    Average the phase space over the cells a real collection volume spans.

    Averaged, not summed: the forward model reads a density off this, and
    summing ``n_cells`` of them multiplies it by ``n_cells``, which drags
    ``alpha`` up by its square root.
    """
    index = int(np.argmin(np.abs(phase_space.x - position)))
    half = n_cells // 2
    lo, hi = max(index - half, 0), min(index + half + 1, phase_space.x.size)
    return replace(
        phase_space,
        f=np.ascontiguousarray(phase_space.f[:, :, lo:hi].mean(axis=2, keepdims=True)),
        x=phase_space.x[index : index + 1],
        meta={**phase_space.meta, "probe_cells": hi - lo},
    )


def spectra(species: dict, args, **treatment: float) -> pt.ThomsonSpectrogram:
    """One treatment, over every frame."""
    probe = args.position * 1e-3
    electrons = [
        probe_volume(species[n], probe, args.probe_cells)
        for n, _, is_e in SPECIES
        if is_e
    ]
    ions = [
        probe_volume(species[n], probe, args.probe_cells)
        for n, _, is_e in SPECIES
        if not is_e
    ]
    angle = np.deg2rad(args.theta) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pt.spectra_from_phase_spaces(
            electrons,
            ions,
            position=probe,
            probe_wavelength=args.probe_wavelength * u.nm,
            epw_wavelengths=np.linspace(*args.epw, args.spectral_bins) * u.nm,
            iaw_wavelengths=np.linspace(*args.iaw, args.spectral_bins) * u.nm,
            epw_notches=np.array(args.notch) * u.nm,
            probe_vec=np.array([np.cos(angle), 0.0, -np.sin(angle)]),
            scatter_vec=np.array([np.cos(angle), 0.0, np.sin(angle)]),
            electron_conditioning={
                "smoothing_width": args.smoothing_width,
                "smoothing_iterations": args.smoothing_iterations,
            },
            ion_conditioning={
                "smoothing_width": args.smoothing_width,
                "smoothing_iterations": args.smoothing_iterations,
            },
            n_quadrature_points=args.quadrature_points,
            progress=False,
            **treatment,
        )


def feature_width(spectrogram) -> np.ndarray:
    """FWHM of the ion feature, from the second moment of the IAW window."""
    grid = spectrogram.iaw_wavelengths * 1e9
    out = np.full(spectrogram.n_time, np.nan)
    for step in range(spectrogram.n_time):
        row = np.clip(spectrogram.iaw[step], 0, None)
        total = np.trapezoid(row, grid)
        if total <= 0 or not np.isfinite(total):
            continue
        centre = np.trapezoid(row * grid, grid) / total
        out[step] = 2.355 * np.sqrt(
            np.trapezoid(row * (grid - centre) ** 2, grid) / total
        )
    return out


def bohm_gross(spectrogram, probe_nm: float) -> tuple[np.ndarray, np.ndarray]:
    """Blue and red satellite wavelengths implied by the reported alpha."""
    wpe = np.sqrt(spectrogram.electron_density * QE**2 / (EPS0 * ME))
    with np.errstate(invalid="ignore", divide="ignore"):
        omega = wpe * np.sqrt(1 + 6 / np.maximum(spectrogram.alpha_epw, 1e-9) ** 2)
    probe = probe_nm * 1e-9
    shift = omega / (2 * np.pi * C)
    return 1 / (1 / probe + shift) * 1e9, 1 / (1 / probe - shift) * 1e9


def satellite_midpoint(spectrogram, probe_nm: float, guard: float = 13.0) -> np.ndarray:
    """
    Doppler shift of the electron feature, as the midpoint of its satellites.

    A bulk drift moves both satellites the same way while the density moves
    them apart, so their midpoint isolates the drift. A centroid of the whole
    window does not: it follows whichever satellite is in view, and once the
    satellites leave the window it follows the noise.
    """
    grid = spectrogram.epw_wavelengths * 1e9
    blue, red = bohm_gross(spectrogram, probe_nm)
    out = np.full(spectrogram.n_time, np.nan)
    for step in range(spectrogram.n_time):
        row = np.clip(spectrogram.epw[step], 0, None)
        if not spectrogram.epw_resolved[step] or row.max() <= 0:
            continue
        if not np.isfinite(row).all():
            continue
        peaks = []
        for centre in (blue[step], red[step]):
            near = (
                (grid > centre - 25)
                & (grid < centre + 25)
                & (np.abs(grid - probe_nm) > guard)
            )
            if near.sum() > 5 and row[near].max() > 0:
                peaks.append(grid[near][np.argmax(row[near])])
        if len(peaks) == 2:
            out[step] = sum(peaks) / 2
    return out


def figure(runs: dict, args) -> None:  # noqa: PLR0915
    """The six-panel summary."""
    d = runs["D"]
    t = d.t * 1e12
    epw_nm = d.epw_wavelengths * 1e9
    iaw_nm = d.iaw_wavelengths * 1e9
    fig, ax = plt.subplots(2, 3, figsize=(16.5, 8.4))

    def spectrogram(a, data, grid, title, *, pct=99.0, exclude=None):
        """
        Linear colour, limit taken from outside *exclude* nm of the probe line.

        The central feature's residual skirt is orders of magnitude above the
        satellites, so a limit set from the whole panel crushes them to black.
        """
        masked = np.ma.masked_invalid(data.T)
        scale_from = masked
        if exclude is not None:
            keep = np.abs(grid - args.probe_wavelength) > exclude
            if keep.sum() > 4:
                scale_from = masked[keep, :]
        vmax = (
            np.percentile(scale_from.compressed(), pct) if scale_from.count() else None
        )
        image = a.imshow(
            masked,
            origin="lower",
            aspect="auto",
            cmap="inferno",
            extent=[t[0], t[-1], grid[0], grid[-1]],
            vmin=0.0,
            vmax=vmax,
        )
        fig.colorbar(image, ax=a, extend="max")
        a.set(xlabel="time (ps)", ylabel="wavelength (nm)", title=title)
        return a

    a = spectrogram(
        ax[0][0],
        d.epw,
        epw_nm,
        r"EPW — ions /$\sqrt{R}$, electron drift /$\sqrt{R}$ (D)",
        pct=99.0,
        exclude=40.0,
    )
    blue, red = bohm_gross(d, args.probe_wavelength)
    a.plot(t, blue, "c--", lw=1.1, label="Bohm–Gross")
    a.plot(t, red, "c--", lw=1.1)
    unresolved = ~d.epw_resolved
    if unresolved.any():
        a.plot(
            t[unresolved],
            np.full(unresolved.sum(), epw_nm[0] + 4),
            "w.",
            ms=3,
            label="tail unresolved",
        )
    a.set_ylim(epw_nm[0], epw_nm[-1])
    a.legend(fontsize=8, loc="upper left")

    spectrogram(
        ax[0][1], d.iaw, iaw_nm, "IAW — same treatment (D)", pct=99.5, exclude=1.5
    )

    a = ax[0][2]
    for key, label, style in (
        ("A", "no rescale", "-"),
        ("B", r"ions /$\sqrt{R}$", "-"),
        ("D", "ions + electron drift", "-"),
        ("C", r"everything /$\sqrt{R}$", "--"),
    ):
        a.plot(t, feature_width(runs[key]), style, lw=1.3, label=label)
    a.set(
        xlabel="time (ps)",
        ylabel="IAW FWHM (nm)",
        title="the ion feature: what the mass ratio moves",
    )
    a.legend(fontsize=8)

    ax[1][0].semilogy(t, d.electron_density * 1e-6, "k-", lw=1.3)
    ax[1][0].set(
        xlabel="time (ps)",
        ylabel=r"$n_e$ (cm$^{-3}$)",
        title="electron density at the probe",
    )

    a = ax[1][1]
    for key, label, style in (
        ("D", "ions + electron drift  (correct)", "-"),
        ("C", r"everything /$\sqrt{R}$", "--"),
    ):
        a.plot(t, runs[key].alpha_epw / np.sqrt(2), style, lw=1.3, label=label)
    a.axhspan(0, 4, color="tab:green", alpha=0.12)
    a.text(t[1], 0.4, "validated to a few %", fontsize=8, color="tab:green")
    a.axhline(9.5, ls=":", color="tab:red")
    a.text(t[1], 10.2, "double-precision limit", fontsize=8, color="tab:red")
    a.set(
        xlabel="time (ps)",
        ylabel=r"$\alpha = 1/k\lambda_{De}$",
        ylim=(0, 24),
        title=r"a drift-only shift leaves $\alpha$ alone",
    )
    a.legend(fontsize=8, loc="upper left")

    a = ax[1][2]
    for key, label in (("B", r"ions /$\sqrt{R}$ only"), ("D", "+ electron drift")):
        mid = satellite_midpoint(runs[key], args.probe_wavelength)
        a.plot(
            t,
            mid - args.probe_wavelength,
            "o-",
            ms=2.5,
            lw=1.1,
            label=f"{label}  (rms {np.nanstd(mid):.1f} nm)",
        )
    a.axhline(0.0, ls=":", color="0.4", lw=1.0)
    a.set(
        xlabel="time (ps)",
        ylabel=f"satellite midpoint $-$ {args.probe_wavelength:.0f} (nm)",
        title="the electron Doppler shift, measured on the satellites",
    )
    a.legend(fontsize=7.5)
    a.text(
        0.02,
        0.03,
        "expected B$\\to$D change is ~5 nm;\nthe estimator scatters by more",
        transform=a.transAxes,
        fontsize=7,
        color="0.35",
    )

    fig.suptitle(
        f"pic_thomson on KinShock, z = {args.position:.2f} mm  —  reduced mass ratio "
        rf"$m_i/m_e={args.mass_ratio:.0f}$, $R=m_p/m_{{sim}}$"
        f"\n{args.probe_cells}-cell probe volume · "
        f"quadrature {args.quadrature_points:.0e} · "
        f"fixed {args.notch[0]:.0f}–{args.notch[1]:.0f} nm notch · "
        "linear colour scale (set from the satellites, not the probe line)",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    MEDIA.mkdir(exist_ok=True)
    path = MEDIA / "13_kinshock_470eV.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main() -> None:
    """Read the run, compare the four treatments, and draw the figure."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run", type=Path, help="the WarpX run directory")
    parser.add_argument("--position", type=float, default=0.69, help="probe z [mm]")
    parser.add_argument("--probe-wavelength", type=float, default=532.0)
    parser.add_argument("--theta", type=float, default=90.0, help="scattering angle")
    parser.add_argument("--mass-ratio", type=float, default=100.0, help="m_i/m_e")
    parser.add_argument("--epw", nargs=2, type=float, default=[400.0, 680.0])
    parser.add_argument("--iaw", nargs=2, type=float, default=[524.0, 540.0])
    parser.add_argument(
        "--notch",
        nargs=2,
        type=float,
        default=[526.0, 538.0],
        help="fixed stray-light notch [nm]; see the module docstring",
    )
    parser.add_argument("--spectral-bins", type=int, default=900)
    parser.add_argument("--velocity-bins", type=int, default=512)
    parser.add_argument("--position-bins", type=int, default=256)
    parser.add_argument(
        "--probe-cells",
        type=int,
        default=11,
        help="cells the collection volume spans; averaged, not summed",
    )
    parser.add_argument(
        "--quadrature-points",
        type=float,
        default=1e4,
        help="sample points for the principal-value integral giving chi",
    )
    parser.add_argument("--smoothing-width", type=float, default=0.25)
    parser.add_argument("--smoothing-iterations", type=int, default=2)
    args = parser.parse_args()

    ratio = const.m_p.si.value / (args.mass_ratio * ME)
    print(f"R = m_p / m_sim = {ratio:.2f}")
    species = read(args.run, args)

    treatments = {
        "A": {},
        "B": {"ion_velocity_scale_factor": ratio},
        "C": {"velocity_scale_factor": ratio},
        "D": {
            "ion_velocity_scale_factor": ratio,
            "electron_drift_scale_factor": ratio,
        },
    }
    runs = {}
    for key, treatment in treatments.items():
        t0 = time.time()
        runs[key] = spectra(species, args, **treatment)
        widths = feature_width(runs[key])
        print(
            f"  {key}: alpha median {np.nanmedian(runs[key].alpha_epw / np.sqrt(2)):5.2f}"
            f"  IAW FWHM median {np.nanmedian(widths):5.2f} nm"
            f"  EPW resolved {int(runs[key].epw_resolved.sum())}/{runs[key].n_time}"
            f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )
    figure(runs, args)


if __name__ == "__main__":
    main()
