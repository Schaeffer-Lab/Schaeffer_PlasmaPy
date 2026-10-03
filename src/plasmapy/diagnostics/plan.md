# Plan: `pic_thomson.py` — synthetic Thomson spectra from generic PIC output

Branch: `feature/pic-thomson-pipeline`
Target file: `src/plasmapy/diagnostics/pic_thomson.py` (single module, next to `thomson.py`)

Goal: fold the `osiris2thomson` pipeline into this fork as **one** module that takes
PIC output (OSIRIS, WarpX, and ideally anything else) and produces synthetic EPW/IAW
Thomson spectra via `thomson.arbitrary_forwardmodel`. Only what is needed for the
spectra — no temperature/B-field/`ufl`/`uth` diagnostics.

______________________________________________________________________

## 1. What the existing pipeline actually does

Source: `/home/hhelal/osiris2thomson/src/osiris2thomson/`, entry point
`synspectra.data_to_spectra()` (942-line module; the rest is smoothing, HDF5
loading, and moments helpers).

Traced end to end, `data_to_spectra` does the following:

| #   | Step                                                                                                                                  | Functions                               | Code-specific?                            |
| --- | ------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------- | ----------------------------------------- |
| 1   | Build OSIRIS filenames `MS/PHA/<field>/<species>/<field>-<species>-NNNNNN.h5` and read every timestep with `osh5io.read_h5`           | `file_name_phase`, `input_PHA_t`        | **Yes**                                   |
| 2   | Read `MS/FLD/<b_field>/…` B-field series                                                                                              | `file_name_mag`, `input_mag_t`          | **Yes** — *drop*                          |
| 3   | If the phase space is 3-D (`p1x1x2`/`p2x1x2`), sum over the transverse spatial axis to get `(t, p, x)`                                | inline                                  | Partly (reduce-to-1D concept is general)  |
| 4   | Momentum axis → velocity axis: `v = u·c/√(1+u²)` where `u = p/(m c)` is OSIRIS proper velocity                                        | inline                                  | Boundary — general once `u` is defined    |
| 5   | Pick the spatial slice \`y_slice = argmin                                                                                             | x − y_value                             | \`                                        |
| 6   | Zeroth moment along `p` → density vs `(t, x)`; scale by reference density `n` [cm⁻³]; first/second moments → drift + `T`              | `moments.moment`, `second_moment_to_eV` | Invariant (only the 0th moment is needed) |
| 7   | Ion fractions `ifract_s = n_s / Σ n_s` at the slice; presence masks (species below 1 % of reference is "absent")                      | `species_presence_mask`                 | Invariant                                 |
| 8   | Smooth VDFs: repeated boxcar (`uniform_filter1d`) along the velocity axis                                                             | `smooth_vdf`                            | Invariant                                 |
| 9   | Normalise so `∫f dv = 1` per (t, x); clip negatives; guard div-by-zero                                                                | `normalize_vdf`                         | Invariant                                 |
| 10  | Half-cosine taper of the VDF tails to kill the sharp PIC noise floor at the grid edge                                                 | `taper_vdfs` / `taper_vdf_edges`        | Invariant                                 |
| 11  | Floor at 1e-30                                                                                                                        | inline                                  | Invariant                                 |
| 12  | "Fudge factor": divide the velocity axes by `√rqm` and re-interpolate onto a padded grid, to undo the reduced ion/electron mass ratio | `rescale_and_pad_vdf`                   | Invariant (but see §5.1)                  |
| 13  | Per timestep, call `thomson.arbitrary_forwardmodel` twice — once over the EPW window (with a notch) and once over the IAW window      | `vdfs_to_spectra`                       | Invariant                                 |
| 14  | Instrument smoothing of the spectrogram (Gaussian 1-D/2-D or boxcar) to e.g. 100 ps / 0.5 nm                                          | `smooth_spectra.*`                      | Invariant                                 |
| 15  | Write everything to a structured HDF5 file                                                                                            | `create_hdf5_file`                      | Invariant                                 |
| 16  | Plot the two spectrograms                                                                                                             | `plot_spectra`                          | Invariant                                 |

**The code-specific part is steps 1–4 only.** Everything from step 5 on operates on a
plain `(n_time, n_v, n_x)` array plus a velocity axis in m/s — exactly the invariance
the task statement assumes.

### What we drop

Temperature (`second_moment_to_eV`, `Tis`, `Te`), the B-field read and
`ion_gyro`, drift velocities as *outputs*, the `TEMPERATURE` /
`PERPENDICULAR_MAGNETIC_FIELD` / `FLOW_VELOCITY` HDF5 groups,
`load_spectra.py`, and `scripts/plot_spectra.py`. Note the 1st/2nd moments are
still computed *inside* `arbitrary_fast_spectral_density_arbdist`
(`thomson.py:390-421`) — the forward model derives drift and thermal speed from the
VDFs itself, so we never need them at pipeline level.

The `moments.py` helper collapses to a single `∫f dp` (Simpson) call; not worth a
separate abstraction.

______________________________________________________________________

## 2. The invariance boundary

One dataclass is the contract between readers and physics:

```python
@dataclass(frozen=True)
class PICPhaseSpace:
    """Reduced 1V phase space f(t, v, x) for a single species, in SI."""

    f: np.ndarray  # (n_time, n_v, n_x), arbitrary normalisation
    v: np.ndarray  # (n_v,) lab-frame velocity along the diagnostic axis [m/s]
    x: np.ndarray  # (n_x,) position along the same axis [m]
    t: np.ndarray  # (n_time,) [s]
    label: str  # PlasmaPy `ParticleLike`, e.g. "e-", "Al 13+", "p+"
    is_electron: bool
    meta: dict  # code name, source paths, normalisations used
```

Rules the readers must satisfy:

- `v` is **lab-frame velocity in m/s** (relativistically correct: `v = u c/√(1+u²)`
  from proper velocity `u`), monotonically increasing, but **not** required to be the
  same grid across species (the forward model takes `e_velocity_axes` and
  `i_velocity_axes` separately, `thomson.py:601-604`).
- `f` may be in arbitrary units; the pipeline normalises. It must, however, be
  *proportional to the true phase-space density* — i.e. macroparticle **weights** must
  be used when histogramming, not raw counts (this bites WarpX; see §3.2).
- Species with different `x` grids are resampled onto the electron grid by the
  pipeline, so readers need not agree on spatial resolution.

Everything downstream consumes only `PICPhaseSpace`. Adding a third PIC code =
writing one reader function.

______________________________________________________________________

## 3. Readers

### 3.1 OSIRIS

The `osiris2thomson` version depends on `osh5io` from the pyVisOS submodule.
**We drop that dependency**: an OSIRIS phase-space file is trivially readable with
plain `h5py`, which is already a core PlasmaPy dependency. Verified against
`OmegaShock/runs/omegashock_w3.5e11_exp/MS/PHA/p1x1/e/p1x1-e-000010.h5`:

```
/p1x1               (1024, 512)        dataset, C-order (p1, x1)
/AXIS/AXIS1         [xmin, xmax]       NAME=x1,  UNITS=c/\omega_p   -> last numpy axis
/AXIS/AXIS2         [pmin, pmax]       NAME=p1,  UNITS=m_e c        -> first numpy axis
root attrs:         TIME (1/\omega_p), ITER, NAME, LABEL
/SIMULATION attrs:  DT, NDIMS, NX, XMIN, XMAX
```

So: numpy axis `k` ↔ file `AXIS{ndim-k}` (Fortran/C order flip), axis values from a
2-element `[min, max]` pair plus the dataset shape. OSIRIS momenta are stored as
proper velocity `u = γv/c` normalised to `m_e c` **for every species** (confirmed by
the `UNITS` attribute), so `v = u c/√(1+u²)` is correct for ions as well and no per-
species mass enters here.

Reader signature:

```text
def read_osiris_phase_space(ms_path, field, species, timesteps, *,
                            is_electron=None, label=None,
                            transverse_axis="sum") -> PICPhaseSpace
```

- `timesteps`: iterable of dump indices, or `None` = every file present.
- 3-D phase spaces (`p1x1x2`, `p2x1x2`) are reduced by summing the transverse
  spatial axis, resolved from the `AXIS*/NAME` attributes rather than by
  string-matching the field name (the current code hard-codes `p2x1x2`/`p1x1x2` and
  raises otherwise — `synspectra.py:592-619`).
- Spatial axis converted from `c/ω_p` to metres using a reference density
  (needed anyway for the physical density scale).
- Time converted from `1/ω_p` to seconds with the same `ω_p`.

### 3.2 WarpX

WarpX writes AMReX plotfiles (verified: `KinShock2020/runs/R1_paper/diags/diag1000000/`
contains `Header`, `WarpXHeader`, `Level_0/`, and one directory per species) — **raw
macroparticles, not a binned phase space**. So the WarpX reader must do the binning
that OSIRIS does in-code:

```text
def read_warpx_phase_space(diag_glob, species, *, mass, n_v=512, n_x=512,
                           v_range=None, x_range=None, axis="z",
                           backend="auto") -> PICPhaseSpace
```

1. Enumerate plotfiles (`sorted(glob("diag1*"))`), one per timestep.
1. Per frame, read `particle_position_*`, `particle_momentum_<axis>`,
   `particle_weight`. Backends: `openpmd-api` if the run wrote openPMD, else `yt`
   (what `KinShock2020/src/kinshock/io.py:71-192` uses today). Both are **optional
   imports** with a clear error message — neither belongs in PlasmaPy's core deps.
1. `u = p_axis / (m c)`; `v = u c/√(1+u²)`.
1. `f[t] = np.histogram2d(v, x, bins=[v_edges, x_edges], weights=w)` — weights are
   essential; a raw count histogram is not a distribution function.
1. Bin edges fixed across all timesteps (computed from a percentile of the first and
   last frames, or user-supplied) so `f` is a well-defined array.
1. WarpX is already SI, so no unit conversion — `v` in m/s, `x` in m, `t` in s
   straight from the dataset.

Note WarpX 1-D runs put the propagation direction in `particle_position_x` even when
the deck calls it `z` (see `kinshock/io.py:121`); the reader takes an explicit
`axis` argument and does not guess.

Because binning is expensive (21 GB of plotfiles for `R1_paper`), the reader should
support a `cache=` path that memoises the binned `(t, v, x)` array to `.npz`.

### 3.3 Generic

`from_arrays(f, v, x, t, ...)` — build a `PICPhaseSpace` directly, so a user with a
third code (PSC, Smilei, EPOCH, hybrid) or a hand-built distribution only has to
produce the array. This is also what the unit tests use.

______________________________________________________________________

## 4. Module layout

Single file, sections in this order:

```
pic_thomson.py
├── __all__
├── PICPhaseSpace                     (dataclass, §2)
├── ThomsonSpectrogram                (dataclass: epw, iaw, wavelengths, t, alpha,
│                                      density, ifract, masks, meta)
│
├── ── readers ─────────────────────────────────────────────
├── read_osiris_phase_space(...)      §3.1  (h5py only)
├── read_warpx_phase_space(...)       §3.2  (optional yt / openpmd-api)
├── from_arrays(...)                  §3.3
│
├── ── conditioning (code-invariant) ───────────────────────
├── _number_density(f, v)             ∫f dv  (Simpson)
├── smooth_vdf(f, window, iterations)
├── normalize_vdf(f, v)               each (t, x) slice integrates to 1
├── taper_vdf_edges(f, threshold)     vectorised half-cosine rolloff
├── rescale_velocity_axis(f, v, factor, target_v)   √R mass-ratio rescale (§5.1)
├── species_presence_mask(n, n_ref, threshold)
│
├── ── forward model driver ────────────────────────────────
├── spectra_from_phase_spaces(electrons, ions, geometry, ...) -> ThomsonSpectrogram
│
├── ── instrument response ─────────────────────────────────
├── apply_instrument_response(spec, fwhm_time, fwhm_wavelength)
│
└── ── output ──────────────────────────────────────────────
    ├── ThomsonSpectrogram.to_hdf5(path)
    └── ThomsonSpectrogram.plot(...)
```

### Public API sketch

```text
# 1. read (code-specific)
e   = read_osiris_phase_space("…/MS", "p1x1", "e",    timesteps, is_electron=True)
al  = read_osiris_phase_space("…/MS", "p1x1", "cham", timesteps, label="Al 13+")

# 2. everything after this is code-agnostic
spec = spectra_from_phase_spaces(
    electrons=e,
    ions=[al],
    position=80 * u.mm,                     # or index / fractional position
    reference_density=9e17 * u.cm**-3,
    probe_wavelength=532 * u.nm,
    epw_wavelengths=np.linspace(457, 607, 500) * u.nm,
    iaw_wavelengths=np.linspace(525, 539, 500) * u.nm,
    notch=[525, 540] * u.nm,
    probe_vec=[1, 0, 0],
    scatter_vec=[cos(63°), sin(63°), 0],
    velocity_scale_factor=50.0,             # mass-ratio reduction R; see §5.1
    smoothing=dict(window=40, iterations=3),
)

spec.to_hdf5("spectra.h5")
spec.plot()
```

`spectra_from_phase_spaces` is the invariant core; the OSIRIS/WarpX difference never
reaches it.

A thin `main()` + `argparse` CLI (`python -m plasmapy.diagnostics.pic_thomson …`)
driven by a YAML/JSON config is a nice-to-have, deferred until the API settles.

______________________________________________________________________

## 5. Physics issues found in the existing pipeline

These are things to fix or explicitly decide, not to port verbatim.

### 5.1 The "fudge factor" — resolved: global rescale, user-supplied

**Settled (user, this session):** the reduced-mass-ratio runs stretch the velocity
axis by `√R` so that Mach-number-like quantities stay invariant when `m_i/m_e` is
reduced by `R`. Recovering physical velocities therefore means dividing **every**
species' axis by `√R` — which is what `synspectra.py:699-712` does. The behaviour is
kept as-is; the correct value of `R` depends on the simulation setup, so it stays a
user-facing argument.

What changes is only naming and defaults, because the old signature conflates two
different quantities:

- `ion_rqms` is documented as "the mass-to-charge ratios of the ion species … from the
  OSIRIS input deck" and is fed to `second_moment_to_eV(…, rqm)` — where it
  unambiguously means the true species `m/mₑ` — while `scale = √ion_rqms[0]` uses it
  as the mass-ratio *reduction* factor. Dropping the temperature diagnostic removes
  the conflict outright.
- New API: `velocity_scale_factor: float = 1.0`, documented as "divide all velocity
  axes by `√velocity_scale_factor`; pass the factor by which the ion/electron mass
  ratio was reduced". Default `1.0` = no correction, so nothing is applied silently.
- Applied uniformly to electrons and every ion species, as today.

One value question to settle when we run the OSIRIS case (not a design question):
OmegaShock's deck has `cham` `rqm = 69` with `rqm_factor: 50` in `run.yaml`, so `R`
is 50, not 69 — whereas the old notebook passed `ion_rqms=[100]`. Confirm `R = 50`
for `omegashock_w3.5e11_exp` at test time.

`rescale_velocity_axis` stays a general "scale this axis by λ, re-interpolate onto a
target grid, zero-pad outside" primitive; the policy lives in
`spectra_from_phase_spaces`.

### 5.2 Ion VDFs are normalised against the electron velocity axis — bug

`synspectra.py:676`:

```text
ions_pyt = [normalize_vdf(ion_pyt, v_e, …) for ion_pyt in ions_pyt]
#                                    ^^^ should be v_ion
```

The forward model assumes `∫f dv = 1` (it takes moments of `efn`/`ifn` directly,
`thomson.py:392-397, 412-417`, and interpolates `ifn` as a PDF at `thomson.py:551`).
In the OmegaShock deck the electron axis spans `u ∈ [−1, 1]` and the `cham` axis
`u ∈ [−0.1, 0.1]` — a factor ~10 in `Δv` — so ion `f` is currently normalised ~10×
wrong, biasing `χ_i` and hence the whole IAW feature. **Fix: each species normalises
on its own axis.**

### 5.3 Each spectrum is renormalised to unit area

`thomson.py:583`: `Skw = Skw / np.trapezoid(Skw, wavelengths)`. Every timestep's
spectrum integrates to 1 over its own window, so the spectrogram carries **shape
only** — no absolute intensity, no EPW-vs-IAW relative weight, and the time axis of a
spectrogram is not a brightness history. This is a property of the forward model, not
the pipeline, but the new module must document it and the plot should not be
labelled in a way that implies absolute power. (Recovering absolute scale would mean
tracking `n_e` and the Thomson cross-section separately — out of scope, but worth a
docstring note.)

### 5.4 Smoothing asymmetry

Electrons get `num_iterations=3` boxcar passes; ions get `num_iterations=0`
(`synspectra.py:672-673`) — i.e. ions are not smoothed at all despite the call. Likely
intentional (the IAW feature is narrow and easily washed out) but undocumented. New
module: separate `electron_smoothing` / `ion_smoothing` arguments, both explicit.

### 5.5 `smooth_vdf` takes `abs()` while `normalize_vdf` clips negatives

Two contradictory policies for the same PIC shot noise, applied in sequence
(`synspectra.py:157` vs `:126`). `abs()` reflects noise into fake signal. Pick one:
clip, and warn (the `normalize_vdf` behaviour, which already has the right comment).

### 5.5a The taper fabricates a pedestal — **found during implementation**

`taper_vdfs` rolls the distribution off from the signal edge all the way to the
*grid boundary*. When the distribution occupies only a small part of the velocity
grid, that rolloff runs across a long stretch of empty axis, and the fabricated
pedestal lands at large :math:`|v|` — exactly where the :math:`v^2` weighting of the
second moment is largest. Measured on a Maxwellian at the default
`threshold_frac=0.005`, the recovered width comes out:

| grid half-width | width error |
| --------------- | ----------- |
| 4σ              | +0.4 %      |
| 6σ              | +5.4 %      |
| 8σ              | +13.7 %     |
| 12σ             | +40.4 %     |

The forward model reads the thermal speed straight off the VDF
(`thomson.py:392-397`), so this propagates directly into `α`, the Bohm-Gross shift,
and any temperature inferred from the result. **This matters for OmegaShock**: the
electron phase space spans `u ∈ [−1, 1]`, i.e. ±c, while a 100 eV electron
distribution has σ ≈ 0.014 c — roughly a 70σ half-width. Whether the pipeline is
actually in this regime depends on where the raw PIC noise floor crosses
`0.005 × peak`; if the shot noise reaches the grid edges the taper does little, and
if it does not the second moment is badly inflated. **Check this explicitly on real
OSIRIS data before trusting the OmegaShock spectra** — it is a candidate explanation
for any anomalously broad EPW feature in the existing outputs.

Implemented: the default is left numerically identical to the original (so the
OSIRIS comparison stays apples-to-apples), but `taper_vdf_edges` grows a
`max_taper_bins` argument that bounds the rolloff, and a `pedestal_warning`
(default 5%) that raises a `RuntimeWarning` when the taper has materially widened
any slice.

### 5.6 Dead/vestigial code to not carry over

`moving_average_numpy` (unused), the `fill_value=1e-20` in `rescale_and_pad_vdf`
described in its own comment as "zero", `import scipy.integrate as integrate` inside
`data_to_spectra` (unused), `t[:, 0]` indexing that assumes OSIRIS's odd
`(n_time, 1)` time array.

### 5.7 Performance

`vdfs_to_spectra` loops over timesteps and calls the numba-JIT'd
`arbitrary_chi` per wavelength window. For 513 OSIRIS dumps × 2 windows × 500
wavelengths this is the dominant cost. Keep the `tqdm` progress bar, and add an
optional `n_jobs`/chunking hook later — do not optimise before the physics is right.

______________________________________________________________________

## 6. Testing

### 6.1 Unit tests (`tests/diagnostics/test_pic_thomson.py`)

1. **Reader round-trip** — `from_arrays` → conditioning → shapes/units preserved.
1. **`normalize_vdf`** — `∫f dv = 1` for every `(t, x)` on a non-uniform-amplitude
   input; per-species axes respected (the §5.2 regression).
1. **`taper_vdf_edges`** — vectorised version matches a straightforward per-slice loop
   (the existing repo has this equivalence test; port it).
1. **Maxwellian consistency** — build an analytic Maxwellian `f(v)` at known `n`, `T`,
   drift; push it through `spectra_from_phase_spaces`; compare against
   `thomson.spectral_density` (the standard PlasmaPy Maxwellian path) on the same
   geometry. This is the real validation that the pipeline's conditioning does not
   distort the physics, and it is code-independent.
1. **OSIRIS reader** — against a tiny committed fixture (one 8×8 phase-space HDF5
   written by the test itself in OSIRIS layout), asserting axis order, the
   `AXIS{ndim-k}` flip, and the `c/ω_p → m` conversion.
1. **Presence masking** — a species that vanishes mid-run yields `NaN` columns, not
   fabricated spectra.

### 6.2 End-to-end against the two shock runs

Both are magnetized piston-driven shocks, which makes a genuine cross-code
comparison possible.

**OSIRIS —** `/home/hhelal/OmegaShock/runs/omegashock_w3.5e11_exp/`

- `MS/PHA/p1x1/{e, cham, targ}/`, 513 dumps, `ps_np = 1024`, `ps_nx = 512`
- deck: `e` rqm −1; `cham` rqm 69; `targ` rqm 68
- `run.yaml`: `reference_density: 9.0e17` cm⁻³, `rqm_factor: 50`, `dx: 0.075`,
  `tmax_gyroperiods: 4`
- electron `u ∈ [−1, 1]`, `cham` `u ∈ [−0.1, 0.1]`, `targ` `u ∈ [−0.05, 0.05]`
- Reference: the existing `osiris2thomson` output `spectra.hdf5` /
  `/home/hhelal/sim_w/MS_spectra.hdf5`. Acceptance: after fixing §5.2 the new EPW
  spectrogram should agree with the old one to within the smoothing tolerance; the IAW
  will legitimately differ, and that difference must be explainable by the
  normalisation fix (check it moves the IAW width in the direction `√10` implies).

**WarpX —** `/home/hhelal/KinShock2020/runs/R1_paper/`

- `diags/diag1*` — Full diagnostics with particles, ~50 frames (`diag1.intervals = 6448`)
- species: `piston_electrons`, `piston_ions`, `amb_electrons`, `amb_ions`
- `mass_ratio = 100`, `theta_e_heat = 0.092`; densities in `config.yaml`
- `diag_fields*` (1289 frames) are field-only (`write_species = 0`) — **not** usable
  for VDFs; ignore them.
- Two electron populations here (piston + ambient). `arbitrary_forwardmodel` supports
  `efract`, so the pipeline should accept a *list* of electron `PICPhaseSpace` and
  build `efract` the same way it builds `ifract` — the OSIRIS pipeline only ever
  passes one. **This is a real API requirement the old code does not have.**

**Cross-code check:** at matched `t·ω_ci`, compare `α`, the EPW peak separation
(→ `n_e`), and the IAW shape. Exact agreement is not expected (different drivers,
different mass ratios); the deliverable is that both paths run through the *same*
invariant core and produce physically sane spectra.

______________________________________________________________________

## 7. Open questions

1. ~~§5.1 mass-ratio convention~~ — resolved: global `1/√R` rescale of all species,
   exposed as `velocity_scale_factor` (default 1.0). Only the numeric value of `R`
   for `omegashock_w3.5e11_exp` (50 vs 69) remains, and that is a test-time check.
1. ~~File name~~ — resolved: `pic_thomson.py`.
1. **Optional deps.** `yt` and `openpmd-api` for WarpX — add a
   `[project.optional-dependencies] pic` extra, or leave them as import-time errors
   with instructions? I lean toward the extra, mirroring the existing `thomson`
   extra for numba/torch.
1. **Multiple electron populations** (§6.2 WarpX): confirm we want `efract` support
   in v1. I plan to build it in, since the WarpX test run needs it.
1. **Upstreamability.** This is a fork-specific module depending on
   `arbitrary_forwardmodel`, which upstream PlasmaPy does not have. Keeping it in
   `plasmapy/diagnostics/` is fine for the fork; worth a header note that it is not an
   upstream-mergeable file as written.

______________________________________________________________________

## 8. Implementation order

1. ✅ **Done.** `PICPhaseSpace`, `from_arrays` + conditioning functions (with §5.2,
   §5.4, §5.5, §5.5a addressed) — plus unit tests 1–4. 70 tests, all passing;
   `ruff check`, `ruff format` and `ty` clean.

   The Maxwellian cross-check (test 4) agrees with `thomson.spectral_density` to a
   normalised L1 difference of **0.036**, with both EPW peaks matching to within one
   wavelength bin (0.31 nm). `ThomsonSpectrogram` was deferred to step 2, since its
   fields follow from the driver's design.

   Two things the tests pinned down, beyond §5.5a:

   - The `eps` guard in `normalize_vdf` (`1e-30 + 1e-12 × max|integral|`) biased
     low-amplitude slices by up to 1e-6 relative when amplitudes span decades.
     Replaced with an exact `np.divide(..., where=integral > 0)`.
   - `arbitrary_forwardmodel` reports `α = √2 · ω_pe / (k σ)` with σ the VDF standard
     deviation, whereas `spectral_density` reports `1/(k λ_De)`. The two differ by
     exactly √2 (measured ratio 1.4143). A test documents this so it is not later
     mistaken for a physics bug. `S(k,ω)` itself is unaffected — `arbitrary_chi`
     uses the normalisation consistently.

1. ✅ **Done.** `spectra_from_phase_spaces` driver, `ThomsonSpectrogram`, `efract`
   support, presence masking. 98 tests in the file, all passing; `ruff` and `ty`
   clean; the full `tests/diagnostics/` suite is green (209 passed).

   Design decisions settled while implementing:

   - **`reference_density` semantics.** Defined as "the physical density
     corresponding to a unit zeroth moment of the supplied phase space". For OSIRIS
     that is `n0`; a reader that already emits SI passes `1 * u.m**-3`. Only the
     absolute density and the presence threshold need it — `ifract` and `efract` are
     ratios and so are normalisation-free.
   - **Fractions are renormalised over the present populations.** The old pipeline
     passed the raw fractions of the surviving species, which then summed to less
     than one; the forward model uses them to split the density
     (`ne = efract * n`, `ni = ifract * n / zbar`), so that silently lost density.
   - **Presence is judged on the raw phase space**, before the conditioning floor,
     which would otherwise make an empty slice look populated. Covered by a test.
   - **Vacuum timesteps** — total electron density below
     `presence_threshold * reference_density` — produce `NaN` rows rather than a
     fabricated spectrum.
   - **Per-species spatial grids** are handled by each species picking its own
     nearest grid point, so readers need not agree on spatial resolution.
   - Conditioning happens inside the driver via `electron_conditioning` /
     `ion_conditioning` dicts (§5.4's asymmetry is now explicit rather than
     implied), with `{"skip": True}` to accept pre-conditioned input.

   One more forward-model convention pinned down by a test: the driver's default
   `scattered_power=True` returns power per unit wavelength, which differs from
   `spectral_density`'s `S(k, ω)` by the frequency-to-wavelength Jacobian
   `(1 + 2Δω/ω_probe) · 2/λ²`. Comparing the two directly gives a spurious L1
   difference of 0.092; with `scattered_power=False` the driver agrees with the
   analytic model at **L1 = 0.032**, and the Jacobian relation reproduces the
   `True` output to a relative error of 7e-18.

1. ✅ **Done.** OSIRIS reader (h5py, no pyVisOS) + unit test 5. 118 tests in the
   file, all passing; full `tests/diagnostics/` suite green (222 passed).
   Also added `tools/pic_thomson_figures.py`, which renders the behaviour the
   tests assert into a git-ignored `media/` directory.

   Confirmed against the real run and the deck:

   - **Momentum is normalised to each species' own mass**, despite the generic
     `m_e c` label OSIRIS writes in the file. The deck's boundary thermal speeds
     settle it: `uth_e / uth_cham = 3.7543e-2 / 2.1868e-3 = 17.17`, which matches
     `√((T_e/m_e)/(T_i/m_i)) = 17.2` for species-mass normalisation and gives
     T_e = 721 eV, T_i = 169 eV. The `m_e c` reading would imply T_i = 0.04 eV.
     So `v = u c/√(1+u²)` is right for every species.
   - **Electron phase space is a negative charge density** (min −2544, max 0).
     The old pipeline's `abs()` inside `smooth_vdf` was load-bearing for this, not
     just noise handling — removing it in step 1 would have zeroed every electron
     distribution. Sign handling now lives in the reader, where the code's
     convention belongs, along with dividing by the charge number so the zeroth
     moment is a *number* density.
   - **The `u → v` map needs its Jacobian.** `f(v) = f(u) γ³/c`. The old pipeline
     relabelled the axis without it. With the Jacobian, `∫f dv` equals the density
     in units of `n0` — validated on real data: the upstream electron density
     reads **n/n₀ = 1.037**.

   Reader-level validation on `omegashock_w3.5e11_exp`: the EPW satellite tracks
   the plasma frequency across the shock — observed shift 9.8 → 20.8 nm against a
   bare `ω_pe` prediction of 8.2 → 18.7 nm, a ratio of 1.1–1.3, which is the
   expected Bohm-Gross excess.

### 3a. The taper pedestal, answered on real data (see §5.5a)

The question flagged in §5.5a is settled, and the answer is worse than the
synthetic estimate. Median width inflation across all appreciably populated cells:

| species | grid half-width | unbounded   | bins=20 | bins=10 | bins=5 | bins=3 |
| ------- | --------------- | ----------- | ------- | ------- | ------ | ------ |
| `e`     | 13σ             | **+67 %**   | −0.1 %  | −0.7 %  | −0.9 % | −1.0 % |
| `cham`  | 14σ             | **+4086 %** | +35 %   | +11 %   | +3.9 % | +1.6 % |
| `targ`  | 23σ             | **+242 %**  | +6.0 %  | +2.8 %  | +1.3 % | +0.7 % |

The ions are catastrophic because their momentum grid (`u ∈ ±0.1`, `±0.05`) is far
wider than the actual thermal spread — cold upstream ions occupy a handful of bins
out of 1024. **The old pipeline's unbounded taper inflated the ion width by a
factor of ~40**, which together with the ion-normalisation bug (§5.2, factor 10)
means the IAW output of the existing `spectra.hdf5` should not be trusted.

Recommended settings for this run: `max_taper_bins=20` for electrons,
`max_taper_bins=3` for ions. A fixed bin count is a blunt knob — a bound expressed
as a fraction of each slice's own width would be better, and is worth considering
in step 7.

Two diagnostics were also de-noised so they mean something on real data: the
negative-value warning in `normalize_vdf` now ignores round-off (boxcar smoothing
leaves ~1e-20 negatives against a 1e-5 peak, which produced 3.2 million spurious
warnings), and the pedestal check ignores cells carrying under 1 % of the peak
weight (an almost-empty vacuum cell has near-zero width, so any taper multiplies
it enormously — it was reporting 14 575 093 %).

4. ✅ **Done.** End-to-end run on `omegashock_w3.5e11_exp`, all 513 dumps, at
   x = 5.00 mm. Driven by `tools/pic_thomson_osiris_comparison.py`, which runs the
   pipeline twice — `legacy` reproducing the old behaviour, `corrected` with the
   taper bounded — and writes `media/07_osiris_end_to_end.png`.

   **No old-pipeline output exists for this run**, so the numerical regression
   against a `spectra.hdf5` was not possible here. The only such file,
   `osiris2thomson/spectra.hdf5`, came from a *different* run
   (`omegashock_w3e12_rqm100_dx0`, n₀ = 1.83e18, 52 dumps at stride 10, sampled
   3 mm into a 4.7 mm domain, `ion_rqms=[74, 68]`, both ions labelled `"p"`). The
   script takes `--reference` and all of those as options, so that comparison can
   still be run against `~/osiris2thomson/MS` whenever it is wanted.

   **The physics validation that replaces it is stronger than the regression would
   have been.** The EPW satellite must sit at the plasma frequency plus a
   Bohm-Gross excess, and the size of that excess is fixed by α, which the forward
   model reports independently:

   | quantity                                                      | value     |
   | ------------------------------------------------------------- | --------- |
   | observed EPW shift / bare `ω_pe` shift, median over 513 steps | **1.291** |
   | `√(1 + 3k²λ_De²)` from the reported α                         | **1.262** |

   Agreement to 2.3%, over the whole run, validating the entire chain: reader unit
   conversions, the `γ³/c` Jacobian, conditioning, and the forward model. Getting
   this required excluding timesteps where the satellite falls inside the notch —
   with the old config's wide `[525, 540]` notch the "peak" is just the notch edge
   and the ratio reads a meaningless 1.579. At n₀ = 9e17 the satellites sit only
   ±8 nm from the probe line, so the notch must be narrower; `[530, 534]` is used.

   **What bounding the taper changes, on real data:**

   | metric    | legacy (unbounded) | corrected (bounded) |
   | --------- | ------------------ | ------------------- |
   | α, median | 1.685              | **3.182**           |
   | α, range  | 1.61 – 10.28       | 1.89 – 15.38        |

   The two configurations differ from each other by a median L1 of **0.900** in
   the EPW window and **0.794** in the IAW window.

   α is a factor **1.888** too small in the legacy configuration — the direct
   consequence of the inflated `vTe` (§3a), since α ∝ 1/`vTe`. For scale, the
   analytic-Maxwellian agreement in step 1 was L1 = 0.032; an L1 of 0.9 between
   legacy and corrected means the two spectrograms are essentially unrelated. The
   figure shows it plainly: the legacy panels are speckled, and after t = 0.5 ns
   the legacy EPW smears into a noisy 480–580 nm band while the corrected
   satellites stay coherent.

   **Conclusion: spectra produced by the old pipeline should be regenerated.**
   Between the ion-normalisation bug (§5.2, factor 10 on `χ_i`), the unbounded
   taper (§3a, `vTe` inflated 67%, ion widths by up to 40×), and the missing
   Jacobian, the differences are not refinements.

### 4a. A performance fix the full run forced

Conditioning the whole `(n_time, n_v, n_x)` block and slicing afterwards peaked at
**88 GB** of RSS on this run and had to be killed: `velocity_scale_factor` oversamples
the velocity axis eightfold, making each conditioned species
513 × 8192 × 512 × 8 B ≈ 17 GB, of which only one spatial column is ever used.

Every conditioning step acts independently on each `(time, position)` slice, so the
driver now reduces to the sampled point *first*, via `PICPhaseSpace.at_position`.
A test asserts the two orderings give bit-identical results. The full run now
completes comfortably.

### 4b. Ion species — resolved: fully stripped carbon

**Settled (user):** `cham` and `targ` are both **fully stripped carbon**, `C 6+`.
That matches what the deck implied — every fully stripped low-Z ion sits at
A/Z ≈ 1.99, and `"p+"` at 1.00 was never consistent with it:

| species | deck rqm | implied A/Z (R = 50) | `"p+"` gives | fully-stripped low-Z gives                       |
| ------- | -------- | -------------------- | ------------ | ------------------------------------------------ |
| `cham`  | 69       | 1.88                 | 1.00         | ~1.99 (He²⁺, C⁶⁺, N⁷⁺, O⁸⁺, Si¹⁴⁺ all within 1%) |
| `targ`  | 68       | 1.85                 | 1.00         | ~1.99                                            |

Measured on the run at stride 32, `"p+"` against `C 6+`:

| quantity         | change                       |
| ---------------- | ---------------------------- |
| IAW rms width    | **2.22× too wide** with `p+` |
| IAW spectrum, L1 | **0.631**                    |
| EPW spectrum, L1 | 0.070                        |
| α                | unchanged (ratio 1.0000)     |

Note the IAW width ratio is **2.22, not the naive √(A/Z) = 1.41**. The simple
scaling assumes the feature follows the ion thermal speed at fixed temperature,
but the ion VDF handed to the forward model does not change with the label at
all — only `Z` and `m` do, and they enter through `χ_i`, whose coefficient
carries `ω_pi² ∝ n_i q²/m` while `n_i = ifract·n/z̄` itself falls by `z̄ = 6`.
The kinetic treatment therefore gives a larger effect than the fluid estimate.
That α and the EPW are essentially untouched is the expected counterpart: both
are set by the electrons.

**One loose end.** `C 6+` has A/Z = 1.987, which needs `rqm_factor = 52.9` for
`cham` and 53.6 for `targ`, while `run.yaml` says 50 — a 6% discrepancy, so
either the deck's `rqm` values are rounded or the reduction factor is really
~53. It propagates into `velocity_scale_factor` as a 3% shift in every velocity
(`√53/√50`). Small, but worth resolving.

`tools/pic_thomson_osiris_comparison.py --ion-rqm` prints this consistency check
on every run, now reporting the `rqm_factor` the chosen label would require.
5\. ✅ **Done.** WarpX reader + caching. 145 tests in the file, all passing; full
`tests/diagnostics/` suite green (246 passed); `ruff` and `ty` clean.

`read_warpx_phase_space` bins raw macroparticles into a `PICPhaseSpace`.
Confirmed against `KinShock2020/runs/R1_paper` (1-D, 51 particle plotfiles,
~3M macroparticles per species per frame, ~0.6 s to read one):

- **Weights carry the density.** `f` is divided by the bin volume, so
  `∫f dv` is a number density in m⁻³ and the driver takes
  `reference_density = 1 * u.m**-3`. Validated twice: `sum(w)/volume` from the
  raw yt arrays gives 7.979e15 m⁻³ against the deck's `namb = 0.008 n0 = 8e15`,
  and the binned reader reproduces **7.9995e15 m⁻³** — 0.006%.
- **`mass` and `label` are different things** and both are required. `mass` is
  the *simulation's* mass (here `Mi = 100 mₑ`), which converts stored momentum
  into the velocities the run actually evolved; `label` names the physical
  species, from which the forward model takes charge and mass. Conflating them
  in a reduced-mass-ratio run is a factor-of-18 error.
- **1-D field naming.** WarpX stores the single spatial coordinate as
  `particle_position_x` even when the deck calls that direction `z`, while
  momenta keep their physical names — so the defaults are
  `particle_position_x` and `particle_momentum_z`, both overridable.
- **Caching** to `.npz`, keyed on a signature of every setting that affects the
  result, so a changed bin count rebuilds rather than silently returning a
  stale grid.
- The auto-derived velocity range is clamped below `c`. Without it, the 1.2×
  headroom around a 0.95`c` particle put the grid edge past `c`, where no
  particle can live and the taper would have had more empty axis to invent a
  tail across.

`yt` is an optional dependency, imported on demand with an actionable error,
and added to `pyproject.toml` as the `pic` extra. The OSIRIS path still needs
only `h5py`.

**Cross-code check, unplanned but valuable:** the driver consumed the WarpX
phase spaces with no code-specific handling, and `efract` handed over from
ambient electrons (100%) to piston electrons (99.7%) as the piston reached the
sampling point — the multi-electron-population path that only the WarpX case
exercises.

### 5a. R1_paper is not a collective-Thomson plasma at 532 nm

Worth knowing before step 6 sets expectations. At the sampled point:

|                  | R1_paper (WarpX) | omegashock_w3.5e11_exp (OSIRIS) |
| ---------------- | ---------------- | ------------------------------- |
| `n_e`            | 8e15 – 2e17 m⁻³  | 9e23 – 4e25 m⁻³                 |
| α at 532 nm, 90° | **~1e-5**        | 1.9 – 15.4                      |

Seven orders of magnitude in density puts R1_paper in the **deeply non-collective**
regime, where there are no EPW or IAW features at all: `S(k, ω) ∝ f_e(ω/k)`, so the
spectrum is just the electron distribution read through the Doppler map, and the
Doppler width spans essentially the whole visible. That is a property of the
simulation — it is scaled to ion-scale physics with reduced electron parameters,
not to a Thomson diagnostic — not a defect in the pipeline.

It does make a *different* and rather clean validation available for step 6:
in the non-collective limit the computed spectrum must reproduce the conditioned
electron VDF under `v = (λ - λ₀)c / (2 λ₀ sin(θ/2))`. A meaningful *collective*
WarpX comparison would need either a denser run or a much longer probe wavelength
(α scales with λ_probe).
6\. ✅ **Done.** End-to-end WarpX run on `R1_paper`, all 51 particle plotfiles,
at x = 30 m. Driven by `tools/pic_thomson_warpx.py`, which writes
`media/08_warpx_phase_space.png` and `media/09_warpx_spectra.png`.

**The non-collective limit gives an exact validation.** With α ~ 1e-5 the
electron susceptibility vanishes, the shielding factor `|1 - χ_e/ε|²` goes to
one, and the ion feature disappears entirely, leaving

> `S(k, ω) ∝ (2π/k) · f_e(ω/k)`

Comparing the pipeline's spectrum against the conditioned electron VDF pushed
through that map, over all 51 timesteps:

|                                  |            |
| -------------------------------- | ---------- |
| normalised L1 difference, median | **0.0000** |
| 90th percentile                  | **0.0009** |

Essentially exact. This is a stronger statement than the OSIRIS Bohm-Gross
check: there the agreement was 2.3% against an approximate dispersion
relation, here the non-collective limit is exact and the pipeline reproduces
it to the noise floor.

Getting it required using the *exact* `k(λ)` the forward model uses,
`k = |k_s - k_0|` with both wavenumbers carrying the
`√(ω² - ω_pe²)` correction. A constant-`k` Doppler map gives L1 = 0.226 —
over a 40–1024 nm window `k` varies by more than an order of magnitude, so the
textbook `Δλ = λ₀ · 2(v/c) sin(θ/2)` is not usable here. That was the check
being wrong, not the pipeline.

**The reader is confirmed by the physics too.** The phase-space figure shows
`amb_ions` with the bifurcated incoming/reflected structure at x ≈ 40–48 m
that is the supercritical shock signature the paper is about, `piston_ions`
in free expansion, and `amb_electrons` swept into a thin compressed sheet
ahead of the piston. Population fractions hand over cleanly from ambient to
piston at t ≈ 830 ns, ions marginally before electrons.

### 6a. Cross-code comparison: what is and is not possible

|                            | R1_paper (WarpX)                        | omegashock_w3.5e11_exp (OSIRIS) |
| -------------------------- | --------------------------------------- | ------------------------------- |
| `n_e` at the sampled point | 7.9e15 – 2.0e17 m⁻³                     | 9.3e23 – 3.5e25 m⁻³             |
| α at 532 nm, 90°           | 7.7e-6 – 3.2e-5                         | 1.9 – 15.4                      |
| regime                     | non-collective                          | collective                      |
| validation used            | exact non-collective limit (L1 = 0.000) | Bohm-Gross shift (2.3%)         |

A *quantitative* cross-code comparison of spectra is not meaningful between these
two runs: they are eight orders of magnitude apart in density and sit on opposite
sides of α = 1. What the exercise does establish is the structural claim the whole
design rests on — **the same `spectra_from_phase_spaces` consumed both codes with
no code-specific handling**, and was independently validated in each regime.

For a collective WarpX comparison you would need either a much denser run, or a
much longer probe wavelength: α scales with λ_probe, so a 10.6 µm CO₂ probe buys a
factor of 20, reaching only ~6e-4 — still non-collective. Density is the binding
constraint.

### 6b. A collective view of the same run

To see what Thomson structure this plasma *would* show, the run was repeated with
a **30 mm (10 GHz) microwave probe** — `media/09_warpx_spectra_collective.png`.
That is the band this density calls for: λ_De is 2.3–9.8 mm, so α ~ 1 needs a
probe of a few cm, and f_pe peaks at 4.0 GHz so 10 GHz still propagates
comfortably. α comes out **0.51 – 2.27**, and the spectrogram shows a proper
collective feature that Doppler-shifts blue to ~25 mm once the flowing piston
plasma reaches the sampling point.

Two things this exercise established:

- **The non-collective check correctly stops holding.** Its L1 rises from 0.0000
  at 532 nm to 0.148 at 30 mm, which is the intended behaviour: shielding is now
  present and the spectrum is no longer just the Doppler-mapped VDF. The
  agreement at 532 nm is therefore a real test, not a tautology.

- **The collective regime needs far more smoothing than the non-collective one.**
  At the light setting used for 532 nm (`smoothing_window = 9`) the collective
  spectrogram is speckled with sharp spurious pixels. Widening to 41 removes them
  entirely:

  | `smoothing_window` | non-collective-check L1 | spectrogram          |
  | ------------------ | ----------------------- | -------------------- |
  | 9                  | 0.360                   | heavily speckled     |
  | 41                 | 0.148                   | clean                |
  | 81                 | 0.072                   | clean, over-smoothed |

  The cause is physical, not a defect: for α > 1 the spectrum carries
  `|1 - χ_e/ε|²`, which diverges as `ε → 0` at the electron-plasma-wave
  resonance. Shot noise in the VDF enters χ through `df/du`, so the resonance
  amplifies it into speckle. At α ≪ 1 the same noise passes through untouched.
  **Anyone running this pipeline in the collective regime should smooth harder
  than the non-collective case suggests.**

There is also a hard ceiling on α for this run, worth knowing before chasing a
better probe. Combining `α = ω_pe/(k σ_e)` with the propagation requirement
`ω_probe > ω_pe` gives `α_max = c / (2 sin(θ/2) σ_e)`. The piston electrons reach
σ_e ≈ 0.54 c in simulation units, capping α at ≈ 1.3 there no matter what probe
is used; only the cold ambient (σ_e ≈ 0.045 c) admits strongly collective
scattering. Relativistic electrons and collective Thomson scattering are close to
mutually exclusive.

Note also that the WarpX spectra are shown in the simulation's **own velocity
units**; `--velocity-scale-factor` is left off by default because the right
convention for this run is unsettled, and it matters a great deal — the electron
distribution reaches 0.54 c in sim units, giving a 1σ Doppler width of 403 nm.
See §5.1: the deck's `mass_ratio = 100` against a real 1836 suggests R = 18.36,
while the paper's Table I reports `c_sim/c_phys = 0.02`, a factor of 50. These
disagree and should be reconciled before the WarpX spectra are read as physical.
7\. ✅ **Done.** Instrument response, HDF5 output, plotting. 27 new tests
(267 in `tests/diagnostics/` overall, all passing); `ruff` and `ty` clean.

- **`ThomsonSpectrogram.apply_instrument_response`** degrades a synthetic
  spectrogram to a real instrument's resolution, taking FWHM values in
  physical units — `time_fwhm=100*u.ps`, `epw_wavelength_fwhm=0.5*u.nm`,
  `iaw_wavelength_fwhm=0.05*u.nm` — with a Gaussian or boxcar kernel. The two
  wavelength windows take separate widths because their dispersions differ by
  an order of magnitude.

  Because `ThomsonSpectrogram` carries `t` in seconds and wavelengths in
  metres, the conversion to bins is direct. The old `smooth_spectra` module
  had to reconstruct it from `ω_p` and the simulation timestep, which is where
  its hard-coded `dt_sim = 20` came from — a footgun the SI contract removes.

  Gaps are handled properly. A vacuum timestep is `NaN`, and a plain filter
  spreads that over every point the kernel reaches; filtering values and the
  validity mask separately and dividing gives the average over valid samples
  alone. A gap narrower than the instrument function is then *filled* from
  either side — which is what an instrument integrating over a finite gate
  really does — while a gap wider than the kernel's reach stays `NaN`. Both
  branches are tested, as is the contrast with the naive filter.

- **`to_hdf5` / `from_hdf5`** round-trip the spectrogram. Every dataset carries
  a `UNITS` attribute, the α convention is recorded on the dataset, and the
  per-row area normalisation is written into a root attribute so it cannot be
  forgotten by whoever reads the file later. Verified round-tripping the real
  129-dump OSIRIS spectrogram.

  Group names follow the `osiris2thomson` layout where the contents line up
  (`SPECTRA/EPW/epws`, `SPECTRA/IAW/iaws`, `SPECTRA/SCATTERING_PARAMETERS`,
  `DENSITY/dens`, `AXES`), but this is **not** a drop-in for that pipeline's
  files: `load_spectra` requires `TEMPERATURE`, `FLOW_VELOCITY` and `VDF`
  groups this module deliberately does not compute, and the axes here are SI
  rather than simulation units. Population fractions are stored as 2-D arrays
  plus a label dataset rather than as `ion_fraction_<name>` datasets, which
  avoids mangling labels like `"Al 13+"`.

- **`plot`** draws the spectrogram, density, α and population fractions.
  Repeated species labels are numbered, since two populations of the same
  element are legitimate but two identical legend entries are not.

Both are wired into `tools/pic_thomson_osiris_comparison.py` via `--output`,
`--instrument-time-fwhm` and `--instrument-wavelength-fwhm`; the degraded
result is `media/10_osiris_instrument_response.png`.

### 7a. Decision: no width-relative taper bound

§3a suggested a taper bound expressed as a fraction of each slice's own width
rather than a fixed bin count. Not implemented, deliberately. A fixed count
proved perfectly workable once calibrated per species (20 for electrons, 3 for
the ions of the OmegaShock run, 20 for WarpX), and the `pedestal_warning` catches
a bad choice with a number that says how bad. A second, width-relative
calibration path would add an API mode to document and test for a benefit that
the measurements do not show.
8\. ✅ **Done.** Docs, changelog, lint, types, full test suite.

| check                      | result                                                |
| -------------------------- | ----------------------------------------------------- |
| full test suite (`tests/`) | **4626 passed**, 6 skipped, 6 xfailed, 1 xpassed      |
| `pre-commit` (all hooks)   | clean                                                 |
| `ty check src/plasmapy/`   | 18 diagnostics, **none** in `pic_thomson.py`          |
| public API audit           | 13 names, all exported, all documented, nothing stray |

- **Docs page** at `docs/ad/diagnostics/pic_thomson.rst`, registered in the
  diagnostics toctree next to `thomson`. It follows that page's structure
  (`currentmodule` + `automodapi`) and adds a worked example plus the four things
  a user has to know that an API reference will not tell them: how each reader
  undoes its code's units, why the taper has to be bounded, why the collective
  regime needs more smoothing, and what the spectra do *not* carry (absolute
  intensity, EPW-to-IAW ratio, the conventional α).
- **Changelog** at `changelog/7.feature.rst`, maintained across every step.
- `uv.lock` regenerated for the new `pic` extra.

Two tooling snags worth recording, since they will recur:

- `typos` silently rewrote `PNGs` to `ONGs` inside a Python docstring — it parses
  identifiers in `.py` files differently from prose. `extend-words` does not
  suppress it; the fix is `[default.extend-identifiers]` in `_typos.toml`.
  **A hook that rewrites text needs its diff read, not just its exit code.**
- `blacken-docs` fails on this plan's Python blocks, which are signature sketches
  rather than runnable code. Those are retagged as plain text blocks.

______________________________________________________________________

## 9. Where this leaves the pipeline

Complete and validated on both codes. What a reader should know before trusting a
number out of it:

**Validated.** Analytic Maxwellian vs `thomson.spectral_density` (L1 = 0.032);
OSIRIS EPW satellite vs Bohm-Gross over 513 dumps (2.3%); WarpX non-collective
limit over 51 frames (L1 = 0.0000); OSIRIS reader density against the deck
(n/n₀ = 1.037); WarpX reader density against the deck (0.006%).

**Still open.**

1. ~~The `cham` and `targ` species~~ — resolved (§4b): both are fully stripped
   carbon, `C 6+`, and every figure and end-to-end run has been regenerated with
   it. A minor loose end remains: `C 6+` implies `rqm_factor ≈ 53` where
   `run.yaml` says 50, a 3% shift in `velocity_scale_factor`.
1. **The WarpX velocity-scaling convention (§6a).** `mass_ratio = 100` against a
   real 1836 suggests R = 18.36, while the paper's Table I reports
   `c_sim/c_phys = 0.02`, a factor of 50. These disagree by 2.7×, and the WarpX
   spectra are currently shown in simulation units because of it.

## 10. Multi-dimensional simulations

Added after the 1-D pipeline was validated. The chosen design was **reduce to the
sampling point at read time**, with a **slab** default for the other directions.

A useful discovery made the change far cheaper than expected: the whole
conditioning core was *already* dimension-agnostic. `smooth_vdf`,
`taper_vdf_edges`, `normalize_vdf`, `number_density` and `rescale_velocity_axis`
all handle `(n_time, n_v, n_x, n_y)` today, because each was written to act along
the velocity axis and treat everything else as columns. Only `from_arrays`
validation and the readers needed work.

**Both readers** gained `transverse_reduction` (`"slab"` or `"chord"`),
`transverse_position`, `slab_halfwidth` and `position`. The transverse directions
are always reduced; `position` additionally reduces the diagnostic axis, so the
reader returns the single point the probe looks at.

- **Reduction averages, it does not sum.** Summing turns a density into a line
  integral and scales it by the number of cells combined — which is what
  `osiris2thomson` did for its 3-D phase spaces, quietly multiplying the density
  handed to the forward model. Verified on both readers: slab and chord give
  *identical* densities despite transverse areas differing by 64×.
- **`read_warpx_phase_space` gained `scatter_direction`.** The quantity a Thomson
  diagnostic measures is the velocity along `k̂`, and a run resolving more than
  one velocity component can now be projected onto it properly, with the Lorentz
  factor taken from the **full** momentum. Naming a single momentum component
  only happens to be right when `k̂` lies along that axis. Position and momentum
  fields are auto-detected, which also removes the need for the caller to know
  that a 1-D WarpX run stores its only coordinate as `particle_position_x`.
- OSIRIS phase spaces are projections fixed at write time, so no reprojection is
  possible there; the module says so rather than pretending otherwise.

Regression-checked: the OSIRIS reader returns **bit-identical** output to the
committed version on the 1-D production run, and the Bohm-Gross validation is
unchanged at 1.292 against 1.262.

### 10a. The WarpX reference run was replaced mid-project

`KinShock2020/runs/R1_paper` was re-run on 2026-08-02 at 11:55 with
`n0 = 6e26` m⁻³ in place of `1e18`, shrinking the domain from 47.8 m to
0.00195 m. **Every WarpX number recorded in §5 and §6 above describes the
previous version of that run.** The reader still validates against the new deck —
measured `4.7997e24` m⁻³ against `namb = 0.008 × 6e26 = 4.8e24`, 0.006% — but the
regime is entirely different:

|                      | old R1_paper | current R1_paper |
| -------------------- | ------------ | ---------------- |
| `n0`                 | 1e18 m⁻³     | 6e26 m⁻³         |
| domain               | 47.8 m       | 0.00195 m        |
| α at 532 nm          | ~1e-5        | **0.21 – 0.42**  |
| non-collective check | L1 = 0.0000  | L1 = 0.0305      |

That the check degrades smoothly with α — 0.0000 at 1e-5, 0.031 at ~0.3, 0.148 at
~1.1, 0.417 at ~1.6 — is a good sign: it is measuring the strength of collective
effects, not passing vacuously. The hard-coded `--position 30.0` in the WarpX tool
went stale with the resize and now defaults to the domain centre.

### 10b. Results on the current WarpX runs

Run at the piston's path rather than the domain centre, which at these densities
and times is still undisturbed ambient and shows nothing.

**`R1_paper`** (still running, 29 of an eventual ~58 frames), probe at x = 0.75 mm:
the piston front sweeps 0.006 → 1.043 mm over 29.4 ps and crosses the probe at
about 20 ps. The spectrogram shows the whole sequence — quiet ambient, an abrupt
broadening at ~15 ps as the compression arrives ahead of the material, **piston
ions arriving ~1.5 ps before piston electrons**, then a distinct blue-shifted
feature near 450 nm from the flowing piston plasma. Density climbs 4.8e24 →
1.05e26 m⁻³ (22×) and α from 0.24 to 0.76.

**`R1_paper_dial` vs `R1_paper_phys`** — a matched pair differing *only* in the
Coulomb logarithm (1.22e5 vs 10.84), so collisionality is the single variable.
Sampled at x = 0.111 mm, last frame:

|                            | dial (collisional) | phys (collisionless) |
| -------------------------- | ------------------ | -------------------- |
| `n_e`                      | 3.49e25 m⁻³        | 1.09e25 m⁻³          |
| α                          | 0.317              | 0.142                |
| piston electron fraction   | **0.74**           | **0.00**             |
| spectrum centroid          | 540.4 nm           | 571.0 nm             |
| spectrum rms width         | 159.7 nm           | 181.0 nm             |
| L1 between the two spectra | **0.239**          |                      |

The synthetic diagnostic separates them clearly. The physical difference it is
seeing: with strong collisions the ablated electrons and ions arrive together as
a coupled fluid, while collisionlessly the piston *ions* run ahead and the piston
electrons have not reached the probe at all. A real Thomson measurement would
distinguish these, which is the point of building the diagnostic.

______________________________________________________________________

**Recommendation.** Spectra produced by the `osiris2thomson` pipeline should be
regenerated. Between the ion-normalisation bug (§5.2, a factor of 10 on `χ_i`),
the unbounded taper (§3a, `vTe` inflated 67%, ion widths by up to 40×) and the
missing Jacobian (§3), the differences are not refinements: on the OmegaShock run
the legacy and corrected spectrograms differ by a median L1 of 0.90, against
0.032 for the analytic-Maxwellian agreement.

______________________________________________________________________

## 11. Reading phase space a code binned itself, and species carried as moments

Two gaps the first ten sections left open, resolved here. Both were researched
before any code was written; the findings are recorded because several of them
contradict what the documentation says.

### 11.1 WarpX does have a phase-space diagnostic

`ParticleHistogram2D` (`Source/Diagnostics/ReducedDiags/ParticleHistogram2D.cpp`)
is a genuine analogue of OSIRIS `p1x1`, and in one respect it is better than
either existing reader: **both axes are arbitrary parser expressions** of
`(t, x, y, z, ux, uy, uz, w)`, with an optional `filter_function` and
`value_function`. So the projection this pipeline needs,

```text
histogram_function_ord = (kx*ux + ky*uy + kz*uz) / sqrt(1 + ux*ux + uy*uy + uz*uz)
```

is evaluated **per particle, inside the code**. `ux` is `PIdx::ux / PhysConst::c`
= γv_x/c (`ParticleHistogram2D.cpp:217`), so dividing by γ gives v·k̂/c exactly,
with γ from the *full* momentum. That is strictly better than
`read_warpx_phase_space`'s post-hoc projection and far better than the OSIRIS
reader, which can only invert `v = uc/√(1+u²)` on a single stored component — an
approximation valid only when the other components vanish.

Output is openPMD, one file per iteration, mesh `"data"`, shape `(n_ord, n_abs)`,
with `axisLabels`, `gridGlobalOffset` = bin minima, `gridSpacing` = bin sizes,
`position = [0.5, 0.5]`, and `setTime`. Every one of those was read back off a
file produced by a real run, and the test fixture reproduces that layout exactly.

Four practical limits:

1. **It cannot be applied retroactively.** No run in `KinShock2020`,
   `H-PICShock`, `LaserProdShock`, or `MagShockZ` declares one.
1. **Bins are frozen at deck time.** Particles outside are discarded before
   anything is written and nothing records how many, so unlike
   `read_warpx_phase_space` the reader cannot warn about a clipped tail.
1. **The default backend is `bp5`** wherever ADIOS2 is compiled in.
   `histogram2d_deck_block` asks for `h5`, which keeps the output readable by
   `h5py` alone — the property that makes the OSIRIS path dependency-light.
   Note that the `warpx-cda/build` tree has openPMD compiled with *neither*
   HDF5 nor ADIOS2, so only `json` works there; that is a build-configuration
   issue, not a code one.
1. **It does not know its own volume.** The bins hold Σw over whatever the
   filter kept, so the conversion to a density is post-processing.
   `histogram2d_deck_block` returns the transverse area it selected so the two
   sides cannot drift apart.

### 11.1a A WarpX bug: `ParticleHistogram2D` needs `value_function` spelled out

Found while validating the generated deck block against a real run.

The constructor reads `value_function` into `m_do_parser_value`
(`ParticleHistogram2D.cpp:121`) and **never uses that flag again** — `grep` finds
it only at its declaration and its assignment. The kernel calls
`fun_valueparser(...)` unconditionally (`:231`), and `compileParser` returns a
default-constructed `amrex::ParserExecutor<N>{}` when the parser is null
(`ParserUtils.H:80-87`). Compare the filter, which *is* guarded by
`if (do_parser_filter)`.

The consequence is not an error. Measured on a 64×128 histogram of 12 800
macroparticles:

| `value_function`             | bad bins            | Σ over finite bins | expected  |
| ---------------------------- | ------------------- | ------------------ | --------- |
| absent                       | 2 758 / 8 192 (34%) | 0                  | 1.0000e21 |
| `w` (the documented default) | 0                   | **1.0000e21**      | 1.0000e21 |

Absent, a third of the bins hold `NaN` and `DBL_MAX` and every finite bin is
zero. A 1-D `ParticleHistogram` on the same run returns 1.0000e21 exactly, so the
particles are there and it is the 2-D path that is broken. Not a threading
artifact — `OMP_NUM_THREADS=1` reproduces it — and not a layout artifact, since a
square histogram reproduces it too.

`histogram2d_deck_block` therefore always emits `value_function = w`, with the
reason in a comment. **Worth reporting upstream**; the fix is to guard the call
the way the filter is guarded.

### 11.1b Validation of `read_openpmd_phase_space`

A 1-D WarpX run of 12 800 electrons, drifting Maxwellian at `uz_m = 0.05`,
`uz_th = 0.02`, uniform at 1e24 m⁻³, read through the generated block:

| quantity   | recovered     | expected  | error  |
| ---------- | ------------- | --------- | ------ |
| density    | 9.9997e23 m⁻³ | 1e24      | 0.003% |
| drift      | 1.49492e7 m/s | 1.49694e7 | 0.13%  |
| rms spread | 5.9507e6 m/s  | 5.9959e6  | 0.75%  |

The spread error is one bin width (6.25e5 m/s) of discretisation. Later frames
drift because an electron-only plasma with no neutralising background decelerates
in its own space charge — physics, not the reader.

### 11.2 Hybrid PIC: the electrons exist only as moments

WarpX's kinetic-ion / fluid-electron solver has **no electron macroparticles**.
Quasineutrality and the absence of displacement current collapse the electron
momentum equation into Ohm's law, and the electron state is three mesh fields.
The `H-PICShock` decks already dump all of them at 102 frames
(`diag_fields.fields_to_plot = Ez Ey By Bx jz rho rho_piston_ions rho_amb_ions Te Pe`):

| moment | source                                     | exactness                                                     |
| ------ | ------------------------------------------ | ------------------------------------------------------------- |
| `n_e`  | `rho / (Z̄ e)`                              | **exact** — quasineutrality is how the solver defines it      |
| `T_e`  | the `Te` field [eV]                        | solved, since this fork has `electron_energy_mode = advected` |
| `u_e`  | `J_e = ∇×B/μ₀ − J_i`, `u_e = −J_e/(e n_e)` | exact                                                         |

`jz` is `current_fp`, the **ion** current — in hybrid the electrons deposit
nothing (`HybridPICModel.cpp:180`), and the total plasma current lives separately
in `hybrid_current_fp_plasma = ∇×B/μ₀`.

The 1-D case is worth stating because it is both exact and free: with only
`∂/∂z` surviving, `(∇×B)_z ≡ 0`, so the total axial current vanishes, the
electrons exactly counterstream the ions, and

```text
u_ez = j_z / (e n_e)
```

with **no magnetic field read at all**. `_curl_needs` works this out from the
resolved directions, so the reader asks for `Bx`/`By` only when the geometry can
actually make a current along k̂.

Measured on `H3_470eV_eheat_coll` at frame 440: `n_e` 1.36e24–1.29e27 m⁻³,
`T_e` 9.3–1175 eV, `u_e` up to 2.2e6 m/s = **0.155 v_th,e** — a real but
subthermal Doppler shift, which is exactly the regime Thomson resolves. End to
end over 11 frames at x = 0.6 L, α ran from **0.30 to 5.14** as the shock crossed
the probe: the diagnostic passes from non-collective to strongly collective
within one run.

A drifting Maxwellian is **inherited, not assumed**. The closure carries one
scalar temperature and no higher moments, so there is nothing in the data from
which to build any other shape. What that costs is a question about the model,
answerable by reconstructing the moments of the `KinShock2020` kinetic twin and
comparing spectra — not attempted here.

One trap worth recording: these reconstructed distributions must be conditioned
with `taper_threshold=None`. The taper exists to replace the discontinuity where
macroparticle shot noise meets the grid edge; an analytic Maxwellian has neither,
so tapering it only fabricates a pedestal and widens the feature (measured: 11%
on the second moment, enough to trip the §3a pedestal warning).

### 11.3 Robustness

Two outright bugs, both fixed:

- **The WarpX cache signature omitted `transverse_area` and `timesteps`.** Both
  change `f` — area scales it linearly — so a cache built from a five-frame test
  read was handed back, silently, for the full series.
- **`momentum_field` defaulted to `particle_momentum_z` in any dimension.** In
  1-D and 2-D the field is missing and the reader raised; in 3-D it exists and
  the projection was quietly onto ẑ regardless of where k̂ pointed. `"auto"` now
  resolves only in 1-D, where there is nothing to guess, and refuses otherwise.

Two hardening measures, both of which found something real:

- **Cross-check against the code's own `rho`.** Reading `diag_phase` from
  `H3_470eV_eheat_coll` gives a **ratio of 0.200092** against the deposited
  charge — the deck's `diag_phase.amb_ions.random_fraction = 0.2`, which
  subsamples the particle output *without* reweighting. Every density built from
  that diagnostic is 5× low and nothing in the file says so; α would have been
  wrong by √5. Because particles and fields live in different diagnostics here
  (`write_species = 0` on one, `fields_to_plot = none` on the other), the check
  takes a `density_reference` prefix and matches by step number.
- **Velocity axis sized from every frame**, not the first and last, plus a count
  of what the histogram discarded. For a shock, what happens between the first
  and last frames *is* the measurement. `velocity_scan="ends"` restores the old
  behaviour where the second read pass is too expensive.

And one ergonomic fix: every reader now records its density scale in
`PICPhaseSpace.meta`, so the driver's `reference_density` defaults correctly per
reader rather than the caller having to remember that OSIRIS means `n₀` and
WarpX means 1 m⁻³. Species that disagree are an error, since one scale multiplies
all of them.

### 11.4 Still open

- The velocity-scaling convention of §6a/§9 is unchanged: `mass_ratio = 100`
  against a real 1836 implies R = 18.36, while the paper's Table I reports
  `c_sim/c_phys = 0.02`, a factor of 50.
- `ParticleHistogram2D`'s `value_function` bug (§11.1a) should go upstream.
- No run yet declares a `ParticleHistogram2D` block, so
  `read_openpmd_phase_space` has been validated only against a purpose-built test
  run, not against a production shock.
- The hybrid-vs-kinetic comparison the moment reconstruction makes possible
  (§11.2) has not been done.

## 12. What the EPW panel was actually showing

Three separate defects had been running together in the electron channel of
`tools/pic_thomson_osiris_comparison.py`, all diagnosed on
`omegashock_w3.5e11_exp` at x = 5 mm. The evidence figure is
`media/11_epw_artifact_diagnosis.png`.

### 12.1 The notch edge (not fixed here)

The notch was fixed at `[530, 534]` nm while the central feature is Doppler
shifted by the flow: `λ²/(2πc)·k·u` is 3.7 nm at 2×10⁶ m/s, most of a 4 nm
notch. Measured at the probe, the two bins flanking the notch carry **0.5% of
the EPW window's area before the piston arrives and 88–98% after it**, so once
each row is normalised to unit area the satellites are crushed into the
remaining percent. That is the bright line at ~530 nm in the figure the whole
investigation started from.

It is made worse by resolution. The EPW grid is 0.4 nm and the central feature
is ~0.1 nm, so the surviving flank value is a point sample of a near-singular
function: shifting the grid by 0.1 nm — no physics change — moves the area
outside 528–536 nm by **27×**, and refining 500 → 32000 bins raises the flank
peak from 2.5e9 to 9.4e10 without converging.

The fix is to derive the mask from the measured extent of the central feature
per frame rather than hard-coding it. **Not done.**

### 12.2 The notch's own arithmetic (fixed)

All four notch code paths in `thomson.py` located the endpoints with `argmin`
and zeroed `[x0:x1]`. `argmin` rounds each edge to the nearest bin centre and
the half-open slice drops the upper endpoint bin, so a requested `[530, 534]`
was realised as `[530.196, 533.403]` — 0.6 nm short on the red side, where the
feature's skirt is brightest. Now a boolean mask on the closed interval.

### 12.3 A smoothing window measured in bins (fixed)

`smoothing_window=40` over three passes, on a distribution occupying **31 of the
1024 bins** OSIRIS's `p1x1` diagnostic spans. A boxcar is a convolution, so it
adds `n(W²−1)Δv²/12` to the second moment of every slice whatever its own width:

|                      | T_e raw     | T_e smoothed | α raw    | α reported |
| -------------------- | ----------- | ------------ | -------- | ---------- |
| ambient, t < 0.33 ns | **38.8 eV** | 813 eV       | **12.0** | 3.2        |
| shocked, t = 0.42 ns | 1858 eV     | 2598 eV      | 3.2      | 2.7        |

So the published α panel was low by ~3.7× in the ambient, and the bias runs from
21× to 1.2× across the run — it does not cancel between two measurements. Median
occupancy over the whole run is 37 bins, i.e. the window was wider than the
distribution nearly everywhere.

`condition_phase_space` now takes `smoothing_width` in thermal speeds and sizes
the window from the narrowest populated slice; `smooth_vdf` gained a
`variance_warning`. Rerunning the comparison moves the median α from **1.69 to
13.99**. `max_taper_width` does the same for the taper rolloff, which is the
same defect in the sibling knob — on H-PICShock's ambient ions the old
`smoothing_window=16` + `max_taper_bins=8` inflated T_i by 45%, the width forms
by 3%.

### 12.4 A satellite with no electrons behind it (fixed)

The deepest one. The satellite reads `f_e` at `√(α²+3)` thermal speeds; the
histogram is populated only as far as its last macroparticle, which for this run
is 3.5–4.8 σ everywhere. So:

| t (ns)    | α         | needs (σ) | has (σ) |
| --------- | --------- | --------- | ------- |
| 0.00–0.33 | 12.0      | 12.1      | 3.5     |
| 0.39–0.48 | 1.9–3.2   | 2.6–3.7   | 3.7–4.0 |
| 0.51–0.64 | 13.9–17.0 | 14.0–17.1 | 3.9–4.8 |

**Only 4 of 23 sampled frames put the resonance inside sampled data.** Everywhere
else the satellite is the taper's cosine rolloff and the 1e-30 floor, amplified
by the ε resonance. Confirmed by construction: with `max_taper_bins` at 5, 10,
20, 40, 80 the apparent satellite moves 510.96 → 508.95 nm, converging only once
the rolloff extends past the resonance.

This supersedes the empirical "α ≲ 9 ceiling" of `RESULTS.md`. The ceiling is not
a constant — it is `v_max/v_th` of the code's dump — and it cannot be bought with
particles, since reaching `nσ` needs `~e^{n²/2}` per cell. `e^{72}` for α = 12.

The driver now records `epw_tail_ratio`, `epw_tail_required` and `epw_resolved`,
and masks the EPW where the check fails. On this run that is 53 of 65 frames; on
the 12 that survive the satellite tracks the plasma-frequency shift to a median
of **0.997**.

Reconstructed populations are a separate case: `from_moments` has no last
macroparticle, so the limit is `velocity_headroom`, whose default of 6 covers
only α ≲ 5.7. H-PICShock's H3 run reaches α = 11.6, so its electron grid was
stopping *before* the resonance — `read_warpx_hybrid_electrons` now takes
`velocity_headroom` and `scripts/thomson.py` sizes it from the measured α.

### 12.5 Still open from §12

- §12.1, the notch mask, is the remaining defect in the EPW channel.
- Whether the α-ceiling entries in `RESULTS.md` survive re-measurement now that
  the H3 electron grid reaches the resonance and α is no longer smoothing-biased.
  The H3 and L2 spectrograms in `media/` predate all of §12 and should be
  regenerated before being quoted.
- An analytic tail beyond the sampled support, in place of taper-then-floor,
  would let the EPW be modelled where it cannot be measured — with the
  assumption stated rather than hidden.

## 13. The taper, replaced

§12 fixed the taper's *extent* (`max_taper_width`) without touching what it was
doing. That was the wrong end of the problem.

### 13.1 The taper was setting the answer, not perturbing it

Measured on `omegashock_w3.5e11_exp` at x = 5 mm, sweeping `max_taper_bins`
alone on identical data:

| `max_taper_bins` | α     | blue satellite |
| ---------------- | ----- | -------------- |
| 5                | 16.00 | 523.38 nm      |
| 20               | 13.47 | 527.39 nm      |
| 80               | 5.28  | 524.18 nm      |
| unbounded        | 0.76  | 524.99 nm      |

**α moves by a factor of 21.** The rolloff leaves a pedestal at large `|v|`,
which is where the `v²` weighting of the second moment lives, so a numerical
knob was setting a reported plasma parameter. The satellite moves with it.

### 13.2 Why the method was wrong

- **The threshold is a fraction of the peak, not a particle count.** For a
  Maxwellian `0.005 × peak` is always 3.26 σ whatever the particle budget. Here
  that put the anchor at 3.24 σ while ≥10 macroparticles only reached 2.60 σ —
  the entire tail hung off a bin holding one or two particles, and that shot
  noise went straight into α and into the satellite amplitude.
- **The shape is wrong in the derivative.** Landau damping reads `∂f/∂v` at the
  resonance, so a half-cosine fabricates the feature's width as well as its
  height.
- **It goes to zero**, so the amplitude is set by where you chose to stop.

### 13.3 `extend_vdf_tail`

Join from counts (the smallest positive raw bin is one macroparticle; join at
the outermost bin with ≥ `min_counts`), fit `ln f` against `(v−v̄)²` over the
resolved band with counts as inverse-variance weights, continue with the fit
beyond the join at **its own intercept** — the join is by construction the
outermost bin still reaching `min_counts`, so anchoring there biases the tail
high by 20–30%, measured. Fall back to the core temperature when the band is too
thin or the fit does not decay.

Validation on a sampled Maxwellian (fitted tail width / true, and the
extrapolated `f` against truth):

| particles | join   | fitted width | f(5σ) | f(8σ) | f(12σ) |
| --------- | ------ | ------------ | ----- | ----- | ------ |
| 1e4       | 2.60 σ | 1.0147       | 1.65  | 3.61  | 16.9   |
| 1e5       | 3.32 σ | 0.9991       | 1.009 | 1.006 | 0.994  |
| 1e6       | 3.99 σ | 0.9991       | 0.972 | 0.922 | 0.829  |

At 1e5 particles the extrapolation is right to 0.6% **three times beyond the
last particle**. The 1e4 row is not a failure of the method but of the data, and
`tail_width_error` says so: ±2.10% on the width predicts a factor of 20.5 at
12 σ, against 16.9 actually observed. A 5% suprathermal component at 2.5× the
core temperature comes back as a tail/core width of 1.221 rather than 1.000, so
a resolved non-Maxwellian tail survives.

The floor goes with the taper: a Maxwellian crosses 1e-30 at about 12 thermal
speeds, so a floor would put a flat pedestal exactly where the satellite reads
at α ~ 12.

### 13.4 What this changes downstream

`epw_tail_uncertainty` replaces blanket masking. On the OSIRIS run the resonance
is at 17.4 σ against data reaching 3.4 σ, and the fitted tail still holds the
amplitude to a factor of **3.2** — so the rows are kept and priced rather than
NaN'd. `mask_unresolved_epw` now defaults to `False`, and is what you want only
with `tail_model=None`.

### 13.5 Still open

- **§12.1, the notch mask, is now the only defect left in that EPW panel.** With
  the tail modelled and α stable, what remains at ~530 nm is the central
  feature's skirt escaping a hard-coded `[530, 534]`.
- The velocity-scaling convention. The deck has electrons at `rqm = -1.0` and
  `uth_bnd(1:3,2,1) = 8.766e-03`, which read as `γv/c` is 39.3 eV and matches
  the raw histogram's 38.8 eV to 1%; applying `velocity_scale_factor = 50` to
  them makes the mapped `T_e` 0.79 eV and α ≈ 12 rather than 1.7. Confirmed with
  the run's owner that the factor **does** apply to the electrons here, so the
  pipeline keeps doing that; the reconciliation with the deck is unresolved and
  matters, because α ≈ 12 vs ≈ 1.7 decides whether the EPW channel is a
  measurement or an extrapolation.

## 14. Why the EPW panel is empty at high alpha, and it is not the notch

Chasing the leftover band at 530 nm turned up the actual reason the electron
channel of `07_osiris_end_to_end.png` carries nothing.

### 14.1 The satellite is there, and enormous

At x = 5 mm, on a grid refined around the Bohm--Gross root:

| step | Bohm--Gross | 0.4 nm production grid                 | refined grid               |
| ---- | ----------- | -------------------------------------- | -------------------------- |
| 0    | 523.85 nm   | peak 1.95e9 at **531.80** (notch edge) | peak 3.01e14 at **523.65** |
| 55   | 488.90 nm   | peak 1.12e9 at **530.20** (notch edge) | peak 2.63e14 at **489.16** |

Position agrees with Bohm--Gross to 0.2 nm. The production grid never sees it:
the FWHM is under 3 pm, and refining 100x raises the peak 160x, so it is a pole,
not a resolved feature.

### 14.2 Two separate limits, both from an undamped wave

- **Resolution.** The EPW resonance width goes as the Landau damping,
  `exp(-alpha^2/2)`. Past alpha ~ 3-4 no practical uniform grid resolves it.
  Bin-averaging does not help: 256x oversampling (0.0016 nm sub-bins) still
  steps over it, and the blue wing carries 0.00-0.14% of the area.

- **Precision.** `Re(eps)` is computed as `1 + chiE + chiI` with `chiE ~ alpha^2`,
  so its floating-point floor is `|chiE| * 2.2e-16`, flat at **2.2e-16**. The
  physical `Im(eps)` at the root falls away underneath it:

  | alpha     | 8       | 9       | 10          | 12      |
  | --------- | ------- | ------- | ----------- | ------- |
  | `Im(eps)` | 4.5e-11 | 5.0e-14 | **2.7e-17** | 9.6e-25 |

  They cross at **alpha ~ 9.5**. Above it `|1 - chiE/eps|^2` plateaus at
  1e13-1e14 regardless of alpha -- rounding noise, not physics.

**This is the mechanism behind the empirical "alpha \<~ 9 ceiling" in
`H-PICShock/RESULTS.md`.** That ceiling was real. It is not a physics ceiling,
not the taper, and not the tail model: it is double precision losing Landau
damping in the dielectric function.

### 14.3 What this run can and cannot show

alpha runs 12-24 at this position under the run's scaling convention, so the
electron channel is above both limits nearly everywhere. The one window where it
works is t = 0.35-0.50 ns, where alpha dips to 2.5-5, and the corrected panel
does show structure there. Probe wavelength scales alpha directly:

| probe  | step 0                     | step 40                | step 55                |
| ------ | -------------------------- | ---------------------- | ---------------------- |
| 532 nm | alpha 11.6, wing 0.00%     | alpha 3.40, wing 5.00% | alpha 16.9, wing 0.00% |
| 266 nm | alpha 5.81, wing 0.03%     | alpha 1.70, wing 6.62% | alpha 8.42, wing 0.00% |
| 133 nm | alpha 2.90, wing **33.9%** | alpha 0.85, wing 10.4% | alpha 4.21, wing 0.78% |

### 14.4 The notch, fixed anyway

`epw_notches="auto"` now sizes the mask from the central feature measured in the
IAW window each frame. It picks 0.5 nm at t = 0 and 25 nm at t = 0.40, which is
right, and it is a genuine improvement -- but it cannot rescue a run sitting
above the resolution and precision limits, and it does not on this one.

### 14.5 The way out

Both limits come from modelling a wave with no damping. Either would remove
them, and both are physical:

- **Collisional damping in `eps`.** At alpha >~ 10 Landau damping is negligible
  and electron-ion collisions dominate; a BGK or Lenard-Bernstein term gives the
  resonance a finite, representable width. This is the physically correct model
  for the regime, not a numerical patch.
- **The instrument function, applied before sampling.** A real spectrometer
  integrates a finite slit over the line. `ThomsonSpectrogram. apply_instrument_response` does this *after* the spectrum is sampled, which is
  too late -- the line has already been missed. Convolving on a grid refined at
  the known resonance positions would be correct.

Neither is done.

### 14.6 Two things a log colour scale showed that a linear one hid

`tools/pic_thomson_osiris_comparison.py` now defaults to a log colour scale, and
it changed the reading of the figure twice.

- **The EPW satellites are visible after all**, in exactly the window §14.3
  predicts: t = 0.33-0.50 ns, where alpha drops to 2.5-5. Both branches track
  the density. On a linear scale that whole structure sat under the skirt of the
  central feature and read as black.
- **The automatic mask was eating them.** At alpha ~ 4.8 the containment
  measurement returned 17 nm, because the IAW window there holds the electron
  feature as well as the ion one, and the mask came out 30 nm wide -- a white
  block straight across the satellites. Capping the half-width at
  `notch_max_fraction` (0.2) of the Bohm-Gross offset fixes it.

The cap had a bug worth recording: it read `alpha_epw[step]`, which is not
assigned until after the EPW call the mask is being built for, so
`_satellite_offset` saw NaN and the cap silently never applied. It now takes the
scattering parameter from the IAW call made just above.

What is left in the corrected panel outside that window is the central feature's
skirt, and at alpha = 12-24 there is no computable satellite to compete with it
-- §14.2, not the mask.

## 15. Validation against a known answer

The shock runs have no analytic spectrum, so a disagreement there cannot be
attributed to anything. Three WarpX runs of a **uniform Maxwellian hydrogen
plasma** fix that: the deck sets `n_e`, `T_e`, `T_i` and the drift, so
`thomson.spectral_density` is the exact answer and the whole chain can be
checked against it. Decks in `tools/warpx_validation_decks/`, harness in
`tools/pic_thomson_warpx_validation.py`.

### 15.1 The reader

| case           | quantity | deck      | recovered | error     |
| -------------- | -------- | --------- | --------- | --------- |
| collective     | `n_e`    | 1e25 m^-3 | 1.0000e25 | **0.00%** |
| collective     | `T_e`    | 100 eV    | 99.92     | **0.08%** |
| collective     | `T_i`    | 50 eV     | 50.03     | **0.05%** |
| non-collective | `T_e`    | 500 eV    | 497.64    | **0.47%** |
| drifting       | `u_e`    | 1.5e6 m/s | 1.4983e6  | **0.12%** |

`T_e` moves from 99.918 to 99.917 eV over a full plasma period, so there is no
numerical heating to confuse with a pipeline error.

### 15.2 The spectrum

Against `spectral_density` at the **measured** moments, so PIC heating is not
charged to the pipeline:

| case           | alpha | alpha error | satellite position        | band power   |
| -------------- | ----- | ----------- | ------------------------- | ------------ |
| non-collective | 0.37  | 0.53%       | exact                     | **0.0-1.6%** |
| collective     | 2.55  | 0.54%       | within one bin (0.33 nm)  | **0.1-1.1%** |
| drifting       | 2.55  | 0.50%       | IAW peaks within 0.027 nm | 5-8%         |

The drifted IAW feature lands at 527.896 nm against the analytic 527.921 --
**0.025 nm**, right sign, right magnitude. The band powers are worse there only
because the split point is taken from the electron drift while the ion feature
moves with the ion drift; the peaks are what the measurement is.

### 15.3 Accuracy against alpha, on analytic input

No PIC noise at all, so this is the model's own accuracy:

| alpha | satellite position error | blue-wing power (vdf vs analytic) |
| ----- | ------------------------ | --------------------------------- |
| 0.31  | 0.000 nm                 | 31.386% vs 31.389%                |
| 1.01  | 0.095 nm                 | 30.972% vs 31.003%                |
| 2.05  | -0.285 nm                | 19.7% vs 19.3%                    |
| 4.12  | -1.045 nm                | 6.52% vs 6.86%                    |
| 8.32  | **+6.3 nm**              | 0.056% vs 0.055%                  |
| 17.4  | **+9.2 nm**              | 0.003% vs 0.002%                  |

**The pipeline is good to a few percent for alpha \<~ 4.** By alpha ~ 8 the
satellites carry under a thousandth of the scattered power and their position is
unreliable -- which is §14.2's precision limit arrived at from a completely
independent direction, and it is a property of the physics and of double
precision, not of the conditioning.

### 15.4 What the exercise found

- **A bug.** WarpX renames a plotfile it is about to overwrite to
  `<name>.old.<pid>`, and `_warpx_plotfiles` globbed `<prefix>*`, so a run that
  had been killed and relaunched came back with a stale plotfile as an extra
  timestep -- carrying another run's data, sorted right after the step it
  duplicates. Now matched against `<prefix><digits>` exactly. Found because the
  first attempt at the collective run was killed and the reader silently
  reported three dumps where there were two.
- **A trap, not a bug.** `spectral_density` returns `S(k, omega)`;
  `arbitrary_forwardmodel` with `scattered_power=True` returns power per unit
  wavelength, which is that times `(1 + 2 omega/omega_0) * 2/lambda^2`.
  Comparing them directly puts a factor of three of tilt across a 280 nm window
  -- 1.9x at the blue end falling to 0.4x at the red -- which looks exactly like
  a pipeline error. Matching the convention takes the non-collective L1 from
  0.151 to **0.0076**.

## 16. Reduced mass ratio: what the pipeline may and may not undo

Applied to `KinShock2020/runs/R1_phase/R1_paper_470eV` -- a **kinetic-electron**
run, unlike the H-PICShock hybrids -- at z = 0.69 mm. Figure:
`media/13_kinshock_470eV.png`.

### 16.1 What the deck fixes

`n_amb = 4.8e24 m^-3` and `B0 = 7.026 T` are physical, `T_e` is physical, and
**the electrons are real electrons** (`species_type = electron`). Only the ions
are light: `m_i = 100 m_e`, so `R = m_p/m_sim = 18.36`.

That splits the quantities cleanly:

| already physical          | corrupted by sqrt(R) = 4.29   |
| ------------------------- | ----------------------------- |
| `lambda_De` (n, T_e only) | `v_A`, `c_s`, `v_ti`          |
| `v_te` (real m_e)         | ion bulk and thermal velocity |
| `omega_pe`                | (and `omega_ci` by R)         |

**So `alpha` and the EPW satellite are correct as read, and only the ions need
rescaling.** Three treatments, measured:

| treatment               | alpha (median) | IAW FWHM (median) | EPW resolved |
| ----------------------- | -------------- | ----------------- | ------------ |
| A, nothing scaled       | 3.15           | 4.41 nm           | 24/51        |
| **B, ions / sqrt(R)**   | **3.15**       | **1.54 nm**       | **24/51**    |
| C, everything / sqrt(R) | 13.51          | 1.43 nm           | 3/51         |

A hands the forward model a 26.4 keV proton where the deck set 1.44 keV at the
simulated mass, so the ion feature is ~3x too wide. C divides the *electron*
thermal speed by 4.29 as well, which is already physical, and drags alpha from
3.15 to 13.51 -- past the double-precision limit of §14.2, hence 3 usable frames
out of 51. **B is the only correct treatment for a run of this kind.**

### 16.2 What no rescale can fix

`u_e/u_i = 0.14` before any rescale -- the two species genuinely do not move
together, because a perpendicular shock carries a real cross-field current. The
reduced mass ratio changes *how* different they are: the current layer is an
ion-scale structure, so a lighter ion makes it sqrt(R) thinner in `d_e` and the
drifts inside it sqrt(R) larger relative to `v_te`. Measured `u_e/v_te` is
0.20-0.36 in the shocked layer, against 0.05-0.08 for the same structure at the
physical mass.

That ratio is a **dimensionless parameter of the problem**, and one velocity
factor cannot restore it: the electron distribution carries a thermal scale that
is already right and a bulk scale that is sqrt(R) too fast, in the same array.
An electron feature from a reduced-mass-ratio run is therefore a spectrum of a
plasma with the wrong drift-to-thermal ratio, and no post-processing undoes it.
The ion feature has no such problem -- both its scales are wrong by the same
sqrt(R) -- which is why `ion_velocity_scale_factor` works and an electron
equivalent would not.

### 16.3 The run itself

alpha runs 0.78-5.5, so unlike the OmegaShock and H-PICShock runs this one sits
**inside the band validated in §15** for most of its history. 24 of 51 frames
have the EPW satellite inside the sampled tail. The satellites track Bohm-Gross
to a median of 5.3 nm on those frames, over a density rise of 4.8e24 -> 2.4e26.

### 16.4 Two more reader bugs, both found here

- **Plotfile ordering.** WarpX pads the step to six digits without truncating,
  so past step 999999 the seven-digit names do not sort lexically against the
  six-digit ones: `diag11002384` (step 1002384) came before `diag1111376`
  (step 111376). The time axis ran backwards in places and the run appeared to
  end at 154 ps instead of 453 ps. Sorting is on the parsed integer now.
- **Velocity resolution.** One velocity axis spans every frame and every cell,
  so it is sized by the fastest macroparticle in the run while the feature is
  carried by the coldest population. Here the piston ions reach 2.6e7 m/s and
  the 10 eV upstream ions got **1.4 bins per thermal width**, a 4% temperature
  bias (`dv^2/12`, confirmed: 10.57 eV recovered against 10.11 eV read from a
  single frame). Now warned about, and reported as `bins_per_thermal_width`.

### 16.5 The electron drift, fixed

§16.2 said one velocity factor cannot serve a species whose thermal scale is
already physical and whose bulk scale is not. It cannot -- but a *translation*
can, and the electrons only ever needed a translation.

Writing `f(v) = g(v - vbar)`, the map `f(v) -> f(v - vbar(1/s - 1))` moves the
mean to `vbar/s` and leaves `g` -- every central moment, the temperature
included -- exactly as it was. That is `rescale_vdf_drift`, wired into the
driver as `electron_drift_scale_factor`. Verified on an analytic Maxwellian:
drift lands on target to 1e-6, temperature preserved to 0.0004%.

Four treatments on R1_paper_470eV at z = 0.69 mm:

| treatment                    | alpha (median) | IAW FWHM    | EPW centroid excursion | EPW resolved |
| ---------------------------- | -------------- | ----------- | ---------------------- | ------------ |
| A, nothing                   | 3.15           | 4.41 nm     | 141.9 nm               | 24/51        |
| B, ions / sqrt(R)            | 3.15           | 1.54 nm     | 138.4 nm               | 24/51        |
| C, everything / sqrt(R)      | 13.51          | 1.43 nm     | 102.4 nm               | **3/51**     |
| **D, ions + electron drift** | **3.15**       | **1.58 nm** | **65.4 nm**            | **24/51**    |

D is what a reduced-ion-mass run with kinetic electrons wants: the ion feature
where B put it, `alpha` and `T_e` untouched, and the electron Doppler shift more
than halved -- put where the physical plasma would put it.

A detail worth keeping: treatment C makes the forward model divide by zero.
Compressing the velocity axis by sqrt(R) squeezes the distribution into a
fraction of its bins, and the model divides by the zeros that leaves. The
numerical complaint is itself part of the argument against C.

Still not restored, and not restorable: the *ratio* of drift to thermal speed.
The simulated plasma really does have a thinner current layer in `d_e` and a
faster drift in `v_te` (measured `u_e/v_te` 0.20-0.36 against 0.05-0.08
physical). D fixes the line centre, which is what a Doppler measurement reads;
it does not make the electron distribution's *shape* that of the physical
plasma.

## 17. The noise in figure 13, and the notch

Two things about `media/13_kinshock_470eV.png` looked wrong: the speckle that
appears in both features once the piston reaches the probe, and the shape of
the EPW stray-light mask. Neither is physical. This section is what they were.

### 17.1 The noise is numerical, and it is two different things

Three measurements, on `R1_paper_470eV` at z = 0.69 mm.

**It is uncorrelated frame to frame.** In the red wing of the EPW window the
fluctuation about a running mean is 0.6 dex -- a factor of four -- and the
correlation between adjacent frames is 0.07-0.34. A shock feature evolving on
a 9 ps frame spacing does not do that.

**It has a fixed period in wavelength, and that period does not depend on the
velocity binning.** Rebinning the cached histogram from 512 to 256 to 128
velocity bins leaves the ringing period at 2.2-2.9 nm while changing its
amplitude, so it is not the velocity grid being read through the Doppler map.

**The conditioned distribution it comes from is smooth.** At step 33 the
piston-electron VDF after conditioning is monotonic beyond the tail join and
has a log-residual of 0.0004 dex out to 4.5 thermal speeds -- `extend_vdf_tail`
joins at 1.5 sigma and everything the EPW wing reads is the fitted Maxwellian.
The distribution has no structure at all where the spectrum has 0.6 dex of it.

So the ringing is made inside the forward model.

### 17.2 The principal-value quadrature

`arbitrary_chi` evaluates

    chi ~ integral f'(u) / (u - xi) du

on a sample grid that is *anchored at xi* and graded away from it --
`nPoints=1e3` points, 80% of them within `inner_range` of the singularity, the
rest spread over the remaining 90% of the axis. The outer spacing is therefore
about `0.009 * deltauMax`, which on this run is 0.1 in u, or 2.9 nm of
wavelength.

As xi sweeps the wavelength axis, that grid slides underneath f'. Against a
smooth f' this is a convergent quadrature and the default is fine: on an
analytic Maxwellian the default reproduces the `nPoints=3e5` answer to **0.0036
dex at alpha = 0.82, 0.0215 at alpha = 2.37 and 0.0234 at alpha = 4.47**, and
the error falls as 1/nPoints. Against a f' carrying macroparticle noise the
sliding grid samples a different set of bumps at every xi and the error
oscillates with the sample spacing.

Raising `n_quadrature_points`, over 51 frames:

| window        | 1e3 (default) | 1e4       | 1e5   |
| ------------- | ------------- | --------- | ----- |
| EPW wing rms  | 0.563 dex     | **0.100** | 0.100 |
| IAW wing rms  | 0.637 dex     | **0.045** | 0.045 |

1e4 and 1e5 agree, so 1e4 is converged. The **IAW feature FWHM moves by up to
14%** between the default and the converged answer, which is a systematic on
every ion-temperature number this pipeline has produced.

This is now a parameter, `n_quadrature_points`, on
`spectra_from_phase_spaces`. The default is unchanged -- the old behaviour is
still what you get unless you ask -- because it is correct for the analytic and
`from_moments` inputs, and only wrong for noisy ones.

### 17.3 What is left is macroparticle statistics, and the timing gives it away

At converged quadrature the wings still carry 0.10 dex. Resampling the
histogram from its own macroparticle quantum as a Poisson draw and re-running
moves the spectrum by **0.17-0.53 dex** (the resample doubles the variance, so
the true figure is that over sqrt(2)). The residual is shot noise.

Counting macroparticles in the probe cell says why it starts when it does:

| t (ps)    | electrons in the probe cell |
| --------- | --------------------------- |
| 0-163     | 12,000 - 47,000             |
| **199**   | **2,162**                   |
| 199-453   | 2,200 - 5,000               |

The ambient population is swept out of the cell and the piston population that
replaces it is loaded at 100x the weight and 20x fewer macroparticles. The
noise in the published figure begins at exactly that frame. It has nothing to
do with the piston "dominating" in any physical sense -- it is the frame where
the cell's sampling collapses.

The EPW wing at 580-655 nm reads the electron distribution at **1.5 to 4.3
thermal speeds**, where 2,000 particles put fewer than ten in a bin. The
collective factor `|1 - chi_e/epsilon|^2` then turns each noise excursion that
carries epsilon near zero into a spurious resonance, which is why the late
frames are a forest of isolated spikes several decades above a near-zero floor
rather than a noisy continuum.

### 17.4 One cell is the wrong probe volume

A Thomson collection volume at 532 nm is tens to a hundred microns. A cell here
is 7.6. Averaging the phase space over the cells the real volume spans is what
the measurement does anyway, and it is free:

| probe volume    | EPW wing rms (dex)          | IAW FWHM (nm)          |
| --------------- | --------------------------- | ---------------------- |
| 1 cell, 7.6 um  | 0.10 0.05 0.25 0.02 0.12    | 1.37 2.60 1.61 1.56    |
| 5 cells, 38 um  | 0.01 0.10 0.17 0.11 0.11    | 1.80 1.67 1.69 1.82    |
| **11 cells, 84 um** | **0.02 0.10 0.06 0.04 0.05** | **2.90 1.99 1.59 1.55** |
| 21 cells, 160 um| crosses too much gradient; frames go NaN |    |

Average, do not sum -- summing 11 cells multiplies the density by 11 and drags
alpha from 3.19 to 11.98.

### 17.5 The notch was never a stray-light notch

`epw_notches="auto"` sized the mask as

    min( 1.5 * (99.9% containment interval of the IAW window),
         0.2 * (Bohm-Gross offset) )

Both halves are wrong for this.

**The containment half measures the wings, not the line.** At the same frame
the interval is 17.8 nm at 0.999 containment and **0.79 nm at 0.99** -- the last
0.1% of the area is spread over the whole window, so the measurement is of
whatever is in the wings. Late in the run part of what was in the wings was the
quadrature ringing of 17.2, so the mask width was being set by a numerical
artefact.

**The cap half is a function of density, not of the instrument.** Where the cap
binds -- most of the first half of the run -- the mask is 0.4 x the Bohm-Gross
offset, so it scales as sqrt(n_e) and reaches 20% of the way to the satellites
by construction.

Between them the applied mask ran **5.1 to 24.0 nm wide, wandering frame to
frame**, against a central feature that is 0.42-2.5 nm wide by the 95%
containment measure -- a ratio of 3 to 19. At t = 145 ps, where alpha = 0.82 and
the spectrum is very nearly a flat non-collective continuum with no central
feature to speak of, it cut a 21 nm hole in it.

It never ate the Bohm-Gross satellites; the cap prevented that. What it did was
change width every frame, which no piece of glass does, and blank real spectrum
between the line and the satellites.

A fixed notch, sized once from the data, is both more honest and easier to
justify. The central feature spans 526.5-537.4 nm over every collective frame
of this run and the nearest satellite is at +-20 nm, so **[526, 538] nm** covers
it everywhere with 14 nm of clearance. That is what figure 13 now uses.

### 17.6 What this invalidates

- **The 440-660 nm EPW window was too narrow.** Past ~250 ps the Bohm-Gross
  satellites are at +-100 to +-131 nm, i.e. outside it. Everything the published
  figure showed in that window at late times was the spike forest of 17.3, not
  the electron feature. The window is now 400-680 nm and the satellites are
  visible tracking Bohm-Gross for the whole second half of the run.
- **The "EPW centroid excursion" column of 16.5 measures that noise.** With the
  satellites outside the window, the centroid of the window was tracking
  spurious spikes. Measured properly -- as the midpoint of the two satellites,
  which is what a bulk drift moves -- the excursion is 26.9 nm (A), 38.9 (B),
  35.5 (D), with an rms scatter about the mean of 6-9 nm. The expected B-to-D
  difference from the measured drift is about 5 nm, which that estimator cannot
  resolve. **The claim that D more than halves the electron Doppler excursion
  does not survive.** What does survive is the argument itself and its direct
  check: `rescale_vdf_drift` puts the mean where it is asked to to 1e-6 and
  leaves the temperature to 0.0004%, which is verified on the distribution, not
  read off a spectrum.
- **Frames with a resolved electron feature: 24/51 becomes 36/51** once the
  window holds the satellites and the probe volume is realistic.
- IAW FWHM medians move: A 4.41 -> 4.57 nm, B 1.54 -> 2.12, D 1.58 -> 2.20. The
  B/D-versus-A conclusion is unchanged; the absolute widths were low.

### 17.7 The same treatment on the OSIRIS figure

`media/07_osiris_end_to_end.png`, from `omegashock_w3.5e11_exp`. Three of the
four corrections transfer; the numbers differ enough to be worth recording.

**Quadrature: it matters here too, but less.** Measured against `nPoints=1e5`
on the `osiris2thomson` run at `alpha ~ 20`:

| window                       | 1e3 (default) | 1e4       |
| ---------------------------- | ------------- | --------- |
| EPW satellite band, 540-560  | 0.257 dex     | **0.030** |
| IAW window, 522-542          | 0.027 dex     | **0.002** |
| IAW FWHM                     | 0.1288 nm     | 0.1277 nm |

The FWHM moves 0.9% rather than KinShock's 14%: this run smooths with four
passes and has up to 5e5 macroparticles in the sampled cell, so `f'` is far
smoother and the sliding quadrature grid has much less to trip over. The
satellite band still carries a quarter of a decade at the default.

**A caution about that measurement.** The first version of it passed
`reference_density` as a bare float where the tool passes `u.cm**-3`, a factor
of 1e6 in density. That put `alpha` at 0.00-0.03 instead of 5-20, i.e. fully
non-collective, where `chi` is negligible and the quadrature cannot matter --
and the measurement duly said the quadrature did not matter. It reads as a
clean negative result and is entirely an artefact of the units.

**The notch: same pathology, tighter constraint.** The automatic mask on this
run ranges over **0.52 to 9.92 nm, a factor of 19, with a median of 0.64**.
But a fixed replacement is harder to choose here than on KinShock: the central
feature spans 520.2-537.4 nm at 1e-3 of its peak while the nearest Bohm-Gross
satellite over the run is only **8.2 nm** out, at 523.8 and 540.2 nm. Those
overlap. Where the central feature is at its widest the two features have
genuinely merged and no mask separates them -- that is the physics of
`alpha ~ 2-3`, not a sizing failure.

The default is now `[527, 537]`, which covers the central feature in 90% of
frames and clears the closest satellite by 3.2 nm.

**Probe volume applies.** Cells are 19.9 microns here, so the default 5-cell
average is a 100 micron collection volume.

**What does not transfer** is the conclusion about which frames are usable. In
the corrected configuration `omegashock_w3.5e11_exp` sits at `alpha` 2.5-23.7,
median 16.4 -- mostly past the double-precision limit of 14.2, so the satellite
is not computable there whatever the quadrature. Where the corrected EPW panel
is thin, that is 14.2, not the treatment. (KinShock is the opposite case, and
that is why it was the run worth chasing: `alpha` 0.78-5.5 puts it inside the
band validated in 15 for most of its history.)

What the colour rule does buy on this figure is the legacy panel's satellites
after 0.5 ns, at 480 and 580 nm. They were always in the data; with the limit
set from the whole panel the central line took the scale and they read as
nearly black.

### 17.8 The OSIRIS runs had the wrong velocity treatment all along

The EPW panel of `07_osiris_end_to_end.png` was empty, and 17.7 attributed that
to `alpha` sitting past the double-precision limit of 14.2. That was true, and
it was not the whole story: **`alpha` was that high because the pipeline was
applying the wrong correction.**

`omegashock_w3.5e11_exp.1d` says:

```text
species { name = "e",    rqm = -1.0 }
species { name = "cham", rqm = 69   }
species { name = "targ", rqm = 68   }
```

`rqm = -1.0` is a **real electron**. So this run has exactly the structure of
16.1 -- kinetic electrons at the physical mass, ions at a reduced one -- and not
the similarity-scaled structure the tool assumed. `T_e`, `v_te`,
`lambda_De` and therefore `alpha` are already physical; only the ion velocities
and the flow the electrons share with the ions are wrong.

`run_pipeline` passed `velocity_scale_factor`, which divides *every* species'
axis by sqrt(R). That is treatment C of 16.5. On this run:

| treatment                                | alpha median | alpha range |
| ---------------------------------------- | ------------ | ----------- |
| all velocities / sqrt(R)  (what it did)   | 11.58        | 1.79-16.77  |
| **ions / sqrt(R) + electron drift**       | **1.64**     | 0.25-2.37   |

The ratio is 11.58 / 1.64 = 7.06 = sqrt(50), which is the whole of it: dividing
the electron axis divides `v_te`, and `alpha = sqrt(2) omega_pe / (k sigma)`
rises by the same factor. It carried a perfectly computable spectrum from
inside the band validated to a few percent in 15 to past the limit where the
satellite cannot be evaluated in double precision at all.

**The satellites were never missing from the physics. They were being scaled
out of existence.**

Switching the default moves the run's own diagnostics too, all in the same
direction:

| quantity                                  | before      | after       |
| ----------------------------------------- | ----------- | ----------- |
| `alpha` (corrected config, stride 10)     | 16.38       | **2.32**    |
| frames whose satellite outruns the data   | 43 of 52    | **30 of 52** |
| thermal widths the satellite needs        | up to 7.4   | **up to 2.0** |
| `epw_tail_uncertainty` at the resonance   | large       | **1.01**    |

The last is the one that matters: at 1.8 thermal widths the satellite sits
*inside* the sampled distribution, so it is a measurement rather than an
extrapolation off the fitted tail.

Both configurations of `07` now use the corrected scaling, so the legacy
comparison still isolates the taper -- which is what it was for -- rather than
conflating it with the velocity treatment. `06`, `07`, `10` and the new `14`
are regenerated.

**What this does not settle** is `R = 50` itself. `rqm = 68.5` against a proton
gives R = 1836/68.5 = 26.8; R = 50 requires A/Z ~ 1.87, the fully-stripped
low-Z ion that is still open decision 1, and the ion labels are still `p+`.
That moves the IAW width, not the `alpha` above.

### 17.9 What figure 14 is

`media/14_osiris_spectra.png`, from `tools/pic_thomson_osiris_spectra.py`: the
EPW and IAW spectrograms for the two treatments, and nothing else. No legacy
column -- the taper is settled and it only crowds the plot.

**Each row is drawn on one absolute colour scale across every timestep.** The
other figures normalise each row to its own area, which is a trap for the EPW:
a frame carrying no signal then looks exactly like a frame carrying a strong
one, and where the notch has removed the central feature the only content left
is the skirt at the notch edge -- 1e-5 of the peak -- which normalisation
promotes into a saturated rail along both notch edges, across every such frame.
That rail is what made the EPW panel unreadable, and widening the notch only
moved it. Sharing one scale makes brightness mean intensity.

## 18. Exact susceptibility, and a regression in 17.8

### 18.1 17.8 shipped with the satellites in the wrong place

17.8 switched the OSIRIS electrons from a whole-axis rescale to a drift-only
translation. That was right for alpha, and it exposed something the whole-axis
path had hidden: OSIRIS bins proper velocity, so after `v = u c / sqrt(1 + u^2)`
the axis is non-uniform -- edge spacing 0.354 of the centre on `u` in +-1 --
and `arbitrary_derivative` assumes `dx = x[1] - x[0]`. `rescale_velocity_axis`
used to resample every species onto a `linspace` and so fixed the grid by
accident; drift-only never resamples, so the raw grid reached the model and
`f'` was wrong by up to 2.8x. The satellites of figures 06, 07, 10 and 14 sat
about 6 nm from Bohm-Gross (515/549 against 520/544 nm on an analytic
Maxwellian). Alpha barely moved, which is why 17.8's table was right and its
figures were not.

Fixed in two places: `condition_phase_space` resamples any non-uniform axis
onto a uniform one at its finest spacing (linear interpolation; width moves by
about dv^2/24 sigma^2, 1e-4 here), and `arbitrary_derivative` raises on a
non-uniform axis rather than return a wrong answer.

### 18.2 The principal value has a closed form

The quadrature in `arbitrary_chi` integrated the piecewise-linear interpolant
of `f'` numerically -- and that integral is elementary. Writing
`L(t) = L(u0) + sum_j c_j (t - u_j)_+` with `c_j` the change of slope at node j,

    PV int L(t)/(t - x) dt = L(u0) ln|(uN - x)/(u0 - x)|
                             + sum_j c_j [(uN - u_j) + (x - u_j) ln|(uN - x)/(u_j - x)|]

Replacing the quadrature with this removes three faults at once: the
first-order convergence in `nPoints` that made noisy input ring (17.2); the
window `xi +- span`, which missed part or all of an ion distribution whenever
`|xi|` exceeded the span -- always, in an EPW window -- and which no `nPoints`
fixed (0.17 dex left at 1e5); and the floor/ceil split that broke the
cancellation across the singularity for odd `nPoints` (22.7% at 1001).

Verified against brute-force quadrature at h/2000 (4e-11 Maxwellian, 4e-8 noisy
histogram) and against the large-xi asymptote `int f / xi^2 + 3 int u^2 f / xi^4`
to 1e-4. `n_quadrature_points`, added in 17.2, is gone; the quadrature
parameters of `arbitrary_chi` are accepted and unused.

### 18.3 What it buys

| | before | after |
|---|---|---|
| 51 KinShock frames, model | 27 s (1e4), 270 s (1e5) | under 1 s |
| per-process JIT | 8.5 s | cached |
| `test_pic_thomson` + `test_thomson` | 44 s | 9 s |
| OSIRIS end to end, 52 frames | minutes | 9 s, now dominated by reading |

Satellites against Bohm-Gross on the 38 collective frames of
`omegashock_w3.5e11_exp`: median +0.12 nm (blue), -0.37 nm (red). The tool's
own check reported "observed/predicted" against the bare plasma-frequency shift
and selected frames by whether *that* cleared the notch, which excluded every
collective frame and compared the non-collective ones, where there is no
resonance; it now compares against Bohm-Gross on frames with alpha > 1.2.

### 18.4 Still open (from the audit)

- `extend_vdf_tail` treats the smallest positive value as one particle; OSIRIS
  deposits fractional, variably weighted contributions, so inferred counts are
  3-200x high and `min_counts=10` means 0.05-3 particles.
- The vacuum threshold is relative to `reference_density`, which WarpX,
  openPMD, `from_moments` and `from_arrays` set to 1 m^-3, so it never fires.
- Ion terms weighted by `Z * ifract` rather than `Z^2 * ifract / Zbar`; ion
  mass passed in units of m_p and converted with the amu (0.72% light).
- A frame's spectrum depends on which other frames are passed (conditioning
  sizes windows from run-wide quantities): chunking a run changes frames by up
  to 2.7 dex.
- The torch path (`autodiff_forwardmodel`) still uses the old quadrature.

## 19. The rest of the audit

Fixed, each with a regression test:

| fault | effect | fix |
|---|---|---|
| ion term weighted Z f instead of Z^2 f / Zbar (five fork paths) | H/C mixture ion feature 24-39% off; one species unaffected | Z^2/Zbar, matching `spectral_density` |
| ion mass in m_p, converted with the amu | ions 0.72% light | convert at the call |
| `arbitrary_forwardmodel` mutated the caller's ion list | strings became `Particle` | copy |
| torch `autodiff_chi` still on the old quadrature | ion truncation, odd-nPoints | closed form, matches numpy to 1e-9 |
| vacuum threshold relative to `reference_density` = 1 m^-3 for SI readers | never fired | `vacuum_density`; SI default from the peak at the probe |
| alpha of an electron mixture an unweighted mean | `epw_resolved` wrong both ways | alpha^2 = sum f_p alpha_p^2; per-population requirement |
| `{"skip": True}` passed unnormalised f to the model | alpha 21 orders out | normalise each slice before the call |
| smoothing window from the narrowest slice of the run | chunking moved frames 2.7 dex | per-slice windows |
| smallest value = one particle, on fractional deposits | OSIRIS counts 12-14x high | shot-noise quantum; deck check 0.84-1.07 (ions), ~1.9 (electrons) |
| cache key = repr | numpy-1/2 caches never matched | canonical JSON, old keys still match |
| every frame area-normalised by the model | no brightness history; the notch-edge rails | driver returns n_e S(k, w) by default |
| automatic notch re-sized per frame | wandered 19-26x | one run-wide notch on the probe |

The last two change what figures show. Figure 14 now has a brightness
history: on `omegashock_w3.5e11_exp` the pre-shock satellites are faint at
9e17 cm^-3 and the post-shock ones bright after the density jumps ~39x, which
is what a streak camera records. Before, every frame was stretched to the same
area. On OSIRIS, honest particle counts move the tail join inward (electrons
3.65 -> 2.48 sigma, ions 3.42 -> 2.51 sigma), and `epw_tail_uncertainty` rises
from 1.01 to 1.35 because more of the tail is now extrapolated and priced.

Still open:

- the SI vacuum default reads the run's peak density, so it is the one
  remaining cross-frame dependence; pass `vacuum_density` to pin it;
- the WarpX reader sizes its velocity grid from the fastest macroparticle in
  the frames it reads, so reading a different subset bins differently;
- the OSIRIS electron shot-noise count is ~2x the deck's (ions are right);
- R = 50 vs rqm 68.5 and the `p+` labels (open decision 1);
- WarpX spectra still in simulation units (11.4).
