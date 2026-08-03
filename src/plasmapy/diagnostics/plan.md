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
