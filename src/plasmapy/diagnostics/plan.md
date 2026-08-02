# Plan: `pic_thomson.py` — synthetic Thomson spectra from generic PIC output

Branch: `feature/pic-thomson-pipeline`
Target file: `src/plasmapy/diagnostics/pic_thomson.py` (single module, next to `thomson.py`)

Goal: fold the `osiris2thomson` pipeline into this fork as **one** module that takes
PIC output (OSIRIS, WarpX, and ideally anything else) and produces synthetic EPW/IAW
Thomson spectra via `thomson.arbitrary_forwardmodel`. Only what is needed for the
spectra — no temperature/B-field/`ufl`/`uth` diagnostics.

---

## 1. What the existing pipeline actually does

Source: `/home/hhelal/osiris2thomson/src/osiris2thomson/`, entry point
`synspectra.data_to_spectra()` (942-line module; the rest is smoothing, HDF5
loading, and moments helpers).

Traced end to end, `data_to_spectra` does the following:

| # | Step | Functions | Code-specific? |
|---|---|---|---|
| 1 | Build OSIRIS filenames `MS/PHA/<field>/<species>/<field>-<species>-NNNNNN.h5` and read every timestep with `osh5io.read_h5` | `file_name_phase`, `input_PHA_t` | **Yes** |
| 2 | Read `MS/FLD/<b_field>/…` B-field series | `file_name_mag`, `input_mag_t` | **Yes** — *drop* |
| 3 | If the phase space is 3-D (`p1x1x2`/`p2x1x2`), sum over the transverse spatial axis to get `(t, p, x)` | inline | Partly (reduce-to-1D concept is general) |
| 4 | Momentum axis → velocity axis: `v = u·c/√(1+u²)` where `u = p/(m c)` is OSIRIS proper velocity | inline | Boundary — general once `u` is defined |
| 5 | Pick the spatial slice `y_slice = argmin|x − y_value|` | inline | Invariant |
| 6 | Zeroth moment along `p` → density vs `(t, x)`; scale by reference density `n` [cm⁻³]; first/second moments → drift + `T` | `moments.moment`, `second_moment_to_eV` | Invariant (only the 0th moment is needed) |
| 7 | Ion fractions `ifract_s = n_s / Σ n_s` at the slice; presence masks (species below 1 % of reference is "absent") | `species_presence_mask` | Invariant |
| 8 | Smooth VDFs: repeated boxcar (`uniform_filter1d`) along the velocity axis | `smooth_vdf` | Invariant |
| 9 | Normalise so `∫f dv = 1` per (t, x); clip negatives; guard div-by-zero | `normalize_vdf` | Invariant |
| 10 | Half-cosine taper of the VDF tails to kill the sharp PIC noise floor at the grid edge | `taper_vdfs` / `taper_vdf_edges` | Invariant |
| 11 | Floor at 1e-30 | inline | Invariant |
| 12 | "Fudge factor": divide the velocity axes by `√rqm` and re-interpolate onto a padded grid, to undo the reduced ion/electron mass ratio | `rescale_and_pad_vdf` | Invariant (but see §5.1) |
| 13 | Per timestep, call `thomson.arbitrary_forwardmodel` twice — once over the EPW window (with a notch) and once over the IAW window | `vdfs_to_spectra` | Invariant |
| 14 | Instrument smoothing of the spectrogram (Gaussian 1-D/2-D or boxcar) to e.g. 100 ps / 0.5 nm | `smooth_spectra.*` | Invariant |
| 15 | Write everything to a structured HDF5 file | `create_hdf5_file` | Invariant |
| 16 | Plot the two spectrograms | `plot_spectra` | Invariant |

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

---

## 2. The invariance boundary

One dataclass is the contract between readers and physics:

```python
@dataclass(frozen=True)
class PICPhaseSpace:
    """Reduced 1V phase space f(t, v, x) for a single species, in SI."""
    f: np.ndarray          # (n_time, n_v, n_x), arbitrary normalisation
    v: np.ndarray          # (n_v,) lab-frame velocity along the diagnostic axis [m/s]
    x: np.ndarray          # (n_x,) position along the same axis [m]
    t: np.ndarray          # (n_time,) [s]
    label: str             # PlasmaPy `ParticleLike`, e.g. "e-", "Al 13+", "p+"
    is_electron: bool
    meta: dict             # code name, source paths, normalisations used
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

---

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

```python
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

```python
def read_warpx_phase_space(diag_glob, species, *, mass, n_v=512, n_x=512,
                           v_range=None, x_range=None, axis="z",
                           backend="auto") -> PICPhaseSpace
```

1. Enumerate plotfiles (`sorted(glob("diag1*"))`), one per timestep.
2. Per frame, read `particle_position_*`, `particle_momentum_<axis>`,
   `particle_weight`. Backends: `openpmd-api` if the run wrote openPMD, else `yt`
   (what `KinShock2020/src/kinshock/io.py:71-192` uses today). Both are **optional
   imports** with a clear error message — neither belongs in PlasmaPy's core deps.
3. `u = p_axis / (m c)`; `v = u c/√(1+u²)`.
4. `f[t] = np.histogram2d(v, x, bins=[v_edges, x_edges], weights=w)` — weights are
   essential; a raw count histogram is not a distribution function.
5. Bin edges fixed across all timesteps (computed from a percentile of the first and
   last frames, or user-supplied) so `f` is a well-defined array.
6. WarpX is already SI, so no unit conversion — `v` in m/s, `x` in m, `t` in s
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

---

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

```python
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

---

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

```python
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
|---|---|
| 4σ | +0.4 % |
| 6σ | +5.4 % |
| 8σ | +13.7 % |
| 12σ | +40.4 % |

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

---

## 6. Testing

### 6.1 Unit tests (`tests/diagnostics/test_pic_thomson.py`)

1. **Reader round-trip** — `from_arrays` → conditioning → shapes/units preserved.
2. **`normalize_vdf`** — `∫f dv = 1` for every `(t, x)` on a non-uniform-amplitude
   input; per-species axes respected (the §5.2 regression).
3. **`taper_vdf_edges`** — vectorised version matches a straightforward per-slice loop
   (the existing repo has this equivalence test; port it).
4. **Maxwellian consistency** — build an analytic Maxwellian `f(v)` at known `n`, `T`,
   drift; push it through `spectra_from_phase_spaces`; compare against
   `thomson.spectral_density` (the standard PlasmaPy Maxwellian path) on the same
   geometry. This is the real validation that the pipeline's conditioning does not
   distort the physics, and it is code-independent.
5. **OSIRIS reader** — against a tiny committed fixture (one 8×8 phase-space HDF5
   written by the test itself in OSIRIS layout), asserting axis order, the
   `AXIS{ndim-k}` flip, and the `c/ω_p → m` conversion.
6. **Presence masking** — a species that vanishes mid-run yields `NaN` columns, not
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

---

## 7. Open questions

1. ~~§5.1 mass-ratio convention~~ — resolved: global `1/√R` rescale of all species,
   exposed as `velocity_scale_factor` (default 1.0). Only the numeric value of `R`
   for `omegashock_w3.5e11_exp` (50 vs 69) remains, and that is a test-time check.
2. ~~File name~~ — resolved: `pic_thomson.py`.
3. **Optional deps.** `yt` and `openpmd-api` for WarpX — add a
   `[project.optional-dependencies] pic` extra, or leave them as import-time errors
   with instructions? I lean toward the extra, mirroring the existing `thomson`
   extra for numba/torch.
4. **Multiple electron populations** (§6.2 WarpX): confirm we want `efract` support
   in v1. I plan to build it in, since the WarpX test run needs it.
5. **Upstreamability.** This is a fork-specific module depending on
   `arbitrary_forwardmodel`, which upstream PlasmaPy does not have. Keeping it in
   `plasmapy/diagnostics/` is fine for the fork; worth a header note that it is not an
   upstream-mergeable file as written.

---

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

2. ✅ **Done.** `spectra_from_phase_spaces` driver, `ThomsonSpectrogram`, `efract`
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
3. ✅ **Done.** OSIRIS reader (h5py, no pyVisOS) + unit test 5. 118 tests in the
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

| species | grid half-width | unbounded | bins=20 | bins=10 | bins=5 | bins=3 |
|---|---|---|---|---|---|---|
| `e` | 13σ | **+67 %** | −0.1 % | −0.7 % | −0.9 % | −1.0 % |
| `cham` | 14σ | **+4086 %** | +35 % | +11 % | +3.9 % | +1.6 % |
| `targ` | 23σ | **+242 %** | +6.0 % | +2.8 % | +1.3 % | +0.7 % |

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

4. End-to-end OSIRIS run vs. the existing `spectra.hdf5`.

   **Open before this step:** the figures label both `cham` and `targ` as
   `"Al 13+"` as a placeholder. The deck gives `cham` rqm 69 and `targ` rqm 68,
   which are different species — the chamber gas and the target material. The real
   labels are needed, since the forward model takes `Z` and mass from them, and
   `ifract`/`zbar` depend on `Z`.

   Also worth investigating: at late times the sampled point reaches α ≈ 16, where
   the EPW satellites become faint relative to the ion feature and the
   per-window area normalisation lets the central line dominate the spectrogram.
   That looks like correct physics for the collective regime rather than a defect,
   but it should be confirmed against the old output.
5. WarpX reader + caching.
6. End-to-end WarpX run; cross-code comparison.
7. HDF5 writer, plotting, instrument response.
8. Changelog entry, docs stub, `uvx pre-commit`, `nox --session ty`.
