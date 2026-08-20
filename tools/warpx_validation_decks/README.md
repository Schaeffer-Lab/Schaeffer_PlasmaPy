# Known-answer WarpX decks for the `pic_thomson` validation

Each deck is a **uniform, stationary (or uniformly drifting), Maxwellian**
hydrogen plasma in 1D. Nothing happens in them, and that is the point: the
density, both temperatures and the drift are set in the deck, so
`~plasmapy.diagnostics.thomson.spectral_density` gives the exact Thomson
spectrum and `tools/pic_thomson_warpx_validation.py` can check the whole
pipeline against it.

| deck                   | n_e (m^-3) | T_e    | T_i    | drift             | alpha |
| ---------------------- | ---------- | ------ | ------ | ----------------- | ----- |
| `inputs_collective`    | 1e25       | 100 eV | 50 eV  | 0                 | 2.55  |
| `inputs_noncollective` | 1e24       | 500 eV | 100 eV | 0                 | 0.36  |
| `inputs_drifting`      | 1e25       | 100 eV | 50 eV  | 1.5e6 m/s along z | 2.55  |

Run one, then validate it:

```bash
mkdir -p ~/thomson_validation/collective && cd ~/thomson_validation/collective
cp ~/Schaeffer_PlasmaPy/tools/warpx_validation_decks/inputs_collective inputs
OMP_NUM_THREADS=8 <warpx>/build/bin/warpx.1d.MPI.OMP.DP.PDP.OPMD.EB.QED inputs

PYTHONPATH=$HOME/Schaeffer_PlasmaPy/src conda run -n thomson \
    python tools/pic_thomson_warpx_validation.py ~/thomson_validation/collective
```

Three things make them usable as a reference rather than just a run:

- **The velocity grid is resolved.** `dz/lambda_De` is 0.43 (0.60 for the
  non-collective deck), so there is no numerical heating to confuse with a
  pipeline error: `T_e` moves by 0.001% over a full plasma period.
- **10,000 particles per cell over 256 cells**, all binned into one velocity
  histogram, because the plasma is uniform. That reaches about four thermal
  speeds, which covers the EPW resonance at these `alpha`.
- **k is along the resolved axis.** The probe geometry in the validation tool
  puts the scattering vector along z whatever the scattering angle, which is the
  geometry a shock is analysed in and the one that lets a bulk drift be checked
  against its Doppler shift.
