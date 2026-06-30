# Examples

Standalone, runnable example scripts for PlasmaPy features. Unlike the
rendered tutorials in `docs/notebooks/`, these are plain `.py` scripts so they
are easy to diff, review, and run from the command line. They are not part of
the installed package.

## Contents

- `diagnostics/thomson_arbitrary_vdf_autodiff_fit.py` — fits a non-Maxwellian
  (super-Gaussian) ion velocity distribution to a synthetic Thomson scattering
  spectrum using a differentiable PyTorch port of
  `plasmapy.diagnostics.thomson.spectral_density_arbitrary`. Requires the
  optional `torch` dependency (`pip install plasmapy[thomson]`). The equivalent
  fit *without* PyTorch (using `scipy.optimize`) is exercised by
  `tests/diagnostics/test_thomson.py::test_spectral_density_arbitrary_supergaussian_fit`.
