.. _pic-thomson:

***********************************
Thomson scattering from PIC output
***********************************

.. currentmodule:: plasmapy.diagnostics.pic_thomson

`plasmapy.diagnostics.pic_thomson` turns the phase-space output of a
particle-in-cell simulation into synthetic Thomson scattering spectra, using the
arbitrary-VDF forward model in
:func:`~plasmapy.diagnostics.thomson.arbitrary_forwardmodel`.

.. note::

   This module builds on ``thomson.arbitrary_forwardmodel``, which exists only
   in the Schaeffer-Lab fork of PlasmaPy.

Only the readers are specific to a simulation code. Each one produces a
`PICPhaseSpace` — a reduced one-velocity-dimension phase space
:math:`f(t, v, x)` in SI units — and everything below that boundary is
code-agnostic, so supporting another code means writing one reader.

.. code-block:: text

    OSIRIS  ->  read_osiris_phase_space  --.
                                            >--  PICPhaseSpace
    WarpX   ->  read_warpx_phase_space   --'          |
                                                      v
                                        condition_phase_space
                                                      |
                                                      v
                                        spectra_from_phase_spaces
                                                      |
                                                      v
                                            ThomsonSpectrogram

Getting started
===============

.. code-block:: python

   import astropy.units as u
   import numpy as np

   from plasmapy.diagnostics import pic_thomson

   electrons = pic_thomson.read_osiris_phase_space(
       "run/MS",
       "p1x1",
       "e",
       reference_density=9e17 * u.cm**-3,
       is_electron=True,
   )
   ions = pic_thomson.read_osiris_phase_space(
       "run/MS",
       "p1x1",
       "cham",
       reference_density=9e17 * u.cm**-3,
       label="C 6+",
   )

   spectra = pic_thomson.spectra_from_phase_spaces(
       electrons,
       [ions],
       position=5 * u.mm,
       reference_density=9e17 * u.cm**-3,
       probe_wavelength=532 * u.nm,
       epw_wavelengths=np.linspace(432, 632, 500) * u.nm,
       iaw_wavelengths=np.linspace(522, 542, 500) * u.nm,
       epw_notches=[530, 534] * u.nm,
       electron_conditioning={"smoothing_iterations": 3, "max_taper_bins": 20},
       ion_conditioning={"max_taper_bins": 3},
       velocity_scale_factor=50,
   )

   spectra.apply_instrument_response(
       time_fwhm=100 * u.ps, epw_wavelength_fwhm=0.5 * u.nm
   ).plot()

Things worth knowing
====================

Reading a simulation's units correctly
--------------------------------------

Each reader has to undo its code's conventions, and the details matter more than
they look. OSIRIS stores momentum as proper velocity normalised to each species'
*own* mass, so :math:`v = u c / \sqrt{1 + u^2}` applies to ions as well as
electrons; its electron phase space is a *negative* charge density; and the
:math:`u \to v` map is nonlinear, so ``f`` needs the Jacobian
:math:`\gamma^3 / c`. WarpX writes raw macroparticles instead, which must be
histogrammed **with their weights** to be a distribution at all.

In a reduced-mass-ratio run the mass that converts momentum to velocity is the
*simulation's* mass, not that of the physical species the population represents.
`read_warpx_phase_space` therefore takes ``mass`` and ``label`` separately.

Sizing the taper
----------------

`taper_vdf_edges` smooths away the discontinuity where a PIC noise floor meets
the edge of the velocity grid. Its rolloff runs to the grid boundary by default,
which is only safe when the distribution roughly fills the grid. When it does
not, the rolloff fabricates a pedestal at large :math:`|v|`, precisely where the
:math:`v^2` weighting of the second moment is largest, and the thermal speed the
forward model reads off the distribution comes out far too high — by 67% for the
electrons of a real OSIRIS run, and by a factor of tens for its ions, whose
momentum grid is far wider than their thermal spread.

Set ``max_taper_bins``. The ``pedestal_warning`` reports how much the taper
widened the distribution, so a bad choice does not pass unnoticed.

How much smoothing
------------------

The collective regime needs markedly more velocity smoothing than the
non-collective one. Where :math:`\alpha > 1` the spectrum carries
:math:`|1 - \chi_e/\epsilon|^2`, which diverges as :math:`\epsilon \to 0` at the
electron-plasma-wave resonance, so shot noise entering :math:`\chi_e` through
:math:`\partial f/\partial u` is amplified into speckle. At :math:`\alpha \ll 1`
the same noise passes through untouched.

What the spectra do and do not carry
------------------------------------

The forward model normalises each spectrum to unit area over its own window, so
a `ThomsonSpectrogram` carries **shape only**. There is no absolute intensity, no
brightness history along the time axis, and no meaningful ratio between the EPW
and IAW features.

The scattering parameter it reports is
:math:`\sqrt{2}\, \omega_{pe} / (k \sigma)`, which is :math:`\sqrt{2}` times the
conventional :math:`1/(k \lambda_{De})` that
:func:`~plasmapy.diagnostics.thomson.spectral_density` returns.

Optional dependencies
=====================

Reading WarpX output needs `yt <https://yt-project.org>`__, available as the
``pic`` extra:

.. code-block:: bash

   pip install plasmapy[pic]

The OSIRIS reader needs only `h5py`, which PlasmaPy requires anyway.

API
---

.. automodapi:: plasmapy.diagnostics.pic_thomson
