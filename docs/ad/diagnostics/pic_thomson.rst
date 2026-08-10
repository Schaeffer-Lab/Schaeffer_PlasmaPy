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

    OSIRIS binned phase space  ->  read_osiris_phase_space     --.
    WarpX macroparticles       ->  read_warpx_phase_space      --|
    openPMD 2-D histogram      ->  read_openpmd_phase_space    --+->  PICPhaseSpace
    hybrid electron moments    ->  read_warpx_hybrid_electrons --|          |
    anything, from moments     ->  from_moments                --'          v
                                                          condition_phase_space
                                                                            |
                                                                            v
                                                      spectra_from_phase_spaces
                                                                            |
                                                                            v
                                                            ThomsonSpectrogram

Which reader to use for a WarpX run depends on what the run wrote. A plotfile
with particles goes through `read_warpx_phase_space`, which histograms the
macroparticles itself. A run that declares a ``ParticleHistogram2D`` reduced
diagnostic has already done that binning, and `read_openpmd_phase_space` just
reads it — cheaper, ``yt``-free, and available at every timestep rather than at
the cadence full plotfiles can afford. A hybrid run has no electron
macroparticles at all, and its electrons come from
`read_warpx_hybrid_electrons`.

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

Simulations with more than one dimension
----------------------------------------

A Thomson diagnostic looks at a small volume, not a whole domain, so the readers
reduce every spatial direction that is not the diagnostic axis. The default,
``transverse_reduction="slab"``, keeps a localized region about
``transverse_position``; ``"chord"`` keeps the whole extent, for a measurement
integrated along that direction. Both **average** over the cells kept rather than
summing them, so the zeroth moment stays a number density whatever volume is
selected -- summing would silently scale the density handed to the forward model
by the number of cells combined.

Passing ``position`` reduces the diagnostic axis too, so a reader returns the
single point the probe looks at rather than a profile.

For a run that resolves more than one velocity component, the quantity the
diagnostic measures is the velocity along the scattering vector. Pass
``scatter_direction`` to `read_warpx_phase_space` and it projects onto that
vector, taking the Lorentz factor from the full momentum. Naming a single
momentum component is only right when :math:`\hat{k}` happens to lie along that
axis. OSIRIS phase spaces are projections fixed when the run was written, so
there the component is whatever ``field`` holds and the choice has to be made
when deciding which diagnostic to dump.

Letting the code bin its own phase space
----------------------------------------

WarpX's ``ParticleHistogram2D`` reduced diagnostic bins both axes from arbitrary
parser expressions of the particle state. That makes it strictly better than
anything a reader can reconstruct afterwards, because the projection onto the
scattering vector can be evaluated per particle, inside the code, with the
Lorentz factor taken from the full momentum:

.. code-block:: text

   (kx*ux + ky*uy + kz*uz) / sqrt(1 + ux*ux + uy*uy + uz*uz)

WarpX hands the parser ``ux`` as :math:`\gamma v_x / c`, so this is
:math:`\vec{v} \cdot \hat{k} / c` exactly. `histogram2d_deck_block` writes the
deck block, and returns the transverse area to hand back to
`read_openpmd_phase_space` so the densities come out right:

.. code-block:: python

   block, area = pic_thomson.histogram2d_deck_block(
       "eps_electrons",
       "electrons",
       position_range=(0, 2e-3) * u.m,
       velocity_range=(-2e8, 2e8) * u.m / u.s,
       scatter_direction=(0, 0, 1),
       transverse_slab={"y": (0.0, 50e-6)},
   )
   print(block)  # paste into the deck

.. warning::

   The generated block states ``value_function = w`` explicitly even though the
   macroparticle weight is the documented default. WarpX reads that option into
   ``m_do_parser_value`` and then never checks it, so with the option absent the
   histogram kernel calls a default-constructed — null — ``ParserExecutor``. The
   result is not an error: the histogram fills with uninitialised memory, ``NaN``
   and ``DBL_MAX`` in some bins and zero in the rest. Stating the default avoids
   it, and is harmless once the bug is fixed upstream.

Two other things about that diagnostic. Its bin edges are fixed in the deck and
particles outside them are discarded before anything is written, with no record
of how many — so unlike `read_warpx_phase_space`, the reader cannot warn about a
clipped tail. And WarpX defaults to the ``bp5`` backend wherever ADIOS2 is
compiled in; the generated block asks for ``h5`` so the output stays readable by
`h5py` alone.

Codes that carry a species as a fluid
-------------------------------------

A kinetic-ion / fluid-electron (hybrid) run has no electron macroparticles.
Its electrons are an inertialess, quasineutral fluid, and everything the run
knows about them is three mesh fields. `read_warpx_hybrid_electrons` reads them
and reconstructs the drifting Maxwellian they imply:

* :math:`n_e = \rho / (\bar{Z} e)`. Quasineutrality is how the solver *defines*
  the electron density, so with only ions carried as particles the deposited
  charge density is the electron density — exact, not an estimate.
* :math:`T_e` from the ``Te`` field when the run solves an electron energy
  equation, otherwise from the barotropic closure
  :math:`T_e = T_{e0} (n_e/n_0)^{\gamma-1}`.
* :math:`\vec{u}_e = -\vec{J}_e / (e n_e)` with
  :math:`\vec{J}_e = \nabla \times \vec{B} / \mu_0 - \vec{J}_i`, where
  :math:`\vec{J}_i` is what the ``j`` fields hold — the electrons deposit
  nothing.

In one dimension that last step is exact and free: :math:`(\nabla \times
\vec{B})_z` vanishes identically, so the total axial current is zero, the
electrons exactly counterstream the ions, and no magnetic field is read at all.

The ions are still macroparticles, so read them as usual and pass both to the
driver. `from_moments` does the reconstruction itself and is not WarpX-specific.

A Maxwellian here is inherited rather than assumed — a fluid closure carries one
scalar temperature and no higher moments, so there is nothing to build any other
shape from. Condition these phase spaces with ``taper_threshold=None``: the taper
exists to replace the discontinuity where shot noise meets the grid edge, and a
reconstructed distribution has neither.

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

Checks the readers make for you
-------------------------------

`read_warpx_phase_space` compares the macroparticles it read against
``rho_<species>``, the charge density the code itself deposited, and warns when
the two disagree. This is what catches a diagnostic written with
``random_fraction``, which subsamples the particle output *without* reweighting
— so every density built from it is low by exactly that factor, with nothing in
the file to say so. Particles and fields routinely live in separate diagnostics,
one with ``write_species = 0`` and the other with ``fields_to_plot = none``, so
point ``density_reference`` at the one carrying the fields and they are matched
up by step number.

It also counts the particles the histogram discarded for falling off the
velocity axis, and sizes that axis from every frame by default rather than from
the first and last — for a shock, what happens in between is the measurement.
Set ``velocity_scan="ends"`` to trade that second read pass for speed.

Optional dependencies
=====================

Reading WarpX plotfiles needs `yt <https://yt-project.org>`__, available as the
``pic`` extra:

.. code-block:: bash

   pip install plasmapy[pic]

The OSIRIS and openPMD readers need only `h5py`, which PlasmaPy requires anyway.
`read_warpx_hybrid_electrons` reads mesh fields from plotfiles, so it needs
``yt`` as well.

API
---

.. automodapi:: plasmapy.diagnostics.pic_thomson
