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

Where the data stops: modelling the tail
----------------------------------------

A PIC histogram is populated only as far as its last macroparticle. The EPW
satellite reads :math:`f_e` at :math:`\sqrt{\alpha^2 + 3}` thermal speeds, which
is routinely much further out — reaching :math:`n\sigma` needs of order
:math:`e^{n^2/2}` particles per cell, so three to five is typical whatever the
budget. Something has to fill the gap, and whatever fills it *is* the electron
feature.

`taper_vdf_edges`, the original filler, is a numerical device rather than a
model: it finds the outermost bin above ``threshold_frac`` of the slice peak and
runs a half-cosine from that bin's value down to zero. Every part of that is a
choice rather than a measurement, and on real data the consequences are large.
The threshold is a fraction of the *peak*, so for a Maxwellian it always lands at
3.26 :math:`\sigma` no matter how many particles were run — on one OSIRIS run
that put the anchor bin at 3.24 :math:`\sigma` while ten macroparticles only
reached 2.60, i.e. **the whole tail hung off a bin holding one or two
particles**. The shape is wrong in the derivative as well as the value, and
:math:`\partial f/\partial v` at the resonance is what sets Landau damping, so
the feature's width is fabricated along with its height. And because the rolloff
leaves a pedestal at large :math:`|v|`, where the :math:`v^2` weighting of the
second moment is largest, it sets :math:`\alpha` too: varying
``max_taper_bins`` over 5, 20, 80 and unbounded moved :math:`\alpha` from 16.0 to
0.76 on the same data. That is a factor of 21 on a reported plasma parameter,
from a knob with no physical meaning.

`extend_vdf_tail` replaces it, and is the default. Per slice and per side:

1. The **macroparticle quantum** is the smallest positive value in the raw
   histogram — one particle — so ``f / quantum`` is a particle count.
2. The **join** is the outermost bin holding at least ``min_counts`` particles,
   10 by default, a 30% counting error. Inside it the histogram is data.
3. A Maxwellian is fitted to :math:`\ln f` against :math:`(v - \bar{v})^2`
   between ``fit_from`` thermal speeds and the join, weighted by particle count,
   which is the inverse-variance weighting since :math:`\mathrm{var}(\ln f)`
   is :math:`1/N`. That gives a **tail temperature**, which is a real
   measurable and is what a fit to measured Thomson data reports.
4. Beyond the join the fit takes over, at its own intercept rather than through
   the join bin — the join is by construction the outermost bin still reaching
   ``min_counts``, hence a selected upward fluctuation, and anchoring there
   biases the tail high by 20–30%.

Below ``min_fit_bins`` in the band, or if the fit returns a non-decaying tail,
the slice falls back to its own core temperature, which is the assumption a fit
cannot improve on when there is no signal to fit.

Nothing is set to zero, so no positive floor is wanted afterwards and
`condition_phase_space` drops it — a floor would put a flat pedestal at about 12
thermal speeds, which is exactly where the satellite reads at
:math:`\alpha \sim 12`.

On a sampled Maxwellian the fitted tail temperature comes back to within 0.1% at
:math:`10^5` particles, and the extrapolated :math:`f` is right to 0.6% at 12
:math:`\sigma` — three times beyond the last particle. The uncertainty is
reported rather than assumed: ``tail_width_error`` is the standard error on
:math:`\sigma_t`, and since :math:`\delta \ln f = (x/\sigma_t)^2 \,
\delta\sigma_t/\sigma_t`, a small error in the width becomes a large one far
out. `spectra_from_phase_spaces` turns it into ``epw_tail_uncertainty``, the
factor by which the EPW amplitude is uncertain at the resonance.

Reconstructed populations — anything from `from_moments` — are skipped: they are
already their own tail model, with no last macroparticle to join at. So are
slices whose dynamic range is too large to have come from counting anything.

`taper_vdf_edges` is still there, and ``tail_model=None`` selects it, with
``max_taper_width`` to bound the rolloff in thermal speeds rather than in bins.

How much smoothing, and in what units
-------------------------------------

The collective regime needs markedly more velocity smoothing than the
non-collective one. Where :math:`\alpha > 1` the spectrum carries
:math:`|1 - \chi_e/\epsilon|^2`, which diverges as :math:`\epsilon \to 0` at the
electron-plasma-wave resonance, so shot noise entering :math:`\chi_e` through
:math:`\partial f/\partial u` is amplified into speckle. At :math:`\alpha \ll 1`
the same noise passes through untouched.

**Ask for it in thermal speeds, not in bins.** Smoothing is a convolution, so a
boxcar of :math:`W` bins applied :math:`n` times adds
:math:`n (W^2 - 1) \Delta v^2 / 12` to the second moment of every slice,
whatever that slice's own width. In velocity units that is a fixed additive
temperature, and the forward model reads the thermal speed straight off the
distribution — so it lands in :math:`T_e`, in :math:`\alpha`, and in the
satellite positions that follow from them.

How bad that is depends entirely on how wide the code's velocity grid happens to
be, which is a choice made in the input deck and has nothing to do with the
plasma. On an OSIRIS run whose momentum diagnostic spanned :math:`\pm 0.71c`
while its electrons occupied :math:`\pm 0.03c` — 31 of 1024 bins — a 40-bin
window with three passes turned a 38.8 eV ambient into 813 eV and moved
:math:`\alpha` from 12.0 to 3.2, which is a different scattering regime. The
same window cost only 18% in the shocked plasma later in the run, so the bias
does not cancel between two measurements either.

Pass ``smoothing_width`` instead, in thermal speeds, and `condition_phase_space`
sizes the window from the narrowest appreciably populated slice it is given. The
default, 0.25, leaves the second moment alone to about 2% over three passes.
`smooth_vdf` also takes a ``variance_warning``, which
`condition_phase_space` sets by default, so a window given in bins says what it
cost.

Whether the electron feature is a measurement at all
----------------------------------------------------

The EPW satellite sits at the Bohm–Gross resonance, so it reads :math:`f_e` at

.. math::

   \frac{v_\phi}{\sigma} = \sqrt{\alpha^2 + 3}

thermal speeds, with :math:`\alpha = 1/(k \lambda_{De})`. A PIC histogram is
populated only as far as its last macroparticle: reaching :math:`n \sigma` needs
of order :math:`e^{n^2/2}` particles per cell, so three to five is typical and
no amount of compute reaches twelve. Past that the conditioning takes over —
`taper_vdf_edges` rolls the tail off and the floor holds it up — and the
resonance of :math:`|1 - \chi_e/\epsilon|^2` amplifies whatever is there into a
satellite. It looks like a measurement, it tracks nothing, and it moves when
``max_taper_bins`` moves.

`spectra_from_phase_spaces` records the two numbers per timestep as
``epw_tail_ratio`` and ``epw_tail_required``, reduces them to ``epw_resolved``,
and by default replaces the EPW spectrum with `~numpy.nan` where the resonance
falls outside the sampled tail. Pass ``mask_unresolved_epw=False`` to keep those
rows. A timestep counts as resolved only when *every* present electron
population reaches the resonance, since the electron susceptibility they share
is what lights the satellite up.

For a population built by `from_moments` there is no last macroparticle: the
grid stops where ``velocity_headroom`` was told to stop, and the tail beyond the
data is the assumed Maxwellian rather than anything measured. The remedy there
is a wider grid — the default of 6 covers the resonance only up to
:math:`\alpha \approx 5.7`, and `read_warpx_hybrid_electrons` takes
``velocity_headroom`` for that reason. ``meta["epw_tail_analytic"]`` names the
populations this applies to, because for them the satellite is only ever as good
as the distribution that was assumed.

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
