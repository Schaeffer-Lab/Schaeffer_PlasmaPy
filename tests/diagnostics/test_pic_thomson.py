"""
Tests for the PIC -> Thomson scattering pipeline in
`plasmapy.diagnostics.pic_thomson`.
"""

import warnings
from dataclasses import replace

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from plasmapy.diagnostics import pic_thomson, thomson

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def maxwellian(v, sigma, drift=0.0):
    """Normalised Maxwellian with standard deviation *sigma*."""
    return np.exp(-((v - drift) ** 2) / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))


def sigma_of(T_eV, mass):
    """Standard deviation of a Maxwellian at *T_eV*, i.e. sqrt(k T / m)."""
    T = (T_eV * u.eV).to(u.J, equivalencies=u.temperature_energy())
    return float(np.sqrt(T / mass).to(u.m / u.s).value)


def block(profile, n_time=3, n_x=4):
    """Broadcast a 1-D velocity profile into a ``(n_time, n_v, n_x)`` block."""
    return np.broadcast_to(
        profile[np.newaxis, :, np.newaxis], (n_time, profile.size, n_x)
    ).copy()


def simple_phase_space(n_time=3, n_v=64, n_x=4, label="e-"):
    """A small, valid `PICPhaseSpace` for structural tests."""
    v = np.linspace(-5e6, 5e6, n_v)
    f = block(maxwellian(v, 1e6), n_time=n_time, n_x=n_x)
    return pic_thomson.from_arrays(
        f=f,
        v=v,
        x=np.linspace(0.0, 1e-3, n_x),
        t=np.linspace(0.0, 1e-9, n_time),
        label=label,
    )


def reference_taper_1d(f, threshold_frac, max_taper_bins=None):
    """
    Straightforward per-slice reference for `pic_thomson.taper_vdf_edges`.

    Deliberately written as an explicit loop so that the vectorised
    implementation is checked against something readable rather than against
    itself.
    """
    out = f.copy()
    n_v = f.size
    limit = n_v if max_taper_bins is None else max_taper_bins
    peak = f.max()
    if peak <= 0:
        return out

    above = np.nonzero(f > threshold_frac * peak)[0]
    if above.size == 0:
        return out
    i_left, i_right = int(above[0]), int(above[-1])

    length = min(i_left, limit)
    start = i_left - length
    denom = length - 1 if length > 1 else 1
    for p in range(start, i_left):
        frac = min(max((p - start) / denom, 0.0), 1.0)
        out[p] = f[i_left] * np.sin(frac * np.pi / 2)
    out[:start] = 0.0

    length = min(n_v - 1 - i_right, limit)
    end = i_right + length
    denom = length - 1 if length > 1 else 1
    for p in range(i_right + 1, end + 1):
        frac = min(max(1.0 - (p - i_right - 1) / denom, 0.0), 1.0)
        out[p] = f[i_right] * np.sin(frac * np.pi / 2)
    out[end + 1 :] = 0.0

    return out


# ---------------------------------------------------------------------------
# 1. The reader contract: from_arrays / PICPhaseSpace
# ---------------------------------------------------------------------------


class TestFromArrays:
    def test_round_trip(self):
        phase_space = simple_phase_space()
        assert phase_space.shape == (3, 64, 4)
        assert phase_space.is_electron
        assert phase_space.label == "e-"
        assert phase_space.meta == {}

    def test_accepts_quantities_and_converts_to_si(self):
        n_v, n_x, n_time = 16, 3, 2
        phase_space = pic_thomson.from_arrays(
            f=np.ones((n_time, n_v, n_x)),
            v=np.linspace(-1, 1, n_v) * u.km / u.s,
            x=np.linspace(0, 2, n_x) * u.mm,
            t=np.linspace(0, 1, n_time) * u.ns,
            label="p+",
        )
        assert phase_space.v[-1] == pytest.approx(1e3)
        assert phase_space.x[-1] == pytest.approx(2e-3)
        assert phase_space.t[-1] == pytest.approx(1e-9)
        assert not phase_space.is_electron

    def test_infers_is_electron(self):
        assert pic_thomson.from_arrays(
            np.ones((1, 4, 1)), np.arange(4.0), [0.0], [0.0], "e-"
        ).is_electron
        assert not pic_thomson.from_arrays(
            np.ones((1, 4, 1)), np.arange(4.0), [0.0], [0.0], "Al 13+"
        ).is_electron

    def test_explicit_is_electron_overrides_inference(self):
        phase_space = pic_thomson.from_arrays(
            np.ones((1, 4, 1)),
            np.arange(4.0),
            [0.0],
            [0.0],
            "e-",
            is_electron=False,
        )
        assert not phase_space.is_electron

    def test_rejects_wrong_dimensionality(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            pic_thomson.from_arrays(np.ones((4, 3)), np.arange(3.0), [0.0], [0.0], "e-")

    @pytest.mark.parametrize("bad_axis", ["t", "v", "x"])
    def test_rejects_mismatched_axis_length(self, bad_axis):
        axes = {"t": np.zeros(2), "v": np.arange(4.0), "x": np.zeros(3)}
        axes[bad_axis] = np.arange(7.0)
        with pytest.raises(ValueError, match=f"{bad_axis} has size"):
            pic_thomson.from_arrays(
                np.ones((2, 4, 3)), axes["v"], axes["x"], axes["t"], "e-"
            )

    @pytest.mark.parametrize(
        "v", [np.array([0.0, 2.0, 1.0, 3.0]), np.array([0.0, 1.0, 1.0, 2.0])]
    )
    def test_rejects_non_increasing_velocity_axis(self, v):
        with pytest.raises(ValueError, match="strictly increasing"):
            pic_thomson.from_arrays(np.ones((1, 4, 1)), v, [0.0], [0.0], "e-")

    def test_slice_position_picks_nearest(self):
        phase_space = simple_phase_space(n_x=5)
        index, sliced = phase_space.slice_position(phase_space.x[3] + 1e-9)
        assert index == 3
        assert sliced.shape == (3, 64)

    def test_slice_position_rejects_out_of_bounds(self):
        phase_space = simple_phase_space()
        with pytest.raises(ValueError, match="outside the spatial axis"):
            phase_space.slice_position(1.0 * u.m)


# ---------------------------------------------------------------------------
# 2. Normalisation -- including the per-species-axis regression
# ---------------------------------------------------------------------------


class TestNormalizeVDF:
    def test_every_slice_integrates_to_one(self):
        v = np.linspace(-4e6, 4e6, 128)
        rng = np.random.default_rng(0)
        # Deliberately varied amplitudes and widths across (t, x).
        f = np.empty((3, v.size, 4))
        for i in range(3):
            for j in range(4):
                amplitude = 10.0 ** rng.integers(-3, 4)
                f[i, :, j] = amplitude * maxwellian(
                    v, 1e6 * (1 + 0.5 * j), drift=1e5 * i
                )

        normalised = pic_thomson.normalize_vdf(f, v)
        integrals = pic_thomson.number_density(normalised, v)
        assert integrals.shape == (3, 4)
        np.testing.assert_allclose(integrals, 1.0, rtol=1e-10)

    def test_normalising_on_another_species_axis_is_wrong(self):
        """
        Regression for the osiris2thomson bug where ion VDFs were normalised
        against the *electron* velocity axis. The forward model treats these
        arrays as probability densities, so dividing by an integral taken over
        a grid 10x wider leaves the ion distribution a factor of 10 too small,
        and the ion susceptibility correspondingly underestimated.
        """
        n_v = 256
        v_electron = np.linspace(-3e8, 3e8, n_v)
        v_ion = np.linspace(-3e7, 3e7, n_v)  # 10x narrower
        f = block(maxwellian(v_ion, 5e6))

        correct = pic_thomson.normalize_vdf(f, v_ion)
        np.testing.assert_allclose(
            pic_thomson.number_density(correct, v_ion), 1.0, rtol=1e-10
        )

        wrong = pic_thomson.normalize_vdf(f, v_electron)
        np.testing.assert_allclose(
            pic_thomson.number_density(wrong, v_ion), 0.1, rtol=1e-6
        )

    def test_clips_negatives_with_a_warning(self):
        v = np.linspace(-1.0, 1.0, 32)
        f = block(maxwellian(v, 0.3))
        f[0, 0, 0] = -5.0
        with pytest.warns(RuntimeWarning, match="clipped 1 negative"):
            normalised = pic_thomson.normalize_vdf(f, v)
        assert np.all(normalised >= 0)

    def test_empty_slice_does_not_divide_by_zero(self):
        v = np.linspace(-1.0, 1.0, 32)
        f = block(maxwellian(v, 0.3))
        f[1, :, 2] = 0.0
        normalised = pic_thomson.normalize_vdf(f, v)
        assert np.all(np.isfinite(normalised))
        assert np.all(normalised[1, :, 2] == 0.0)


# ---------------------------------------------------------------------------
# 3. Tapering -- vectorised implementation vs an explicit reference loop
# ---------------------------------------------------------------------------


class TestTaperVDFEdges:
    @pytest.mark.parametrize("threshold", [0.005, 0.05, 0.2])
    @pytest.mark.parametrize("max_taper_bins", [None, 1, 2, 15, 500])
    def test_matches_reference_loop(self, threshold, max_taper_bins):
        rng = np.random.default_rng(42)
        v = np.linspace(-5.0, 5.0, 201)
        f = np.empty((4, v.size, 3))
        for i in range(4):
            for j in range(3):
                f[i, :, j] = maxwellian(
                    v, 0.4 + 0.3 * j, drift=0.5 * (i - 1.5)
                ) + 1e-4 * rng.random(v.size)

        tapered = pic_thomson.taper_vdf_edges(
            f,
            threshold_frac=threshold,
            max_taper_bins=max_taper_bins,
            pedestal_warning=None,
        )

        expected = np.empty_like(f)
        for i in range(4):
            for j in range(3):
                expected[i, :, j] = reference_taper_1d(
                    f[i, :, j], threshold, max_taper_bins
                )

        np.testing.assert_allclose(tapered, expected, rtol=0, atol=0)

    def test_zero_slice_is_untouched(self):
        v = np.linspace(-1.0, 1.0, 51)
        f = block(maxwellian(v, 0.2))
        f[0, :, 1] = 0.0
        tapered = pic_thomson.taper_vdf_edges(f, pedestal_warning=None)
        np.testing.assert_array_equal(tapered[0, :, 1], 0.0)

    def test_signal_spanning_the_grid_is_untouched(self):
        v = np.linspace(-1.0, 1.0, 51)
        f = block(np.ones_like(v))
        np.testing.assert_allclose(pic_thomson.taper_vdf_edges(f), f)

    def test_tails_are_monotonic_towards_the_edges(self):
        v = np.linspace(-5.0, 5.0, 201)
        f = block(maxwellian(v, 0.5))
        tapered = pic_thomson.taper_vdf_edges(
            f, threshold_frac=0.05, pedestal_warning=None
        )[0, :, 0]
        peak_index = int(np.argmax(tapered))
        assert np.all(np.diff(tapered[: peak_index + 1]) >= 0)
        assert np.all(np.diff(tapered[peak_index:]) <= 0)

    def test_removes_the_noise_floor_discontinuity(self):
        v = np.linspace(-5.0, 5.0, 201)
        profile = maxwellian(v, 0.4)
        profile[np.abs(v) > 2.0] = 0.0  # hard truncation, as PIC data has
        f = block(profile)
        tapered = pic_thomson.taper_vdf_edges(
            f, threshold_frac=0.005, pedestal_warning=None
        )[0, :, 0]
        jump_before = np.abs(np.diff(profile)).max()
        jump_after = np.abs(np.diff(tapered)).max()
        assert jump_after <= jump_before
        assert tapered[0] == pytest.approx(0.0)

    @pytest.mark.parametrize(
        ("half_width", "expected_growth"),
        [(6, 0.054), (8, 0.137), (12, 0.404)],
    )
    def test_unbounded_taper_inflates_width_on_a_wide_grid(
        self, half_width, expected_growth
    ):
        """
        The default rolloff runs to the grid boundary, so a distribution
        occupying a small part of the velocity grid picks up a pedestal at
        large |v| -- exactly where the v^2 weighting of the second moment is
        largest. The forward model reads the thermal speed straight off the
        VDF, so this is a direct error in alpha, not a cosmetic one.

        These numbers back the warning in the `taper_vdf_edges` docstring.
        """
        sigma = 1.0
        v = np.linspace(-half_width * sigma, half_width * sigma, 2001)
        f = block(maxwellian(v, sigma))

        with pytest.warns(RuntimeWarning, match="fabricating a pedestal"):
            tapered = pic_thomson.taper_vdf_edges(f, threshold_frac=0.005)
        normalised = pic_thomson.normalize_vdf(tapered, v)[0, :, 0]
        width = np.sqrt(np.trapezoid(normalised * v**2, v))
        np.testing.assert_allclose(width / sigma - 1.0, expected_growth, atol=5e-3)

    def test_bounded_taper_preserves_width(self):
        sigma = 1.0
        v = np.linspace(-12 * sigma, 12 * sigma, 2001)
        f = block(maxwellian(v, sigma))
        tapered = pic_thomson.taper_vdf_edges(
            f, threshold_frac=0.005, max_taper_bins=30
        )
        normalised = pic_thomson.normalize_vdf(tapered, v)[0, :, 0]
        width = np.sqrt(np.trapezoid(normalised * v**2, v))
        np.testing.assert_allclose(width, sigma, rtol=5e-3)

    def test_no_pedestal_warning_when_the_grid_fits_the_data(self):
        # Warnings are configured as errors for this test suite, so simply
        # calling through is the assertion.
        v = np.linspace(-4.0, 4.0, 2001)
        pic_thomson.taper_vdf_edges(block(maxwellian(v, 1.0)), threshold_frac=0.005)

    def test_pedestal_warning_can_be_silenced(self):
        v = np.linspace(-12.0, 12.0, 2001)
        pic_thomson.taper_vdf_edges(block(maxwellian(v, 1.0)), pedestal_warning=None)

    def test_bounded_taper_zeroes_beyond_the_rolloff(self):
        v = np.linspace(-12.0, 12.0, 1001)
        f = block(maxwellian(v, 1.0))
        tapered = pic_thomson.taper_vdf_edges(
            f, threshold_frac=0.005, max_taper_bins=10
        )[0, :, 0]
        assert tapered[0] == 0.0
        assert tapered[-1] == 0.0
        assert np.count_nonzero(tapered) < np.count_nonzero(f[0, :, 0])

    def test_rejects_negative_max_taper_bins(self):
        with pytest.raises(ValueError, match="must be non-negative"):
            pic_thomson.taper_vdf_edges(np.ones((1, 8, 1)), max_taper_bins=-1)

    def test_respects_the_axis_argument(self):
        v = np.linspace(-5.0, 5.0, 101)
        f = block(maxwellian(v, 0.5))
        along_1 = pic_thomson.taper_vdf_edges(f, axis=1, pedestal_warning=None)
        along_0 = pic_thomson.taper_vdf_edges(
            np.moveaxis(f, 1, 0), axis=0, pedestal_warning=None
        )
        np.testing.assert_allclose(np.moveaxis(along_0, 0, 1), along_1)


# ---------------------------------------------------------------------------
# 4. Smoothing, moments, rescaling, presence masks
# ---------------------------------------------------------------------------


class TestSmoothVDF:
    def test_zero_iterations_is_a_no_op(self):
        f = simple_phase_space().f
        np.testing.assert_array_equal(pic_thomson.smooth_vdf(f, 5, 0), f)

    def test_suppresses_shot_noise(self):
        rng = np.random.default_rng(7)
        v = np.linspace(-5.0, 5.0, 401)
        clean = maxwellian(v, 1.0)
        noisy = block(clean + 0.01 * rng.standard_normal(v.size))
        smoothed = pic_thomson.smooth_vdf(noisy, window=21, iterations=3)
        clean_block = block(clean)
        assert np.std(smoothed - clean_block) < np.std(noisy - clean_block)

    def test_approximately_conserves_the_integral(self):
        v = np.linspace(-5.0, 5.0, 401)
        f = block(maxwellian(v, 1.0))
        smoothed = pic_thomson.smooth_vdf(f, window=11, iterations=3)
        np.testing.assert_allclose(
            pic_thomson.number_density(smoothed, v),
            pic_thomson.number_density(f, v),
            rtol=1e-3,
        )

    def test_does_not_reflect_negatives(self):
        """
        Taking abs() -- as the original pipeline did -- turns shot noise into
        fabricated signal. Smoothing must leave the sign structure alone.
        """
        v = np.linspace(-1.0, 1.0, 32)
        f = block(np.full(v.size, -1.0))
        smoothed = pic_thomson.smooth_vdf(f, window=3, iterations=1)
        assert np.all(smoothed < 0)

    def test_cleans_non_finite_values(self):
        f = np.ones((1, 8, 1))
        f[0, 3, 0] = np.nan
        f[0, 4, 0] = np.inf
        assert np.all(np.isfinite(pic_thomson.smooth_vdf(f, 3, 1)))

    @pytest.mark.parametrize(("window", "iterations"), [(0, 1), (-1, 1), (3, -1)])
    def test_rejects_invalid_arguments(self, window, iterations):
        with pytest.raises(ValueError):
            pic_thomson.smooth_vdf(np.ones((1, 4, 1)), window, iterations)


class TestNumberDensity:
    def test_matches_analytic_integral(self):
        v = np.linspace(-8.0, 8.0, 2001)
        f = block(3.5 * maxwellian(v, 1.0))
        np.testing.assert_allclose(pic_thomson.number_density(f, v), 3.5, rtol=1e-6)

    def test_collapses_the_velocity_axis(self):
        phase_space = simple_phase_space(n_time=3, n_v=64, n_x=4)
        assert pic_thomson.number_density(phase_space.f, phase_space.v).shape == (3, 4)


class TestRescaleVelocityAxis:
    def test_preserves_the_integral(self):
        v = np.linspace(-4e7, 4e7, 1024)
        f = pic_thomson.normalize_vdf(block(maxwellian(v, 4e6)), v)
        v_out, f_out = pic_thomson.rescale_velocity_axis(f, v, scale=np.sqrt(50))
        np.testing.assert_allclose(
            pic_thomson.number_density(f_out, v_out), 1.0, rtol=1e-4
        )

    def test_compresses_the_distribution_width(self):
        v = np.linspace(-4e7, 4e7, 1024)
        sigma = 4e6
        f = pic_thomson.normalize_vdf(block(maxwellian(v, sigma)), v)
        scale = np.sqrt(50)
        v_out, f_out = pic_thomson.rescale_velocity_axis(f, v, scale=scale)

        second_moment = np.trapezoid(f_out[0, :, 0] * v_out**2, v_out)
        np.testing.assert_allclose(np.sqrt(second_moment), sigma / scale, rtol=1e-3)

    def test_scale_of_one_is_a_faithful_resampling(self):
        v = np.linspace(-4e7, 4e7, 512)
        f = block(maxwellian(v, 4e6))
        v_out, f_out = pic_thomson.rescale_velocity_axis(f, v, scale=1.0, target_v=v)
        np.testing.assert_allclose(v_out, v)
        np.testing.assert_allclose(f_out, f, rtol=1e-12)

    def test_default_target_axis_oversamples(self):
        v = np.linspace(-1.0, 1.0, 100)
        f = block(maxwellian(v, 0.2))
        v_out, f_out = pic_thomson.rescale_velocity_axis(f, v, scale=7.1)
        assert v_out.size == 8 * 100
        assert f_out.shape[1] == 8 * 100

    def test_zero_pads_outside_the_compressed_domain(self):
        v = np.linspace(-1.0, 1.0, 128)
        f = block(maxwellian(v, 0.1))
        v_out, f_out = pic_thomson.rescale_velocity_axis(f, v, scale=4.0, target_v=v)
        outside = np.abs(v_out) > 0.25 + 1e-12
        np.testing.assert_array_equal(f_out[0, outside, 0], 0.0)

    def test_rejects_non_positive_scale(self):
        with pytest.raises(ValueError, match="must be positive"):
            pic_thomson.rescale_velocity_axis(
                np.ones((1, 4, 1)), np.arange(4.0), scale=0.0
            )


class TestSpeciesPresenceMask:
    def test_thresholds_against_the_reference_density(self):
        density = np.array([1e18, 1e15, 5e16, 0.0])
        mask = pic_thomson.species_presence_mask(density, 1e18, threshold=1e-2)
        np.testing.assert_array_equal(mask, [True, False, True, False])


# ---------------------------------------------------------------------------
# 5. The conditioning pipeline as a whole
# ---------------------------------------------------------------------------


class TestConditionPhaseSpace:
    def test_output_is_normalised_and_floored(self):
        phase_space = simple_phase_space(n_v=256)
        conditioned = pic_thomson.condition_phase_space(
            phase_space,
            smoothing_window=5,
            smoothing_iterations=2,
            pedestal_warning=None,
        )
        np.testing.assert_allclose(
            pic_thomson.number_density(conditioned.f, conditioned.v),
            1.0,
            rtol=1e-6,
        )
        assert np.all(conditioned.f >= pic_thomson.DEFAULT_FLOOR)

    def test_does_not_mutate_the_input(self):
        phase_space = simple_phase_space()
        before = phase_space.f.copy()
        pic_thomson.condition_phase_space(
            phase_space,
            smoothing_window=5,
            smoothing_iterations=2,
            pedestal_warning=None,
        )
        np.testing.assert_array_equal(phase_space.f, before)

    def test_records_provenance(self):
        conditioned = pic_thomson.condition_phase_space(
            simple_phase_space(),
            smoothing_iterations=3,
            velocity_scale_factor=50,
            pedestal_warning=None,
        )
        conditioning = conditioned.meta["conditioning"]
        assert conditioning["smoothing_iterations"] == 3
        assert conditioning["velocity_scale_factor"] == 50

    def test_velocity_scale_factor_divides_by_sqrt_r(self):
        v = np.linspace(-4e7, 4e7, 512)
        sigma = 4e6
        phase_space = pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma)),
            v=v,
            x=np.zeros(4),
            t=np.arange(3.0),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(
            phase_space, taper_threshold=None, velocity_scale_factor=50
        )
        second_moment = np.trapezoid(
            conditioned.f[0, :, 0] * conditioned.v**2, conditioned.v
        )
        np.testing.assert_allclose(
            np.sqrt(second_moment), sigma / np.sqrt(50), rtol=1e-3
        )

    def test_preserves_a_maxwellian(self):
        """
        Conditioning must not distort the physics: an analytic Maxwellian in,
        the same Maxwellian out.

        The taper threshold is set well below the default here. At the default
        0.005 the distribution is cut at 3.3 sigma, which discards ~1.4% of the
        variance -- inherent to tapering at that level, not a defect, but it
        swamps the check this test is making.
        """
        sigma, drift = 5e6, 2.5e6
        v = np.linspace(drift - 6 * sigma, drift + 6 * sigma, 2048)
        phase_space = pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma, drift)),
            v=v,
            x=np.zeros(4),
            t=np.arange(3.0),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(
            phase_space,
            smoothing_window=5,
            smoothing_iterations=1,
            taper_threshold=1e-4,
            max_taper_bins=30,
        )
        f = conditioned.f[0, :, 0]

        recovered_drift = np.trapezoid(f * v, v)
        recovered_sigma = np.sqrt(np.trapezoid(f * (v - recovered_drift) ** 2, v))
        np.testing.assert_allclose(recovered_drift, drift, rtol=1e-3)
        np.testing.assert_allclose(recovered_sigma, sigma, rtol=1e-3)

    def test_default_settings_warn_and_inflate_width_on_a_wide_grid(self):
        """
        Companion to the test above: with the default taper on a 6-sigma grid
        the recovered width comes out ~5% high, and the user is told.
        """
        sigma = 5e6
        v = np.linspace(-6 * sigma, 6 * sigma, 2048)
        phase_space = pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma)),
            v=v,
            x=np.zeros(4),
            t=np.arange(3.0),
            label="e-",
        )
        with pytest.warns(RuntimeWarning, match="fabricating a pedestal"):
            conditioned = pic_thomson.condition_phase_space(phase_space)
        f = conditioned.f[0, :, 0]
        recovered_sigma = np.sqrt(np.trapezoid(f * v**2, v))
        np.testing.assert_allclose(recovered_sigma / sigma - 1.0, 0.054, atol=5e-3)


# ---------------------------------------------------------------------------
# 6. End-to-end physics check against the Maxwellian forward model
# ---------------------------------------------------------------------------


N_E = 5e18 * u.cm**-3
T_E = 100 * u.eV
T_I = 50 * u.eV
PROBE_WAVELENGTH = 532 * u.nm
PROBE_VEC = np.array([1.0, 0.0, 0.0])
SCATTER_VEC = np.array([0.0, 1.0, 0.0])
WAVELENGTHS = np.linspace(480, 590, 351) * u.nm


@pytest.fixture(scope="module")
def arbitrary_result():
    """Spectrum from a conditioned analytic Maxwellian, via the VDF model."""
    pytest.importorskip("numba")
    sigma_e = sigma_of(T_E.value, const.m_e)
    sigma_i = sigma_of(T_I.value, const.m_p)

    v_e = np.linspace(-8 * sigma_e, 8 * sigma_e, 4001)
    v_i = np.linspace(-8 * sigma_i, 8 * sigma_i, 4001)

    electrons = pic_thomson.from_arrays(
        f=block(maxwellian(v_e, sigma_e), n_time=2, n_x=2),
        v=v_e,
        x=np.zeros(2),
        t=np.arange(2.0),
        label="e-",
    )
    ions = pic_thomson.from_arrays(
        f=block(maxwellian(v_i, sigma_i), n_time=2, n_x=2),
        v=v_i,
        x=np.zeros(2),
        t=np.arange(2.0),
        label="p+",
    )
    # A low taper threshold and a bounded rolloff, so that this test measures
    # the pipeline rather than the truncation the default taper applies to a
    # distribution sitting on a grid much wider than itself. See
    # TestTaperVDFEdges for that effect on its own.
    conditioning = {"taper_threshold": 1e-4, "max_taper_bins": 50}
    electrons = pic_thomson.condition_phase_space(electrons, **conditioning)
    ions = pic_thomson.condition_phase_space(ions, **conditioning)

    # Some environments ship a `numba_scipy` build that is incompatible with
    # their scipy; numba emits a UserWarning about it the first time it JITs
    # anything. It is unrelated to this test, and this suite turns warnings
    # into errors, so filter that one message only.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*numba_scipy.*")
        return thomson.arbitrary_forwardmodel(
            wavelengths=WAVELENGTHS,
            probe_wavelength=PROBE_WAVELENGTH,
            e_velocity_axes=[electrons.v] * u.m / u.s,
            i_velocity_axes=[ions.v] * u.m / u.s,
            efn=[electrons.f[0, :, 0]] * u.s / u.m,
            ifn=[ions.f[0, :, 0]] * u.s / u.m,
            n=N_E,
            ion_species=["p+"],
            probe_vec=PROBE_VEC,
            scatter_vec=SCATTER_VEC,
        )


@pytest.fixture(scope="module")
def maxwellian_result():
    """The same plasma, through PlasmaPy's analytic Maxwellian model."""
    return thomson.spectral_density(
        WAVELENGTHS,
        PROBE_WAVELENGTH,
        N_E,
        T_e=T_E,
        T_i=T_I,
        ions=["p+"],
        probe_vec=PROBE_VEC,
        scatter_vec=SCATTER_VEC,
    )


def as_array(spectrum):
    """Strip units and tensor wrappers from a returned spectrum."""
    if isinstance(spectrum, u.Quantity):
        spectrum = spectrum.value
    return np.asarray(spectrum, dtype=float)


class TestMaxwellianConsistency:
    """
    Push an analytic Maxwellian through the conditioning pipeline and into
    `thomson.arbitrary_forwardmodel`, then compare against
    `thomson.spectral_density`, which solves the same problem analytically for
    a Maxwellian plasma. This is the check that the conditioning steps do not
    distort the physics, and it is independent of any PIC code.
    """

    def test_spectra_agree(self, arbitrary_result, maxwellian_result):
        skw_arbitrary = as_array(arbitrary_result[1])
        skw_maxwellian = as_array(maxwellian_result[1])
        wavelengths = WAVELENGTHS.to(u.m).value

        # `arbitrary_forwardmodel` normalises S(k, w) to unit area over the
        # requested window; `spectral_density` does not. Compare shapes.
        skw_arbitrary = skw_arbitrary / np.trapezoid(skw_arbitrary, wavelengths)
        skw_maxwellian = skw_maxwellian / np.trapezoid(skw_maxwellian, wavelengths)

        l1_error = np.trapezoid(np.abs(skw_arbitrary - skw_maxwellian), wavelengths)
        assert l1_error < 0.05, f"normalised L1 difference {l1_error:.3f}"

    def test_epw_peaks_agree(self, arbitrary_result, maxwellian_result):
        skw_arbitrary = as_array(arbitrary_result[1])
        skw_maxwellian = as_array(maxwellian_result[1])
        wavelengths = WAVELENGTHS.to(u.nm).value

        # Ignore the central IAW region; compare the two EPW satellites.
        for side in (wavelengths < 520, wavelengths > 545):
            peak_arbitrary = wavelengths[side][np.argmax(skw_arbitrary[side])]
            peak_maxwellian = wavelengths[side][np.argmax(skw_maxwellian[side])]
            assert abs(peak_arbitrary - peak_maxwellian) < 1.0

    def test_alpha_conventions_differ_by_sqrt_two(
        self, arbitrary_result, maxwellian_result
    ):
        """
        The two models report different scattering parameters for the same
        plasma. `spectral_density` uses alpha = 1 / (k lambda_De), while
        `arbitrary_forwardmodel` computes sqrt(2) * wpe / (k * sigma) with
        sigma the standard deviation of the VDF, which is sqrt(2) larger.
        Pinning the ratio here documents the convention so it is not mistaken
        for a physics discrepancy.
        """
        alpha_arbitrary = float(np.asarray(arbitrary_result[0]))
        alpha_maxwellian = float(np.mean(np.asarray(maxwellian_result[0])))
        np.testing.assert_allclose(
            alpha_arbitrary / alpha_maxwellian, np.sqrt(2), rtol=2e-2
        )


# ---------------------------------------------------------------------------
# 7. The forward-model driver
# ---------------------------------------------------------------------------

REFERENCE_DENSITY = 5e18 * u.cm**-3
EPW_WINDOW = np.linspace(480, 590, 141) * u.nm
IAW_WINDOW = np.linspace(529, 535, 61) * u.nm
DRIVER_CONDITIONING = {"taper_threshold": 1e-4, "max_taper_bins": 50}


def species_phase_space(
    label, T_eV, mass, *, n_time=3, n_x=4, n_v=1024, amplitude=1.0, drift=0.0
):
    """
    A Maxwellian `PICPhaseSpace` whose zeroth moment is *amplitude*, so that
    `spectra_from_phase_spaces` reads its density as amplitude * reference.
    """
    sigma = sigma_of(T_eV, mass)
    v = np.linspace(drift - 8 * sigma, drift + 8 * sigma, n_v)
    profile = amplitude * maxwellian(v, sigma, drift)
    return pic_thomson.from_arrays(
        f=block(profile, n_time=n_time, n_x=n_x),
        v=v,
        x=np.linspace(0.0, 1e-3, n_x),
        t=np.linspace(0.0, 2e-9, n_time),
        label=label,
    )


def run_driver(electrons=None, ions=None, **kwargs):
    """`spectra_from_phase_spaces` with the shared test geometry filled in."""
    if electrons is None:
        electrons = species_phase_space("e-", 100, const.m_e)
    if ions is None:
        ions = [species_phase_space("p+", 50, const.m_p)]
    settings = {
        "position": 0.5e-3,
        "reference_density": REFERENCE_DENSITY,
        "probe_wavelength": PROBE_WAVELENGTH,
        "epw_wavelengths": EPW_WINDOW,
        "probe_vec": PROBE_VEC,
        "scatter_vec": SCATTER_VEC,
        "electron_conditioning": DRIVER_CONDITIONING,
        "ion_conditioning": DRIVER_CONDITIONING,
        "progress": False,
    }
    settings.update(kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*numba_scipy.*")
        return pic_thomson.spectra_from_phase_spaces(electrons, ions, **settings)


@pytest.fixture(scope="module")
def single_population_spectrogram():
    pytest.importorskip("numba")
    return run_driver(iaw_wavelengths=IAW_WINDOW)


class TestDriverStructure:
    def test_shapes_and_labels(self, single_population_spectrogram):
        spectrogram = single_population_spectrogram
        assert spectrogram.n_time == 3
        assert spectrogram.epw.shape == (3, EPW_WINDOW.size)
        assert spectrogram.iaw.shape == (3, IAW_WINDOW.size)
        assert spectrogram.electron_labels == ["e-"]
        assert spectrogram.ion_labels == ["p+"]
        assert spectrogram.efract.shape == (1, 3)
        assert spectrogram.ifract.shape == (1, 3)

    def test_wavelength_axes_are_si(self, single_population_spectrogram):
        np.testing.assert_allclose(
            single_population_spectrogram.epw_wavelengths,
            EPW_WINDOW.to_value(u.m),
        )
        np.testing.assert_allclose(
            single_population_spectrogram.iaw_wavelengths,
            IAW_WINDOW.to_value(u.m),
        )

    def test_spectra_are_finite_and_positive(self, single_population_spectrogram):
        spectrogram = single_population_spectrogram
        assert np.all(np.isfinite(spectrogram.epw))
        assert np.all(np.isfinite(spectrogram.iaw))
        assert np.all(np.isfinite(spectrogram.alpha_epw))
        assert spectrogram.epw.max() > 0

    def test_identical_timesteps_give_identical_spectra(
        self, single_population_spectrogram
    ):
        epw = single_population_spectrogram.epw
        np.testing.assert_allclose(epw[0], epw[1], rtol=1e-12)
        np.testing.assert_allclose(epw[0], epw[2], rtol=1e-12)

    def test_density_uses_the_reference_scale(self, single_population_spectrogram):
        # The test VDFs integrate to 1, so the density is the reference density.
        np.testing.assert_allclose(
            single_population_spectrogram.electron_density,
            REFERENCE_DENSITY.to_value(u.m**-3),
            rtol=1e-6,
        )

    def test_skipping_the_iaw_window(self):
        pytest.importorskip("numba")
        spectrogram = run_driver()
        assert spectrogram.iaw is None
        assert spectrogram.iaw_wavelengths is None
        assert spectrogram.alpha_iaw is None

    def test_repr_is_informative(self, single_population_spectrogram):
        text = repr(single_population_spectrogram)
        assert "e-" in text
        assert "p+" in text
        assert "n_time=3" in text

    def test_records_provenance(self, single_population_spectrogram):
        meta = single_population_spectrogram.meta
        assert meta["reference_density"] == REFERENCE_DENSITY.to_value(u.m**-3)
        assert meta["presence_threshold"] == 1e-2
        assert set(meta["species_meta"]) == {"e-", "p+"}


class TestDriverValidation:
    def test_rejects_empty_ion_list(self):
        with pytest.raises(ValueError, match="must contain at least one"):
            run_driver(ions=[])

    def test_rejects_mismatched_time_axis_length(self):
        with pytest.raises(ValueError, match="timesteps"):
            run_driver(ions=[species_phase_space("p+", 50, const.m_p, n_time=5)])

    def test_rejects_differing_time_values(self):
        ion = species_phase_space("p+", 50, const.m_p)
        shifted = pic_thomson.from_arrays(
            f=ion.f, v=ion.v, x=ion.x, t=ion.t + 1e-9, label="p+"
        )
        with pytest.raises(ValueError, match="different time axes"):
            run_driver(ions=[shifted])

    def test_rejects_non_phase_space_input(self):
        with pytest.raises(TypeError, match="PICPhaseSpace objects"):
            run_driver(ions=[np.zeros((3, 4, 5))])

    def test_rejects_non_positive_reference_density(self):
        with pytest.raises(ValueError, match="reference_density must be positive"):
            run_driver(reference_density=0.0)

    def test_rejects_position_outside_the_grid(self):
        with pytest.raises(ValueError, match="outside the spatial axis"):
            run_driver(position=1.0 * u.m)

    def test_rejects_velocity_scale_factor_in_conditioning(self):
        with pytest.raises(ValueError, match="must be the same for every species"):
            run_driver(
                electron_conditioning={"velocity_scale_factor": 50},
            )

    def test_warns_when_an_ion_is_passed_as_an_electron(self):
        with pytest.warns(RuntimeWarning, match="not flagged as one"):
            run_driver(electrons=species_phase_space("p+", 100, const.m_p))

    @pytest.mark.parametrize("notches", [np.zeros((2, 3)), np.zeros(3)])
    def test_rejects_malformed_notches(self, notches):
        with pytest.raises(ValueError, match="sequence of pairs"):
            run_driver(epw_notches=notches * u.nm)

    def test_conditioning_can_be_skipped(self):
        pytest.importorskip("numba")
        electrons = pic_thomson.condition_phase_space(
            species_phase_space("e-", 100, const.m_e), **DRIVER_CONDITIONING
        )
        ions = [
            pic_thomson.condition_phase_space(
                species_phase_space("p+", 50, const.m_p), **DRIVER_CONDITIONING
            )
        ]
        skipped = run_driver(
            electrons=electrons,
            ions=ions,
            electron_conditioning={"skip": True},
            ion_conditioning={"skip": True},
        )
        conditioned = run_driver()
        np.testing.assert_allclose(skipped.epw, conditioned.epw, rtol=1e-12)


class TestDriverPopulations:
    def test_fractions_reflect_relative_densities(self):
        pytest.importorskip("numba")
        spectrogram = run_driver(
            ions=[
                species_phase_space("p+", 50, const.m_p, amplitude=0.75),
                species_phase_space("He-4 2+", 50, 4 * const.m_p, amplitude=0.25),
            ]
        )
        np.testing.assert_allclose(spectrogram.ifract[:, 0], [0.75, 0.25], rtol=1e-6)
        assert spectrogram.ion_labels == ["p+", "He-4 2+"]

    def test_two_electron_populations_use_efract(self):
        """
        The WarpX runs carry a piston and an ambient electron population. Two
        identical populations at half weight each must give the same spectrum
        as one population at full weight.
        """
        pytest.importorskip("numba")
        single = run_driver()
        split = run_driver(
            electrons=[
                species_phase_space("e-", 100, const.m_e, amplitude=0.5),
                species_phase_space("e-", 100, const.m_e, amplitude=0.5),
            ]
        )
        np.testing.assert_allclose(split.efract[:, 0], [0.5, 0.5], rtol=1e-6)
        np.testing.assert_allclose(
            split.electron_density, single.electron_density, rtol=1e-6
        )
        np.testing.assert_allclose(split.epw, single.epw, rtol=1e-8)

    def test_absent_population_is_excluded_and_fractions_renormalise(self):
        """
        A trace species must neither contribute nor leave the remaining
        fractions summing to less than one -- the forward model uses them to
        split the density between populations.
        """
        pytest.importorskip("numba")
        trace = species_phase_space("He-4 2+", 50, 4 * const.m_p, amplitude=1e-6)
        with_trace = run_driver(ions=[species_phase_space("p+", 50, const.m_p), trace])
        without_trace = run_driver()

        assert not with_trace.ion_present[1].any()
        assert with_trace.ion_present[0].all()
        np.testing.assert_allclose(with_trace.epw, without_trace.epw, rtol=1e-8)

    def test_vacuum_timesteps_are_nan(self):
        pytest.importorskip("numba")
        electrons = species_phase_space("e-", 100, const.m_e)
        empty = electrons.f.copy()
        empty[1] = 0.0  # no plasma at all at the second timestep
        electrons = pic_thomson.from_arrays(
            f=empty, v=electrons.v, x=electrons.x, t=electrons.t, label="e-"
        )
        spectrogram = run_driver(electrons=electrons)

        assert np.all(np.isnan(spectrogram.epw[1]))
        assert np.isnan(spectrogram.alpha_epw[1])
        assert np.all(np.isfinite(spectrogram.epw[0]))
        assert np.all(np.isfinite(spectrogram.epw[2]))
        np.testing.assert_array_equal(
            spectrogram.electron_present[0], [True, False, True]
        )

    def test_presence_is_judged_before_conditioning(self):
        """
        The floor that conditioning applies would make an empty slice look
        populated, so the presence test must run on the raw phase space.
        """
        pytest.importorskip("numba")
        electrons = species_phase_space("e-", 100, const.m_e)
        empty = electrons.f.copy()
        empty[0] = 0.0
        electrons = pic_thomson.from_arrays(
            f=empty, v=electrons.v, x=electrons.x, t=electrons.t, label="e-"
        )
        spectrogram = run_driver(electrons=electrons)
        assert np.isnan(spectrogram.alpha_epw[0])


class TestDriverPhysics:
    def test_matches_the_analytic_maxwellian_model(self):
        """
        End-to-end: the driver's spectrum for a Maxwellian plasma must agree
        with `thomson.spectral_density` for the same plasma.

        Compared with ``scattered_power=False``, since `spectral_density`
        returns S(k, omega) while the driver's default returns power per unit
        wavelength. The conversion between them is checked separately below.
        """
        pytest.importorskip("numba")
        spectrogram = run_driver(scattered_power=False)
        _, analytic = thomson.spectral_density(
            EPW_WINDOW,
            PROBE_WAVELENGTH,
            REFERENCE_DENSITY,
            T_e=100 * u.eV,
            T_i=50 * u.eV,
            ions=["p+"],
            probe_vec=PROBE_VEC,
            scatter_vec=SCATTER_VEC,
        )
        wavelengths = EPW_WINDOW.to_value(u.m)
        driver = spectrogram.epw[0] / np.trapezoid(spectrogram.epw[0], wavelengths)
        analytic = as_array(analytic)
        analytic = analytic / np.trapezoid(analytic, wavelengths)

        l1_error = np.trapezoid(np.abs(driver - analytic), wavelengths)
        assert l1_error < 0.05, f"normalised L1 difference {l1_error:.3f}"

    def test_scattered_power_applies_the_wavelength_jacobian(self):
        """
        ``scattered_power=True`` -- the default -- reports power per unit
        wavelength rather than S(k, omega). The two differ by the Jacobian of
        the frequency-to-wavelength change of variables, and each is separately
        renormalised to unit area.
        """
        pytest.importorskip("numba")
        wavelengths = EPW_WINDOW.to_value(u.m)
        spectral_density = run_driver(scattered_power=False).epw[0]
        power = run_driver(scattered_power=True).epw[0]

        speed_of_light = const.c.si.value
        probe_frequency = 2 * np.pi * speed_of_light / PROBE_WAVELENGTH.to_value(u.m)
        shift = 2 * np.pi * speed_of_light / wavelengths - probe_frequency
        jacobian = (1 + 2 * shift / probe_frequency) * 2 / wavelengths**2

        expected = spectral_density * jacobian
        expected = expected / np.trapezoid(expected, wavelengths)
        power = power / np.trapezoid(power, wavelengths)
        np.testing.assert_allclose(expected, power, rtol=1e-10)

    def test_notch_zeroes_the_requested_band(self):
        pytest.importorskip("numba")
        spectrogram = run_driver(epw_notches=[520, 545] * u.nm)
        wavelengths = spectrogram.epw_wavelengths * 1e9
        notched = (wavelengths > 521) & (wavelengths < 544)
        np.testing.assert_allclose(spectrogram.epw[0][notched], 0.0, atol=1e-12)
        assert spectrogram.epw[0][~notched].max() > 0

    def test_velocity_scale_factor_shifts_the_epw_feature(self):
        """
        Undoing a reduced mass ratio narrows every velocity distribution, which
        lowers the electron thermal speed and pulls the EPW satellites in
        towards the probe wavelength.
        """
        pytest.importorskip("numba")
        wavelengths = EPW_WINDOW.to_value(u.nm)
        red = wavelengths > 545

        uncorrected = run_driver().epw[0]
        corrected = run_driver(velocity_scale_factor=9.0).epw[0]

        peak_uncorrected = wavelengths[red][np.argmax(uncorrected[red])]
        peak_corrected = wavelengths[red][np.argmax(corrected[red])]
        assert peak_corrected < peak_uncorrected


# ---------------------------------------------------------------------------
# 8. The OSIRIS reader
# ---------------------------------------------------------------------------

OSIRIS_REFERENCE_DENSITY = 9e17 * u.cm**-3


def write_osiris_file(directory, field, species, dump, data, *, axes, time):
    """
    Write one file in OSIRIS phase-space layout.

    ``axes`` lists ``(name, min, max)`` in **numpy** axis order; the writer
    reverses them into the Fortran-ordered ``AXIS1..N`` that OSIRIS emits, which
    is exactly the convention the reader has to undo.
    """
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{field}-{species}-{dump:06d}.h5"
    ndim = data.ndim
    with h5py.File(path, "w") as handle:
        handle.create_dataset(field, data=np.asarray(data, dtype=np.float32))
        group = handle.create_group("AXIS")
        for k, (name, lo, hi) in enumerate(axes):
            node = group.create_dataset(
                f"AXIS{ndim - k}", data=np.array([lo, hi], dtype=np.float64)
            )
            node.attrs["NAME"] = np.array([name.encode()], dtype="S256")
            node.attrs["UNITS"] = np.array([b"a.u."], dtype="S256")
        handle.attrs["TIME"] = np.array([time], dtype=np.float64)
        handle.attrs["ITER"] = np.array([dump * 10], dtype=np.int32)
    return path


def make_osiris_run(
    tmp_path,
    *,
    field="p1x1",
    species="e",
    n_p=64,
    n_x=8,
    n_dumps=3,
    u_max=1.0,
    x_max=100.0,
    sign=-1.0,
    transverse=None,
):
    """Build a small OSIRIS ``MS`` tree and return its path plus the raw blocks."""
    directory = tmp_path / "MS" / "PHA" / field / species
    u_axis = np.linspace(-u_max, u_max, n_p)
    blocks = []
    for dump in range(n_dumps):
        profile = np.exp(-(u_axis**2) / (2 * 0.2**2)) * (1 + 0.1 * dump)
        if transverse is None:
            data = sign * np.outer(profile, np.ones(n_x))
            axes = [("p1", -u_max, u_max), ("x1", 0.0, x_max)]
        else:
            data = sign * np.einsum(
                "p,x,y->pxy", profile, np.ones(n_x), np.ones(transverse)
            )
            axes = [
                ("p1", -u_max, u_max),
                ("x1", 0.0, x_max),
                ("x2", 0.0, 10.0),
            ]
        write_osiris_file(
            directory, field, species, dump, data, axes=axes, time=100.0 * dump
        )
        blocks.append(np.abs(data))
    return tmp_path / "MS", u_axis, blocks


class TestOsirisReader:
    def test_axis_order_and_shapes(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, n_p=64, n_x=8, n_dumps=3)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        # The file stores AXIS1 = x1 and AXIS2 = p1, so numpy axis 0 is momentum.
        assert phase_space.shape == (3, 64, 8)
        assert phase_space.label == "e-"
        assert phase_space.meta["spatial_axis"] == "x1"

    def test_velocity_axis_is_the_relativistic_map(self, tmp_path):
        ms, u_axis, _ = make_osiris_run(tmp_path, u_max=1.0)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        expected = u_axis * const.c.si.value / np.sqrt(1 + u_axis**2)
        np.testing.assert_allclose(phase_space.v, expected, rtol=1e-12)
        # u = 1 is 0.707 c, not c -- the check that catches a missing gamma.
        assert phase_space.v[-1] == pytest.approx(
            const.c.si.value / np.sqrt(2), rel=1e-12
        )

    def test_spatial_and_time_axes_are_converted_to_si(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, n_x=8, x_max=100.0, n_dumps=3)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        density = OSIRIS_REFERENCE_DENSITY.to_value(u.m**-3)
        omega_p = np.sqrt(
            density * const.e.si.value**2 / (const.eps0.si.value * const.m_e.si.value)
        )
        skin_depth = const.c.si.value / omega_p

        np.testing.assert_allclose(phase_space.x[-1], 100.0 * skin_depth, rtol=1e-12)
        np.testing.assert_allclose(
            phase_space.t, [0.0, 100.0, 200.0] / omega_p, rtol=1e-12
        )
        np.testing.assert_allclose(
            phase_space.meta["plasma_frequency"], omega_p, rtol=1e-12
        )

    def test_electron_charge_density_sign_is_removed(self, tmp_path):
        """
        OSIRIS writes electron phase space as a negative charge density. Left
        alone it would be clipped away entirely by `normalize_vdf`.
        """
        ms, _, _ = make_osiris_run(tmp_path, sign=-1.0)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        assert np.all(phase_space.f >= 0)
        assert phase_space.f.max() > 0

    def test_charge_weighting_divides_by_the_charge_number(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, species="al", sign=1.0)
        weighted = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "al",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            label="Al 13+",
            progress=False,
        )
        unweighted = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "al",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            label="Al 13+",
            charge_weighted=False,
            progress=False,
        )
        np.testing.assert_allclose(weighted.f * 13, unweighted.f, rtol=1e-12)

    def test_jacobian_preserves_the_momentum_space_integral(self, tmp_path):
        r"""
        With the :math:`\gamma^3 / c` Jacobian applied, integrating over velocity
        reproduces the integral over proper velocity -- so the zeroth moment is
        still the density in units of the reference density.
        """
        ms, u_axis, blocks = make_osiris_run(tmp_path, n_p=512, u_max=1.0)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        in_u = np.trapezoid(blocks[0][:, 0], u_axis)
        in_v = pic_thomson.number_density(phase_space.f, phase_space.v)[0, 0]
        np.testing.assert_allclose(in_v, in_u, rtol=1e-3)

    def test_jacobian_can_be_disabled(self, tmp_path):
        ms, u_axis, _ = make_osiris_run(tmp_path)
        with_jacobian = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        without = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            relativistic_jacobian=False,
            progress=False,
        )
        gamma = np.sqrt(1 + u_axis**2)
        ratio = with_jacobian.f[0, :, 0] / without.f[0, :, 0]
        np.testing.assert_allclose(ratio, gamma**3 / const.c.si.value, rtol=1e-12)

    def test_transverse_spatial_axis_is_summed(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=32, n_x=8, transverse=4
        )
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1x2",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        assert phase_space.shape == (3, 32, 8)
        assert phase_space.meta["summed_axes"] == ["x2"]

    def test_can_keep_the_other_spatial_axis(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=32, n_x=8, transverse=4
        )
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1x2",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            spatial_axis="x2",
            progress=False,
        )
        assert phase_space.shape == (3, 32, 4)
        assert phase_space.meta["summed_axes"] == ["x1"]

    def test_selects_requested_dumps(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, n_dumps=5)
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            timesteps=[0, 2, 4],
            progress=False,
        )
        assert phase_space.shape[0] == 3
        assert phase_space.meta["dumps"] == [0, 2, 4]

    def test_rejects_a_missing_dump(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, n_dumps=3)
        with pytest.raises(FileNotFoundError, match="dump 9 not found"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1",
                "e",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                is_electron=True,
                timesteps=[0, 9],
                progress=False,
            )

    def test_rejects_a_missing_species_directory(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path)
        with pytest.raises(FileNotFoundError, match="no OSIRIS phase-space"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1",
                "nonexistent",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                is_electron=True,
                progress=False,
            )

    def test_requires_a_label_for_ions(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, species="al", sign=1.0)
        with pytest.raises(ValueError, match="species label is required"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1",
                "al",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                progress=False,
            )

    def test_rejects_an_unknown_spatial_axis(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path)
        with pytest.raises(ValueError, match="no spatial axis 'x7'"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1",
                "e",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                is_electron=True,
                spatial_axis="x7",
                progress=False,
            )

    def test_rejects_non_positive_reference_density(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path)
        with pytest.raises(ValueError, match="reference_density must be positive"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1",
                "e",
                reference_density=0 * u.cm**-3,
                is_electron=True,
                progress=False,
            )

    def test_output_feeds_the_driver(self, tmp_path):
        """The reader's output must be directly usable by the driver."""
        pytest.importorskip("numba")
        ms, _, _ = make_osiris_run(tmp_path, n_p=256, n_x=8, n_dumps=2)
        make_osiris_run(tmp_path, species="al", n_p=256, n_x=8, n_dumps=2, sign=1.0)
        electrons = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            progress=False,
        )
        ions = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1",
            "al",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            label="Al 13+",
            progress=False,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            spectrogram = pic_thomson.spectra_from_phase_spaces(
                electrons,
                [ions],
                position=electrons.x[4],
                reference_density=OSIRIS_REFERENCE_DENSITY,
                probe_wavelength=PROBE_WAVELENGTH,
                epw_wavelengths=EPW_WINDOW,
                electron_conditioning={"max_taper_bins": 20},
                ion_conditioning={"max_taper_bins": 20},
                progress=False,
            )
        assert spectrogram.epw.shape == (2, EPW_WINDOW.size)
        assert np.all(np.isfinite(spectrogram.epw))


# ---------------------------------------------------------------------------
# 9. Diagnostics must not cry wolf
# ---------------------------------------------------------------------------


class TestDiagnosticFalseAlarms:
    def test_roundoff_negatives_do_not_warn(self):
        """
        Boxcar smoothing of a strictly non-negative distribution leaves
        negatives of order 1e-20 against a 1e-5 peak. Those are round-off, and
        warning about millions of them buries the real signal.
        """
        v = np.linspace(-5.0, 5.0, 201)
        f = block(maxwellian(v, 1.0))
        f[0, 3, 0] = -1e-18 * f.max()
        # Warnings are errors in this suite, so calling through is the assertion.
        pic_thomson.normalize_vdf(f, v)

    def test_physical_negatives_still_warn(self):
        v = np.linspace(-5.0, 5.0, 201)
        f = block(maxwellian(v, 1.0))
        f[0, 3, 0] = -0.01 * f.max()
        with pytest.warns(RuntimeWarning, match="clipped 1 negative"):
            pic_thomson.normalize_vdf(f, v)

    def test_empty_cells_do_not_trigger_the_pedestal_warning(self):
        """
        A near-empty cell has a near-zero width, so any taper multiplies it
        enormously. Real PIC data is full of such cells in vacuum, and letting
        them drive the check makes it fire always and mean nothing.
        """
        v = np.linspace(-4.0, 4.0, 401)
        f = block(maxwellian(v, 1.0), n_time=1, n_x=3)
        # A vacuum cell holding a single noise spike.
        f[0, :, 2] = 0.0
        f[0, 200, 2] = 1e-9 * f.max()
        f[0, 201, 2] = 1e-9 * f.max()
        pic_thomson.taper_vdf_edges(f, threshold_frac=0.005)

    def test_a_populated_cell_still_triggers_the_pedestal_warning(self):
        v = np.linspace(-12.0, 12.0, 401)
        f = block(maxwellian(v, 1.0), n_time=1, n_x=3)
        f[0, :, 2] = 0.0
        with pytest.warns(RuntimeWarning, match="fabricating a pedestal"):
            pic_thomson.taper_vdf_edges(f, threshold_frac=0.005)


# ---------------------------------------------------------------------------
# 10. Reducing to the sampled point before conditioning
# ---------------------------------------------------------------------------


class TestAtPosition:
    def test_reduces_to_one_cell(self):
        phase_space = simple_phase_space(n_x=5)
        reduced = phase_space.at_position(phase_space.x[3])
        assert reduced.shape == (3, 64, 1)
        assert reduced.meta["sampled_index"] == 3
        np.testing.assert_array_equal(reduced.f[:, :, 0], phase_space.f[:, :, 3])

    def test_conditioning_commutes_with_reduction(self):
        """
        Every conditioning step acts independently on each (time, position)
        slice, so reducing first must give bit-identical results. The driver
        relies on this: carrying unused spatial cells through a velocity
        rescale that oversamples the velocity axis costs orders of magnitude
        in memory.
        """
        v = np.linspace(-4e7, 4e7, 512)
        rng = np.random.default_rng(11)
        f = np.empty((3, v.size, 6))
        for i in range(3):
            for j in range(6):
                f[i, :, j] = maxwellian(
                    v, 4e6 * (1 + 0.3 * j), drift=1e6 * i
                ) + 1e-4 * rng.random(v.size)
        phase_space = pic_thomson.from_arrays(
            f=f, v=v, x=np.linspace(0, 1e-3, 6), t=np.arange(3.0), label="e-"
        )
        settings = {
            "smoothing_window": 9,
            "smoothing_iterations": 2,
            "max_taper_bins": 20,
            "velocity_scale_factor": 9.0,
            "pedestal_warning": None,
        }
        target = phase_space.x[4]

        reduce_then_condition = pic_thomson.condition_phase_space(
            phase_space.at_position(target), **settings
        )
        condition_then_reduce = pic_thomson.condition_phase_space(
            phase_space, **settings
        ).at_position(target)

        np.testing.assert_array_equal(reduce_then_condition.f, condition_then_reduce.f)
        np.testing.assert_array_equal(reduce_then_condition.v, condition_then_reduce.v)

    def test_rejects_out_of_bounds(self):
        with pytest.raises(ValueError, match="outside the spatial axis"):
            simple_phase_space().at_position(1.0 * u.m)


# ---------------------------------------------------------------------------
# 11. The WarpX reader
# ---------------------------------------------------------------------------


def _no_yt_needed():
    """Stand-in for the yt module, which the fake frame reader never uses."""
    return object()


def _yt_must_not_be_used():
    """Fail the test if the reader tries to read rather than use its cache."""
    pytest.fail("the cache was not used")


WARPX_DENSITY = 1e16  # m^-3
WARPX_DOMAIN = 10.0  # m
WARPX_SIGMA = 1e6  # m/s


def fake_warpx_frames(monkeypatch, mass, *, drift=0.0, n_particles=400_000):
    """
    Stand in for the yt layer with macroparticles drawn from a known
    Maxwellian, so the reader's binning and normalisation can be checked
    without any WarpX output on disk.
    """
    speed_of_light = const.c.si.value

    def frame(_yt, plotfile, _species, _fields):
        index = int(str(plotfile).rsplit("diag1", 1)[1])
        rng = np.random.default_rng(index)
        velocity = rng.normal(drift, WARPX_SIGMA, n_particles)
        gamma = 1.0 / np.sqrt(1.0 - (velocity / speed_of_light) ** 2)
        momentum = mass * gamma * velocity
        position = rng.uniform(0.0, WARPX_DOMAIN, n_particles)
        # Weights chosen so the domain holds exactly WARPX_DENSITY per m^3.
        weight = np.full(n_particles, WARPX_DENSITY * WARPX_DOMAIN / n_particles)
        domain = (np.zeros(3), np.array([WARPX_DOMAIN, 1.0, 1.0]))
        return position, momentum, weight, index * 1e-9, domain

    monkeypatch.setattr(pic_thomson, "_load_yt", _no_yt_needed)
    monkeypatch.setattr(pic_thomson, "_warpx_frame", frame)


def make_warpx_plotfiles(tmp_path, n_frames=3, prefix="diag1"):
    """Create empty plotfile directories for the reader to enumerate."""
    diags = tmp_path / "diags"
    for index in range(n_frames):
        (diags / f"{prefix}{index:06d}").mkdir(parents=True)
    return diags


class TestWarpxReader:
    def test_shapes_and_axes(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=256,
            n_position_bins=64,
            progress=False,
        )
        assert phase_space.shape == (3, 256, 64)
        assert phase_space.label == "p+"
        assert not phase_space.is_electron
        assert phase_space.x[0] > 0
        assert phase_space.x[-1] < WARPX_DOMAIN
        np.testing.assert_allclose(phase_space.t, [0.0, 1e-9, 2e-9])

    def test_weights_give_a_number_density(self, tmp_path, monkeypatch):
        """
        The zeroth moment must come out as a physical number density in m^-3,
        which is what lets the driver be handed reference_density = 1 m^-3.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=512,
            n_position_bins=32,
            progress=False,
        )
        density = pic_thomson.number_density(phase_space.f, phase_space.v)
        # Tolerance set by macroparticle shot noise, not by the scaling.
        np.testing.assert_allclose(density[0], WARPX_DENSITY, rtol=0.05)

    def test_recovers_the_input_distribution(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        drift = 3e5
        fake_warpx_frames(monkeypatch, mass, drift=drift)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=512,
            n_position_bins=8,
            progress=False,
        )
        f = phase_space.f[0, :, 3]
        v = phase_space.v
        total = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / total
        width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / total)
        np.testing.assert_allclose(mean, drift, rtol=0.05)
        np.testing.assert_allclose(width, WARPX_SIGMA, rtol=0.05)

    def test_momentum_conversion_is_relativistic(self, tmp_path, monkeypatch):
        """
        Velocity must come from p/(m c) through v = u c / sqrt(1 + u^2), not
        from p/m, which would exceed c for energetic particles.
        """
        mass = const.m_e.si.value
        speed_of_light = const.c.si.value

        def frame(_yt, _plotfile, _species, _fields):
            # A single particle at u = 3, i.e. v = 0.949 c.
            momentum = np.array([3.0 * mass * speed_of_light])
            return (
                np.array([5.0]),
                momentum,
                np.array([1.0]),
                0.0,
                (np.zeros(3), np.array([WARPX_DOMAIN, 1.0, 1.0])),
            )

        monkeypatch.setattr(pic_thomson, "_load_yt", _no_yt_needed)
        monkeypatch.setattr(pic_thomson, "_warpx_frame", frame)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "electrons",
            mass=mass,
            is_electron=True,
            n_velocity_bins=64,
            progress=False,
        )
        expected = 3.0 * speed_of_light / np.sqrt(10.0)
        assert expected < speed_of_light
        # The auto-derived axis is clamped to below c as well.
        assert phase_space.v.max() < speed_of_light
        # The single particle lands in the bin holding its velocity.
        occupied = phase_space.v[phase_space.f[0, :, :].sum(axis=1) > 0]
        assert abs(occupied[0] - expected) < np.diff(phase_space.v)[0]

    def test_explicit_ranges_are_respected(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            velocity_range=(-5e6, 5e6),
            position_range=(2.0, 8.0),
            n_velocity_bins=100,
            n_position_bins=60,
            progress=False,
        )
        assert phase_space.v[0] > -5e6
        assert phase_space.v[-1] < 5e6
        assert 2.0 < phase_space.x[0] < 2.2
        assert 7.8 < phase_space.x[-1] < 8.0

    def test_cache_round_trip(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)
        cache = tmp_path / "cache" / "amb_ions.npz"

        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 128,
            "n_position_bins": 16,
            "velocity_range": (-4e6, 4e6),
            "progress": False,
        }
        first = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", cache=cache, **settings
        )
        assert cache.is_file()

        # With the yt layer removed entirely, only the cache can answer.
        monkeypatch.setattr(pic_thomson, "_load_yt", _yt_must_not_be_used)
        second = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", cache=cache, **settings
        )
        np.testing.assert_array_equal(first.f, second.f)
        np.testing.assert_array_equal(first.v, second.v)
        np.testing.assert_array_equal(first.t, second.t)
        assert second.meta["cache"] == str(cache)

    def test_cache_is_rebuilt_when_settings_change(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        cache = tmp_path / "amb_ions.npz"

        coarse = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=64,
            cache=cache,
            progress=False,
        )
        fine = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=128,
            cache=cache,
            progress=False,
        )
        assert coarse.shape[1] == 64
        assert fine.shape[1] == 128

    def test_selects_requested_frames(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=5)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            timesteps=[0, 2, 4],
            n_velocity_bins=32,
            progress=False,
        )
        assert phase_space.shape[0] == 3
        np.testing.assert_allclose(phase_space.t, [0.0, 2e-9, 4e-9])

    def test_rejects_missing_plotfiles(self, tmp_path, monkeypatch):
        fake_warpx_frames(monkeypatch, const.m_e.si.value)
        (tmp_path / "diags").mkdir()
        with pytest.raises(FileNotFoundError, match="no diag1"):
            pic_thomson.read_warpx_phase_space(
                tmp_path / "diags",
                "amb_ions",
                mass=1e-27,
                label="p+",
                progress=False,
            )

    def test_rejects_a_frame_beyond_the_series(self, tmp_path, monkeypatch):
        fake_warpx_frames(monkeypatch, const.m_e.si.value)
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)
        with pytest.raises(IndexError, match="beyond the 2 present"):
            pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=1e-27,
                label="p+",
                timesteps=[0, 7],
                progress=False,
            )

    def test_requires_a_label_for_ions(self, tmp_path, monkeypatch):
        fake_warpx_frames(monkeypatch, const.m_e.si.value)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="species label is required"):
            pic_thomson.read_warpx_phase_space(
                diags, "amb_ions", mass=1e-27, progress=False
            )

    def test_rejects_non_positive_mass(self, tmp_path, monkeypatch):
        fake_warpx_frames(monkeypatch, const.m_e.si.value)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="mass must be positive"):
            pic_thomson.read_warpx_phase_space(
                diags, "amb_ions", mass=0.0, label="p+", progress=False
            )

    def test_accepts_mass_as_a_quantity(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e
        fake_warpx_frames(monkeypatch, mass.si.value)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=64,
            progress=False,
        )
        assert phase_space.meta["mass"] == pytest.approx(mass.si.value)

    def test_output_feeds_the_driver(self, tmp_path, monkeypatch):
        """
        The whole point of the reader contract: WarpX output must reach the
        same driver as OSIRIS output, with no code-specific handling.
        """
        pytest.importorskip("numba")
        electron_mass = const.m_e.si.value
        fake_warpx_frames(monkeypatch, electron_mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)

        electrons = pic_thomson.read_warpx_phase_space(
            diags,
            "electrons",
            mass=electron_mass,
            is_electron=True,
            n_velocity_bins=512,
            n_position_bins=8,
            progress=False,
        )
        fake_warpx_frames(monkeypatch, 100 * electron_mass)
        ions = pic_thomson.read_warpx_phase_space(
            diags,
            "ions",
            mass=100 * electron_mass,
            label="p+",
            n_velocity_bins=512,
            n_position_bins=8,
            progress=False,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            spectrogram = pic_thomson.spectra_from_phase_spaces(
                electrons,
                [ions],
                position=electrons.x[4],
                reference_density=1 * u.m**-3,
                probe_wavelength=PROBE_WAVELENGTH,
                epw_wavelengths=EPW_WINDOW,
                electron_conditioning={"max_taper_bins": 20},
                ion_conditioning={"max_taper_bins": 20},
                progress=False,
            )
        assert spectrogram.epw.shape == (2, EPW_WINDOW.size)
        assert np.all(np.isfinite(spectrogram.epw))
        # reference_density = 1 m^-3 means the reader's own scaling carries the
        # density, so it must come back out at the value the weights encode.
        np.testing.assert_allclose(
            spectrogram.electron_density, WARPX_DENSITY, rtol=0.05
        )


# ---------------------------------------------------------------------------
# 12. Instrument response, HDF5 output, plotting
# ---------------------------------------------------------------------------


def synthetic_spectrogram(n_time=41, n_epw=64, n_iaw=32, gap=None):
    """A `ThomsonSpectrogram` with known structure, built without the driver."""
    epw_wavelengths = np.linspace(480e-9, 590e-9, n_epw)
    iaw_wavelengths = np.linspace(528e-9, 536e-9, n_iaw)
    t = np.linspace(0.0, 4e-9, n_time)

    epw = np.zeros((n_time, n_epw))
    iaw = np.zeros((n_time, n_iaw))
    for step in range(n_time):
        centre = 532e-9 + 20e-9 * (step / (n_time - 1) - 0.5)
        epw[step] = np.exp(-(((epw_wavelengths - centre) / 4e-9) ** 2))
        epw[step] /= np.trapezoid(epw[step], epw_wavelengths)
        iaw[step] = np.exp(-(((iaw_wavelengths - 532e-9) / 0.7e-9) ** 2))
        iaw[step] /= np.trapezoid(iaw[step], iaw_wavelengths)

    present_e = np.ones((1, n_time), dtype=bool)
    present_i = np.ones((2, n_time), dtype=bool)
    if gap is not None:
        epw[gap] = np.nan
        iaw[gap] = np.nan
        present_e[:, gap] = False
        present_i[:, gap] = False

    return pic_thomson.ThomsonSpectrogram(
        epw=epw,
        epw_wavelengths=epw_wavelengths,
        iaw=iaw,
        iaw_wavelengths=iaw_wavelengths,
        t=t,
        alpha_epw=np.linspace(1.0, 3.0, n_time),
        alpha_iaw=np.linspace(1.0, 3.0, n_time),
        electron_density=np.full(n_time, 5e24),
        efract=np.ones((1, n_time)),
        ifract=np.vstack([np.full(n_time, 0.7), np.full(n_time, 0.3)]),
        electron_present=present_e,
        ion_present=present_i,
        position=1.5e-3,
        electron_labels=["e-"],
        ion_labels=["p+", "Al 13+"],
        meta={"reference_density": 5e24, "probe_vec": [1.0, 0.0, 0.0]},
    )


class TestInstrumentResponse:
    def test_broadens_the_spectral_feature(self):
        spectrogram = synthetic_spectrogram()

        def width(row, axis):
            total = np.trapezoid(row, axis)
            mean = np.trapezoid(row * axis, axis) / total
            return np.sqrt(np.trapezoid(row * (axis - mean) ** 2, axis) / total)

        before = width(spectrogram.epw[20], spectrogram.epw_wavelengths)
        degraded = spectrogram.apply_instrument_response(epw_wavelength_fwhm=6 * u.nm)
        after = width(degraded.epw[20], degraded.epw_wavelengths)

        # Gaussians add in quadrature: sqrt(before^2 + sigma_instrument^2).
        sigma_instrument = (6e-9) / pic_thomson.FWHM_PER_SIGMA
        np.testing.assert_allclose(after, np.hypot(before, sigma_instrument), rtol=0.02)

    def test_time_smoothing_suppresses_a_moving_feature(self):
        """A feature that sweeps in wavelength is blurred by a long gate."""
        spectrogram = synthetic_spectrogram()
        degraded = spectrogram.apply_instrument_response(time_fwhm=1 * u.ns)
        assert degraded.epw[20].max() < spectrogram.epw[20].max()

    def test_leaves_the_input_untouched(self):
        spectrogram = synthetic_spectrogram()
        before = spectrogram.epw.copy()
        spectrogram.apply_instrument_response(time_fwhm=1 * u.ns)
        np.testing.assert_array_equal(spectrogram.epw, before)

    def test_a_gap_does_not_destroy_its_neighbours(self):
        """
        A vacuum timestep must not spread NaN over every row the kernel
        touches, which is what a plain filter does.
        """
        spectrogram = synthetic_spectrogram(gap=slice(18, 23))
        degraded = spectrogram.apply_instrument_response(time_fwhm=0.4 * u.ns)

        naive = gaussian_filter(spectrogram.epw, sigma=(2.0, 0.0), mode="nearest")
        assert np.isnan(naive).sum() > np.isnan(spectrogram.epw).sum()
        assert np.isnan(degraded.epw).sum() <= np.isnan(spectrogram.epw).sum()

        # Rows away from the gap stay finite and normalised.
        assert np.all(np.isfinite(degraded.epw[0]))
        np.testing.assert_allclose(
            np.trapezoid(degraded.epw[0], degraded.epw_wavelengths), 1.0, rtol=0.05
        )

    def test_a_narrow_gap_is_filled_from_its_neighbours(self):
        """
        Within the kernel's reach a gap is filled, because that is what an
        instrument integrating over a finite gate actually does.
        """
        spectrogram = synthetic_spectrogram(gap=slice(20, 21))
        degraded = spectrogram.apply_instrument_response(time_fwhm=0.5 * u.ns)
        assert np.all(np.isfinite(degraded.epw[20]))

    def test_a_gap_wider_than_the_kernel_stays_empty(self):
        spectrogram = synthetic_spectrogram(gap=slice(10, 31))
        degraded = spectrogram.apply_instrument_response(time_fwhm=0.2 * u.ns)
        assert np.all(np.isnan(degraded.epw[20]))
        assert np.all(np.isfinite(degraded.epw[0]))

    def test_boxcar_kernel(self):
        spectrogram = synthetic_spectrogram()
        degraded = spectrogram.apply_instrument_response(
            epw_wavelength_fwhm=6 * u.nm, kernel="boxcar"
        )
        assert degraded.epw[20].max() < spectrogram.epw[20].max()

    def test_rejects_an_unknown_kernel(self):
        with pytest.raises(ValueError, match="must be 'gaussian' or 'boxcar'"):
            synthetic_spectrogram().apply_instrument_response(
                time_fwhm=1 * u.ns, kernel="triangle"
            )

    def test_records_the_settings(self):
        degraded = synthetic_spectrogram().apply_instrument_response(
            time_fwhm=100 * u.ps, epw_wavelength_fwhm=0.5 * u.nm
        )
        response = degraded.meta["instrument_response"]
        assert response["time_fwhm_s"] == pytest.approx(1e-10)
        assert response["epw_wavelength_fwhm_m"] == pytest.approx(0.5e-9)
        assert response["iaw_wavelength_fwhm_m"] is None

    def test_no_widths_is_a_no_op(self):
        spectrogram = synthetic_spectrogram()
        degraded = spectrogram.apply_instrument_response()
        np.testing.assert_allclose(degraded.epw, spectrogram.epw, rtol=1e-12)

    def test_handles_a_spectrogram_without_an_iaw_window(self):
        spectrogram = replace(synthetic_spectrogram(), iaw=None, iaw_wavelengths=None)
        degraded = spectrogram.apply_instrument_response(time_fwhm=1 * u.ns)
        assert degraded.iaw is None


class TestHDF5RoundTrip:
    def test_round_trip(self, tmp_path):
        spectrogram = synthetic_spectrogram(gap=slice(5, 8))
        path = tmp_path / "spectra.h5"
        spectrogram.to_hdf5(path)
        restored = pic_thomson.ThomsonSpectrogram.from_hdf5(path)

        for name in (
            "epw",
            "iaw",
            "epw_wavelengths",
            "iaw_wavelengths",
            "t",
            "alpha_epw",
            "alpha_iaw",
            "electron_density",
            "efract",
            "ifract",
            "electron_present",
            "ion_present",
        ):
            np.testing.assert_array_equal(
                getattr(restored, name), getattr(spectrogram, name), err_msg=name
            )
        assert restored.position == pytest.approx(spectrogram.position)
        assert restored.electron_labels == spectrogram.electron_labels
        assert restored.ion_labels == spectrogram.ion_labels
        assert restored.meta["reference_density"] == 5e24

    def test_round_trip_without_an_iaw_window(self, tmp_path):
        spectrogram = replace(
            synthetic_spectrogram(), iaw=None, iaw_wavelengths=None, alpha_iaw=None
        )
        path = tmp_path / "spectra.h5"
        spectrogram.to_hdf5(path)
        restored = pic_thomson.ThomsonSpectrogram.from_hdf5(path)
        assert restored.iaw is None
        assert restored.iaw_wavelengths is None
        assert restored.alpha_iaw is None
        np.testing.assert_array_equal(restored.epw, spectrogram.epw)

    def test_units_are_recorded(self, tmp_path):
        path = tmp_path / "spectra.h5"
        synthetic_spectrogram().to_hdf5(path)
        with h5py.File(path, "r") as handle:
            assert handle["DENSITY/dens"].attrs["UNITS"] == "m^-3"
            assert handle["AXES/TIME_AXES/time"].attrs["UNITS"] == "s"
            assert handle["AXES/WAVELENGTH_AXES/epw_wavelengths"].attrs["UNITS"] == "m"
            assert handle.attrs["POSITION_UNITS"] == "m"
            # The per-row normalisation is easy to forget; it is on the file.
            assert "unit area" in handle.attrs["NORMALISATION"]
            assert (
                "lambda_De"
                in (
                    handle["SPECTRA/SCATTERING_PARAMETERS/alpha_epw"].attrs[
                        "CONVENTION"
                    ]
                )
            )

    def test_species_labels_with_awkward_characters(self, tmp_path):
        """Labels like 'Al 13+' must survive, spaces and plus signs included."""
        path = tmp_path / "spectra.h5"
        synthetic_spectrogram().to_hdf5(path)
        restored = pic_thomson.ThomsonSpectrogram.from_hdf5(path)
        assert restored.ion_labels == ["p+", "Al 13+"]

    def test_overwrites_an_existing_file(self, tmp_path):
        path = tmp_path / "spectra.h5"
        synthetic_spectrogram(n_time=41).to_hdf5(path)
        synthetic_spectrogram(n_time=11).to_hdf5(path)
        assert pic_thomson.ThomsonSpectrogram.from_hdf5(path).n_time == 11


class TestPlot:
    def test_returns_a_figure(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        figure, axes = synthetic_spectrogram().plot()
        assert axes.shape == (2, 2)
        matplotlib.pyplot.close(figure)

    def test_writes_a_file(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        path = tmp_path / "spectra.png"
        figure, _ = synthetic_spectrogram().plot(save=path)
        assert path.is_file()
        matplotlib.pyplot.close(figure)

    def test_copes_without_an_iaw_window(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        spectrogram = replace(synthetic_spectrogram(), iaw=None, iaw_wavelengths=None)
        figure, _ = spectrogram.plot()
        matplotlib.pyplot.close(figure)

    def test_copes_with_all_nan_rows(self):
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        figure, _ = synthetic_spectrogram(gap=slice(0, 10)).plot()
        matplotlib.pyplot.close(figure)

    def test_repeated_species_labels_are_numbered_in_the_legend(self):
        """
        Two populations of the same species are legitimate -- an ambient and a
        piston plasma of the same element -- but two identically named legend
        entries are not.
        """
        matplotlib = pytest.importorskip("matplotlib")
        matplotlib.use("Agg")
        spectrogram = replace(synthetic_spectrogram(), ion_labels=["p+", "p+"])
        figure, axes = spectrogram.plot()
        labels = [text.get_text() for text in axes[1][1].get_legend().get_texts()]
        assert "ifract p+ #1" in labels
        assert "ifract p+ #2" in labels
        matplotlib.pyplot.close(figure)
