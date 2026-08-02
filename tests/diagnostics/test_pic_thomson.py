"""
Tests for the PIC -> Thomson scattering pipeline in
`plasmapy.diagnostics.pic_thomson`.
"""

import warnings

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest

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
