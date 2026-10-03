"""
Tests for the PIC -> Thomson scattering pipeline in
`plasmapy.diagnostics.pic_thomson`.
"""

import warnings
from dataclasses import replace
from pathlib import Path

import astropy.constants as const
import astropy.units as u
import h5py
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from plasmapy.diagnostics import pic_thomson, thomson
from plasmapy.diagnostics.pic_thomson import histogram2d_deck_block

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
            # This fixture's velocity grid is only a few thermal widths wide, so
            # a 5-bin boxcar does inflate the second moment; that is the subject
            # of TestSmoothingWidth, not of this test.
            smoothing_variance_warning=None,
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

    def test_the_taper_inflates_the_width_on_a_wide_grid(self):
        """
        Companion to the test above: with the taper on a 6-sigma grid the
        recovered width comes out ~5% high, and the user is told. This is why
        the taper is no longer the default -- see `extend_vdf_tail`.
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
            conditioned = pic_thomson.condition_phase_space(
                phase_space, tail_model=None
            )
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

    def test_takes_the_density_scale_from_the_readers(self):
        """
        Every reader records the scale its ``f`` is in, so the caller should not
        have to remember that OSIRIS means n0 and WarpX means 1 per cubic metre.
        """
        pytest.importorskip("numba")
        scale = REFERENCE_DENSITY.to_value(u.m**-3)
        electrons = replace(
            species_phase_space("e-", 100, const.m_e),
            meta={"reference_density": scale},
        )
        ions = replace(
            species_phase_space("p+", 50, const.m_p),
            meta={"reference_density": scale},
        )
        inferred = run_driver(electrons=electrons, ions=[ions], reference_density=None)
        explicit = run_driver(electrons=electrons, ions=[ions])
        assert inferred.meta["reference_density"] == pytest.approx(scale)
        np.testing.assert_allclose(inferred.epw, explicit.epw, rtol=1e-12)

    def test_rejects_phase_spaces_that_disagree_about_the_scale(self):
        electrons = replace(
            species_phase_space("e-", 100, const.m_e),
            meta={"reference_density": 1e24},
        )
        ions = replace(
            species_phase_space("p+", 50, const.m_p),
            meta={"reference_density": 1e25},
        )
        with pytest.raises(ValueError, match="disagree about the density scale"):
            run_driver(electrons=electrons, ions=[ions], reference_density=None)

    def test_reports_a_missing_density_scale(self):
        with pytest.raises(ValueError, match="carry no 'reference_density'"):
            run_driver(reference_density=None)

    def test_rejects_velocity_scale_factor_in_conditioning(self):
        with pytest.raises(ValueError, match="not through the per-species"):
            run_driver(
                electron_conditioning={"velocity_scale_factor": 50},
            )

    def test_warns_when_an_ion_is_passed_as_an_electron(self):
        # A proton standing in for an electron also puts the EPW resonance some
        # 77 thermal widths out, so the tail check fires too. Match on the
        # category and assert on the message, rather than letting pytest
        # re-emit the second warning into the error filter.
        with pytest.warns(RuntimeWarning) as record:
            run_driver(electrons=species_phase_space("p+", 100, const.m_p))
        assert any("not flagged as one" in str(entry.message) for entry in record)

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
        assert phase_space.meta["transverse_axes"] == ["x2"]

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
        assert phase_space.meta["transverse_axes"] == ["x1"]

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


class FakeDataset:
    """Just enough of a yt dataset for the reader to introspect."""

    def __init__(self, species, position_fields, momentum_fields, dimensions):
        if isinstance(species, str):
            species = (species,)
        self.field_list = [
            (kind, name)
            for kind in species
            for name in (*position_fields, *momentum_fields, "particle_weight")
        ]
        self.particle_types = tuple(species)
        self.domain_dimensions = np.asarray(dimensions)


class FakeYt:
    """Stand-in for the yt module; `load` ignores the path."""

    def __init__(self, dataset):
        self._dataset = dataset

    def load(self, _path):
        return self._dataset


def _no_yt_needed():
    """The default fake yt: a 1-D run with all three momentum components."""
    return FakeYt(
        FakeDataset(
            TEST_SPECIES,
            ("particle_position_x",),
            ("particle_momentum_x", "particle_momentum_y", "particle_momentum_z"),
            (64, 1, 1),
        )
    )


def _yt_must_not_be_used():
    """Fail the test if the reader tries to read rather than use its cache."""
    pytest.fail("the cache was not used")


TEST_SPECIES = ("amb_ions", "piston_ions", "electrons", "ions")
WARPX_DENSITY = 1e16  # m^-3
WARPX_DOMAIN = 10.0  # m
WARPX_SIGMA = 1e6  # m/s


def fake_warpx_frames(
    monkeypatch, mass, *, drift=0.0, n_particles=400_000, n_spatial=1
):
    """
    Stand in for the yt layer with macroparticles drawn from a known
    Maxwellian, so the reader's binning and normalisation can be checked
    without any WarpX output on disk.
    """
    speed_of_light = const.c.si.value

    def frame(_yt, plotfile, _species, _positions=None, _momenta=None):
        index = int(str(plotfile).rsplit("diag1", 1)[1])
        rng = np.random.default_rng(index)
        # The drifting Maxwellian is along z; x and y carry only a little
        # thermal spread, so a projection onto z recovers the input while a
        # projection onto x does not.
        velocity = np.stack(
            [
                rng.normal(0.0, 0.1 * WARPX_SIGMA, n_particles),
                rng.normal(0.0, 0.1 * WARPX_SIGMA, n_particles),
                rng.normal(drift, WARPX_SIGMA, n_particles),
            ]
        )
        speed = np.sqrt(np.sum(velocity**2, axis=0))
        gamma = 1.0 / np.sqrt(1.0 - (speed / speed_of_light) ** 2)
        momenta = mass * gamma * velocity
        positions = rng.uniform(0.0, WARPX_DOMAIN, (n_spatial, n_particles))
        # Weights chosen so the domain holds exactly WARPX_DENSITY per m^3.
        cell_volume = WARPX_DOMAIN**n_spatial
        weight = np.full(n_particles, WARPX_DENSITY * cell_volume / n_particles)
        domain = (
            np.zeros(3),
            np.array([WARPX_DOMAIN] * n_spatial + [1.0] * (3 - n_spatial)),
        )
        return positions, momenta, weight, index * 1e-9, domain

    dataset = FakeDataset(
        TEST_SPECIES,
        tuple(f"particle_position_{c}" for c in "xyz"[:n_spatial]),
        ("particle_momentum_x", "particle_momentum_y", "particle_momentum_z"),
        (64,) * n_spatial + (1,) * (3 - n_spatial),
    )
    monkeypatch.setattr(pic_thomson, "_load_yt", lambda: FakeYt(dataset))
    monkeypatch.setattr(pic_thomson, "_warpx_frame", frame)


COARSE_VELOCITY_AXIS = pytest.mark.filterwarnings(
    # These fixtures hold a handful of macroparticles, so their "thermal width"
    # is a couple of bins by construction. The velocity-resolution check has its
    # own tests; here it is noise.
    "ignore:the velocity axis gives species:RuntimeWarning"
)


def make_warpx_plotfiles(tmp_path, n_frames=3, prefix="diag1"):
    """Create empty plotfile directories for the reader to enumerate."""
    diags = tmp_path / "diags"
    for index in range(n_frames):
        (diags / f"{prefix}{index:06d}").mkdir(parents=True, exist_ok=True)
    return diags


def fake_warpx_frames_with_a_fast_middle(monkeypatch, mass, *, n_frames):
    """
    Frames whose fastest particles live in the middle of the series -- the
    shape a shock has, and the one a first-and-last scan cannot see.
    """
    speed_of_light = const.c.si.value
    middle = n_frames // 2

    def frame(_yt, plotfile, _species, _positions=None, _momenta=None):
        index = int(str(plotfile).rsplit("diag1", 1)[1])
        rng = np.random.default_rng(index)
        # Ends are cold; the middle frame is ten times hotter.
        sigma = WARPX_SIGMA * (10.0 if index == middle else 1.0)
        velocity = np.stack(
            [
                rng.normal(0.0, 0.01 * WARPX_SIGMA, 20_000),
                rng.normal(0.0, 0.01 * WARPX_SIGMA, 20_000),
                rng.normal(0.0, sigma, 20_000),
            ]
        )
        speed = np.sqrt(np.sum(velocity**2, axis=0))
        gamma = 1.0 / np.sqrt(1.0 - (speed / speed_of_light) ** 2)
        positions = rng.uniform(0.0, WARPX_DOMAIN, (1, 20_000))
        weight = np.full(20_000, WARPX_DENSITY * WARPX_DOMAIN / 20_000)
        domain = (np.zeros(3), np.array([WARPX_DOMAIN, 1.0, 1.0]))
        return positions, mass * gamma * velocity, weight, index * 1e-9, domain

    dataset = FakeDataset(
        TEST_SPECIES,
        ("particle_position_x",),
        ("particle_momentum_x", "particle_momentum_y", "particle_momentum_z"),
        (64, 1, 1),
    )
    monkeypatch.setattr(pic_thomson, "_load_yt", lambda: FakeYt(dataset))
    monkeypatch.setattr(pic_thomson, "_warpx_frame", frame)


class TestWarpxVelocityScan:
    """
    Sizing the velocity axis from the first and last frames alone silently
    clips whatever the run does in between -- which, for a shock, is the entire
    measurement.
    """

    N_FRAMES = 5

    @COARSE_VELOCITY_AXIS
    def test_scanning_every_frame_covers_the_middle(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames_with_a_fast_middle(monkeypatch, mass, n_frames=self.N_FRAMES)
        diags = make_warpx_plotfiles(tmp_path, n_frames=self.N_FRAMES)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            velocity_scan="all",
            n_velocity_bins=256,
            progress=False,
        )
        # The axis has to reach past the hot middle frame, which is ten times
        # the thermal speed of the frames at either end.
        assert phase_space.v.max() > 10 * WARPX_SIGMA
        assert phase_space.meta["clipped_count"] == 0

    @COARSE_VELOCITY_AXIS
    def test_scanning_only_the_ends_clips_and_says_so(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames_with_a_fast_middle(monkeypatch, mass, n_frames=self.N_FRAMES)
        diags = make_warpx_plotfiles(tmp_path, n_frames=self.N_FRAMES)

        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 256,
            "progress": False,
        }
        with pytest.warns(RuntimeWarning, match="fell outside the velocity axis"):
            ends = pic_thomson.read_warpx_phase_space(
                diags, "amb_ions", velocity_scan="ends", **settings
            )
        every = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", velocity_scan="all", **settings
        )
        # The ends are cold, so an axis sized from them alone falls far short
        # of the one sized from the whole series.
        assert ends.v.max() < 0.5 * every.v.max()
        assert ends.meta["clipped_count"] > 0

    def test_rejects_an_unknown_scan(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)
        with pytest.raises(ValueError, match="velocity_scan must be"):
            pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                velocity_scan="middle",
                progress=False,
            )


class TestDepositedChargeCheck:
    """
    The macroparticles a diagnostic writes are not always the macroparticles
    the code deposited. ``random_fraction`` subsamples them without
    reweighting, so every density built from that output is low by exactly that
    factor, and nothing in the file says so.
    """

    def test_a_subsampled_diagnostic_is_caught(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        # The code deposited five times what the diagnostic wrote: exactly what
        # `random_fraction = 0.2` produces.
        monkeypatch.setattr(
            pic_thomson,
            "_warpx_deposited_weight",
            lambda _dataset, _species, _charge: 5.0 * WARPX_DENSITY * WARPX_DOMAIN,
        )
        with pytest.warns(RuntimeWarning, match="random_fraction"):
            phase_space = pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                n_velocity_bins=64,
                progress=False,
            )
        assert phase_space.meta["density_check"]["ratio"] == pytest.approx(0.2)

    def test_agreement_is_silent_and_recorded(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        monkeypatch.setattr(
            pic_thomson,
            "_warpx_deposited_weight",
            lambda _dataset, _species, _charge: WARPX_DENSITY * WARPX_DOMAIN,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            phase_space = pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                n_velocity_bins=64,
                progress=False,
            )
        assert phase_space.meta["density_check"]["ratio"] == pytest.approx(1.0)

    def test_can_be_switched_off(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        monkeypatch.setattr(
            pic_thomson,
            "_warpx_deposited_weight",
            lambda *_args: pytest.fail("the check should not have run"),
        )
        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            validate_density=False,
            n_velocity_bins=64,
            progress=False,
        )
        assert phase_space.meta["density_check"] is None

    def test_a_plotfile_without_the_field_is_skipped(self, tmp_path, monkeypatch):
        """Most plotfiles carry no per-species charge density; that is not an error."""
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=64,
            progress=False,
        )
        assert phase_space.meta["density_check"] is None

    def test_finds_the_field_by_name(self):
        dataset = FakeDataset(
            ("amb_ions",),
            ("particle_position_x",),
            ("particle_momentum_z",),
            (8, 1, 1),
        )
        assert pic_thomson._warpx_rho_field(dataset, "amb_ions") is None
        dataset.field_list.append(("boxlib", "rho_amb_ions"))
        assert pic_thomson._warpx_rho_field(dataset, "amb_ions") == (
            "boxlib",
            "rho_amb_ions",
        )


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

        def frame(_yt, _plotfile, _species, _positions=None, _momenta=None):
            # A single particle at u = 3 along z, i.e. v = 0.949 c.
            momentum = np.array([[0.0], [0.0], [3.0 * mass * speed_of_light]])
            return (
                np.array([[5.0]]),
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

    def test_a_clipped_velocity_axis_is_reported(self, tmp_path, monkeypatch):
        """
        ``np.histogram2d`` discards out-of-range samples without a word, and a
        truncated tail is exactly what the taper would then smooth over -- so
        the reader has to count what it lost.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        with pytest.warns(RuntimeWarning, match="fell outside the velocity axis"):
            phase_space = pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                # A quarter of a sigma: most of the distribution is outside.
                velocity_range=(-WARPX_SIGMA / 4, WARPX_SIGMA / 4),
                n_velocity_bins=64,
                progress=False,
            )
        assert phase_space.meta["clipped_count"] > 0
        assert phase_space.meta["clipped_weight_fraction"] > 0.5

    def test_narrowing_the_position_range_is_not_clipping(self, tmp_path, monkeypatch):
        """
        Sampling one point is a deliberate selection, like the transverse slab.
        Counting the rest of the domain as lost would make the check cry wolf
        on every single-position read.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            position=WARPX_DOMAIN / 2,
            n_velocity_bins=64,
            progress=False,
        )
        assert phase_space.meta["clipped_count"] == 0

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

    def test_cache_is_keyed_on_the_frames_read(self, tmp_path, monkeypatch):
        """
        ``timesteps`` changes which frames ``f`` holds, so a cache built from a
        short test read must not be handed back for the full series.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=4)
        cache = tmp_path / "amb_ions.npz"
        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 32,
            "velocity_range": (-4e6, 4e6),
            "cache": cache,
            "progress": False,
        }

        subset = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", timesteps=[0], **settings
        )
        every = pic_thomson.read_warpx_phase_space(diags, "amb_ions", **settings)
        assert subset.shape[0] == 1
        assert every.shape[0] == 4

    def test_cache_is_keyed_on_the_transverse_area(self, tmp_path, monkeypatch):
        """``transverse_area`` scales ``f`` linearly, so it has to key the cache."""
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        cache = tmp_path / "amb_ions.npz"
        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 32,
            "velocity_range": (-4e6, 4e6),
            "cache": cache,
            "progress": False,
        }

        unit = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", transverse_area=1.0, **settings
        )
        doubled = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", transverse_area=2.0, **settings
        )
        np.testing.assert_allclose(2.0 * doubled.f, unit.f, rtol=1e-12)

    @COARSE_VELOCITY_AXIS
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
            # Without a record of how rows were scaled, the file says so.
            assert handle.attrs["NORMALISATION"] == "not recorded"
        for flag, phrase in ((True, "unit area"), (False, "n_e S(k, omega)")):
            spectrogram = synthetic_spectrogram()
            spectrogram.meta["normalise_each_frame"] = flag
            spectrogram.to_hdf5(path)
            with h5py.File(path, "r") as handle:
                assert phrase in handle.attrs["NORMALISATION"]
        with h5py.File(path, "r") as handle:
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


# ---------------------------------------------------------------------------
# 13. Multi-dimensional simulations
# ---------------------------------------------------------------------------


class TestOsirisNDim:
    """
    OSIRIS phase spaces can carry more than one spatial axis, as ``p1x1x2``
    does. The extra axes have to be reduced to the volume the probe looks at.
    """

    def test_slab_and_chord_give_the_same_density(self, tmp_path):
        """
        The reduction averages rather than sums, so the zeroth moment stays a
        density however many cells it combines. Summing -- what the
        osiris2thomson pipeline did -- would scale it by the cell count.
        """
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=64, n_x=8, transverse=16
        )
        read = {
            "reference_density": OSIRIS_REFERENCE_DENSITY,
            "is_electron": True,
            "progress": False,
        }
        slab = pic_thomson.read_osiris_phase_space(ms, "p1x1x2", "e", **read)
        chord = pic_thomson.read_osiris_phase_space(
            ms, "p1x1x2", "e", transverse_reduction="chord", **read
        )
        np.testing.assert_allclose(
            pic_thomson.number_density(slab.f, slab.v),
            pic_thomson.number_density(chord.f, chord.v),
            rtol=1e-12,
        )
        assert slab.meta["transverse_axes"] == ["x2"]
        assert slab.meta["transverse_reduction"] == "slab"

    def test_a_wider_slab_still_gives_a_density(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=64, n_x=8, transverse=16
        )
        read = {
            "reference_density": OSIRIS_REFERENCE_DENSITY,
            "is_electron": True,
            "progress": False,
        }
        narrow = pic_thomson.read_osiris_phase_space(ms, "p1x1x2", "e", **read)
        wide = pic_thomson.read_osiris_phase_space(
            ms, "p1x1x2", "e", slab_halfwidth=1.0, **read
        )
        np.testing.assert_allclose(
            pic_thomson.number_density(wide.f, wide.v),
            pic_thomson.number_density(narrow.f, narrow.v),
            rtol=1e-12,
        )

    def test_transverse_position_selects_a_different_slab(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=32, n_x=8, transverse=16
        )
        phase_space = pic_thomson.read_osiris_phase_space(
            ms,
            "p1x1x2",
            "e",
            reference_density=OSIRIS_REFERENCE_DENSITY,
            is_electron=True,
            transverse_position={"x2": 0.0},
            progress=False,
        )
        assert phase_space.shape == (3, 32, 8)

    def test_rejects_an_unknown_transverse_axis(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=32, n_x=8, transverse=4
        )
        with pytest.raises(ValueError, match=r"not.*transverse axes"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1x2",
                "e",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                is_electron=True,
                transverse_position={"x7": 0.0},
                progress=False,
            )

    def test_rejects_an_unknown_reduction(self, tmp_path):
        ms, _, _ = make_osiris_run(
            tmp_path, field="p1x1x2", n_p=32, n_x=8, transverse=4
        )
        with pytest.raises(ValueError, match=r"'slab' or 'chord'"):
            pic_thomson.read_osiris_phase_space(
                ms,
                "p1x1x2",
                "e",
                reference_density=OSIRIS_REFERENCE_DENSITY,
                is_electron=True,
                transverse_reduction="average",
                progress=False,
            )

    def test_position_reduces_to_a_point(self, tmp_path):
        ms, _, _ = make_osiris_run(tmp_path, n_p=64, n_x=16)
        read = {
            "reference_density": OSIRIS_REFERENCE_DENSITY,
            "is_electron": True,
            "progress": False,
        }
        profile = pic_thomson.read_osiris_phase_space(ms, "p1x1", "e", **read)
        target = float(profile.x[5])
        point = pic_thomson.read_osiris_phase_space(
            ms, "p1x1", "e", position=target, **read
        )
        assert point.shape == (3, 64, 1)
        np.testing.assert_allclose(point.x, [target], rtol=1e-12)
        np.testing.assert_allclose(point.f[:, :, 0], profile.f[:, :, 5], rtol=1e-12)

    def test_a_point_read_still_feeds_the_driver(self, tmp_path):
        pytest.importorskip("numba")
        ms, _, _ = make_osiris_run(tmp_path, n_p=256, n_x=8, n_dumps=2)
        make_osiris_run(tmp_path, species="al", n_p=256, n_x=8, n_dumps=2, sign=1.0)
        read = {
            "reference_density": OSIRIS_REFERENCE_DENSITY,
            "progress": False,
            "position": 50.0 * 5.6e-6,
        }
        electrons = pic_thomson.read_osiris_phase_space(
            ms, "p1x1", "e", is_electron=True, **read
        )
        ions = pic_thomson.read_osiris_phase_space(
            ms, "p1x1", "al", label="Al 13+", **read
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            spectrogram = pic_thomson.spectra_from_phase_spaces(
                electrons,
                [ions],
                position=electrons.x[0],
                reference_density=OSIRIS_REFERENCE_DENSITY,
                probe_wavelength=PROBE_WAVELENGTH,
                epw_wavelengths=EPW_WINDOW,
                electron_conditioning={"max_taper_bins": 20},
                ion_conditioning={"max_taper_bins": 20},
                progress=False,
            )
        assert np.all(np.isfinite(spectrogram.epw))


class TestWarpxNDim:
    def test_detects_the_dimensionality(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            scatter_direction=(0, 0, 1),
            n_velocity_bins=64,
            n_position_bins=8,
            progress=False,
        )
        assert phase_space.meta["n_spatial"] == 2
        assert phase_space.meta["position_fields"] == [
            "particle_position_x",
            "particle_position_y",
        ]
        assert phase_space.meta["transverse_axes"] == ["particle_position_y"]

    def test_slab_and_chord_give_the_same_density(self, tmp_path, monkeypatch):
        """
        The scattering volume changes, so the number of particles binned
        changes, but the density they represent must not.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        settings = {
            "mass": mass,
            "label": "p+",
            "scatter_direction": (0, 0, 1),
            "n_velocity_bins": 256,
            "n_position_bins": 8,
            "velocity_range": (-5e6, 5e6),
            "progress": False,
        }
        slab = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", slab_halfwidth=WARPX_DOMAIN / 8, **settings
        )
        chord = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", transverse_reduction="chord", **settings
        )
        assert slab.meta["transverse_area"] < chord.meta["transverse_area"]
        np.testing.assert_allclose(
            pic_thomson.number_density(slab.f, slab.v).mean(),
            pic_thomson.number_density(chord.f, chord.v).mean(),
            rtol=0.05,
        )
        np.testing.assert_allclose(
            pic_thomson.number_density(chord.f, chord.v).mean(),
            WARPX_DENSITY,
            rtol=0.05,
        )

    def test_scatter_direction_selects_the_velocity_component(
        self, tmp_path, monkeypatch
    ):
        """
        The fake distribution drifts along z with a much narrower spread in x,
        so projecting onto each axis must give visibly different widths. This
        is what a run with more than one velocity component needs, and what a
        single momentum component cannot express.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, drift=3e5)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 512,
            "n_position_bins": 4,
            "progress": False,
        }

        def moments(phase_space):
            f = phase_space.f[0, :, 2]
            v = phase_space.v
            total = np.trapezoid(f, v)
            mean = np.trapezoid(f * v, v) / total
            width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / total)
            return mean, width

        along_z = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", scatter_direction=(0, 0, 1), **settings
        )
        along_x = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", scatter_direction=(1, 0, 0), **settings
        )
        mean_z, width_z = moments(along_z)
        mean_x, width_x = moments(along_x)

        np.testing.assert_allclose(mean_z, 3e5, rtol=0.05)
        np.testing.assert_allclose(width_z, WARPX_SIGMA, rtol=0.05)
        assert abs(mean_x) < 0.1 * abs(mean_z)
        np.testing.assert_allclose(width_x, 0.1 * WARPX_SIGMA, rtol=0.05)

    def test_naming_a_momentum_component_matches_that_direction(
        self, tmp_path, monkeypatch
    ):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, drift=3e5)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        settings = {
            "mass": mass,
            "label": "p+",
            "n_velocity_bins": 256,
            "n_position_bins": 4,
            "velocity_range": (-4e6, 4e6),
            "progress": False,
        }
        named = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", momentum_field="particle_momentum_x", **settings
        )
        vector = pic_thomson.read_warpx_phase_space(
            diags, "amb_ions", scatter_direction=(1, 0, 0), **settings
        )
        np.testing.assert_array_equal(named.f, vector.f)

    @COARSE_VELOCITY_AXIS
    def test_gamma_uses_the_full_momentum(self, tmp_path, monkeypatch):
        r"""
        For a particle with momentum in more than one direction, the velocity
        along :math:`\hat{k}` is :math:`p_k / (m\gamma)` with :math:`\gamma`
        from the *total* momentum. Taking gamma from the projected component
        alone would overstate the velocity.
        """
        mass = const.m_e.si.value
        speed_of_light = const.c.si.value

        def frame(_yt, _plotfile, _species, _positions=None, _momenta=None):
            # u = (3, 0, 4): |u| = 5, so gamma = sqrt(26), and the velocity
            # along z is 4 c / sqrt(26), not 4 c / sqrt(17).
            momentum = mass * speed_of_light * np.array([[3.0], [0.0], [4.0]])
            return (
                np.array([[5.0]]),
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
            "amb_ions",
            mass=mass,
            label="p+",
            scatter_direction=(0, 0, 1),
            n_velocity_bins=2048,
            progress=False,
        )
        expected = 4.0 * speed_of_light / np.sqrt(26.0)
        naive = 4.0 * speed_of_light / np.sqrt(17.0)
        occupied = phase_space.v[phase_space.f[0].sum(axis=1) > 0]
        spacing = float(np.diff(phase_space.v)[0])
        assert abs(occupied[0] - expected) < spacing
        assert abs(occupied[0] - naive) > spacing

    def test_position_reduces_to_a_point(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)

        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            scatter_direction=(0, 0, 1),
            position=WARPX_DOMAIN / 2,
            transverse_position=[WARPX_DOMAIN / 2],
            slab_halfwidth=WARPX_DOMAIN / 8,
            velocity_range=(-5e6, 5e6),
            n_velocity_bins=256,
            progress=False,
        )
        assert phase_space.shape[2] == 1
        np.testing.assert_allclose(phase_space.x, [WARPX_DOMAIN / 2], rtol=1e-12)
        np.testing.assert_allclose(
            pic_thomson.number_density(phase_space.f, phase_space.v)[0, 0],
            WARPX_DENSITY,
            rtol=0.05,
        )

    def test_rejects_a_bad_axis(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="axis must index"):
            pic_thomson.read_warpx_phase_space(
                diags, "amb_ions", mass=mass, label="p+", axis=5, progress=False
            )

    def test_rejects_a_zero_scatter_direction(self, tmp_path, monkeypatch):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="zero vector"):
            pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                scatter_direction=(0, 0, 0),
                progress=False,
            )

    def test_rejects_the_wrong_number_of_transverse_positions(
        self, tmp_path, monkeypatch
    ):
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="one coordinate per transverse axis"):
            pic_thomson.read_warpx_phase_space(
                diags,
                "amb_ions",
                mass=mass,
                label="p+",
                scatter_direction=(0, 0, 1),
                transverse_position=[0.0, 1.0],
                progress=False,
            )

    def test_reports_a_missing_momentum_component(self, tmp_path, monkeypatch):
        """A 2-D run may not carry every momentum component."""
        mass = const.m_e.si.value
        dataset = FakeDataset(
            "electrons",
            ("particle_position_x", "particle_position_y"),
            ("particle_momentum_x",),
            (64, 64, 1),
        )
        monkeypatch.setattr(pic_thomson, "_load_yt", lambda: FakeYt(dataset))
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(KeyError, match="particle_momentum_z"):
            pic_thomson.read_warpx_phase_space(
                diags,
                "electrons",
                mass=mass,
                is_electron=True,
                momentum_field="particle_momentum_z",
                progress=False,
            )

    def test_refuses_to_guess_the_projection_beyond_one_dimension(
        self, tmp_path, monkeypatch
    ):
        """
        Momenta are stored for all three directions whatever the geometry, so a
        default projection in 2-D or 3-D is a silently wrong answer rather than
        an error. It has to be refused.
        """
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="no single momentum component"):
            pic_thomson.read_warpx_phase_space(
                diags, "amb_ions", mass=mass, label="p+", progress=False
            )

    def test_one_dimensional_runs_still_default(self, tmp_path, monkeypatch):
        """A 1-D run resolves one direction, so there is nothing to guess."""
        mass = 100 * const.m_e.si.value
        fake_warpx_frames(monkeypatch, mass)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        phase_space = pic_thomson.read_warpx_phase_space(
            diags,
            "amb_ions",
            mass=mass,
            label="p+",
            n_velocity_bins=64,
            progress=False,
        )
        assert phase_space.meta["scatter_direction"] == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# Moment-based reconstruction, for codes that carry a species as a fluid
# ---------------------------------------------------------------------------


class TestFromMoments:
    """
    A fluid species has no macroparticles, so its distribution has to be built
    from the moments the code does carry. On a grid that contains it, that
    reconstruction must return exactly the moments it was given.
    """

    DENSITY = 1e25  # m^-3
    TEMPERATURE = 100.0  # eV
    DRIFT = 4e5  # m/s

    def build(self, **kwargs):
        settings = {
            "t": [0.0],
            "x": [0.0],
            "label": "e-",
        }
        settings.update(kwargs)
        return pic_thomson.from_moments(
            [[self.DENSITY]],
            [[self.TEMPERATURE]] * u.eV,
            [[self.DRIFT]],
            **settings,
        )

    def moments(self, phase_space):
        f = phase_space.f[0, :, 0]
        v = phase_space.v
        total = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / total
        width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / total)
        return total, mean, width

    def test_recovers_the_moments_it_was_given(self):
        density, mean, width = self.moments(self.build())
        expected_width = np.sqrt(
            (self.TEMPERATURE * u.eV).to_value(u.J) / const.m_e.si.value
        )
        np.testing.assert_allclose(density, self.DENSITY, rtol=1e-6)
        np.testing.assert_allclose(mean, self.DRIFT, rtol=1e-6)
        np.testing.assert_allclose(width, expected_width, rtol=1e-6)

    def test_temperature_units_are_honoured(self):
        """Electronvolts and the equivalent kelvin give the same distribution."""
        in_ev = self.build()
        kelvin = (self.TEMPERATURE * u.eV / const.k_B).to(u.K)
        in_kelvin = pic_thomson.from_moments(
            [[self.DENSITY]],
            [[kelvin.value]] * u.K,
            [[self.DRIFT]],
            t=[0.0],
            x=[0.0],
            label="e-",
            v=in_ev.v,
        )
        np.testing.assert_allclose(in_kelvin.f, in_ev.f, rtol=1e-9)

    def test_a_bare_temperature_is_read_as_kelvin(self):
        bare = pic_thomson.from_moments(
            [[self.DENSITY]],
            [[1.0e6]],
            t=[0.0],
            x=[0.0],
            label="e-",
        )
        _, _, width = self.moments(bare)
        expected = np.sqrt(const.k_B.si.value * 1.0e6 / const.m_e.si.value)
        np.testing.assert_allclose(width, expected, rtol=1e-6)

    def test_rejects_a_temperature_that_is_not_one(self):
        with pytest.raises(u.UnitConversionError, match="K or in energy units"):
            pic_thomson.from_moments(
                [[self.DENSITY]],
                [[1.0]] * u.m,
                t=[0.0],
                x=[0.0],
                label="e-",
            )

    def test_mass_sets_the_width(self):
        """
        The thermal spread is sqrt(kT/m), so a hundredfold mass gives a tenfold
        narrower distribution at the same temperature.
        """
        light = self.build(label="p+", mass=const.m_p)
        heavy = self.build(label="p+", mass=100 * const.m_p)
        _, _, width_light = self.moments(light)
        _, _, width_heavy = self.moments(heavy)
        np.testing.assert_allclose(width_light / width_heavy, 10.0, rtol=1e-3)

    def test_default_velocity_axis_contains_the_distribution(self):
        phase_space = self.build()
        assert phase_space.meta["moment_recovery_error"] < 1e-6
        assert phase_space.v.min() < self.DRIFT < phase_space.v.max()

    def test_warns_when_the_grid_clips_the_distribution(self):
        """
        A supplied axis that does not contain the Maxwellian holds less plasma
        than the moments describe, and nothing downstream would notice.
        """
        thermal = np.sqrt((self.TEMPERATURE * u.eV).to_value(u.J) / const.m_e.si.value)
        with pytest.warns(RuntimeWarning, match="does not contain the reconstructed"):
            pic_thomson.from_moments(
                [[self.DENSITY]],
                [[self.TEMPERATURE]] * u.eV,
                [[self.DRIFT]],
                t=[0.0],
                x=[0.0],
                label="e-",
                v=np.linspace(self.DRIFT - thermal, self.DRIFT + thermal, 128),
            )

    def test_records_an_si_density_scale(self):
        """The driver should not need to be told; f is already in SI."""
        assert self.build().meta["reference_density"] == 1.0

    def test_rejects_moments_of_different_shapes(self):
        with pytest.raises(ValueError, match="must share a shape"):
            pic_thomson.from_moments(
                [[1e25, 1e25]],
                [[100.0]] * u.eV,
                t=[0.0],
                x=[0.0, 1.0],
                label="e-",
            )

    def test_rejects_a_zero_temperature_where_there_is_plasma(self):
        with pytest.raises(ValueError, match="temperature must be positive"):
            pic_thomson.from_moments(
                [[self.DENSITY]],
                [[0.0]] * u.eV,
                t=[0.0],
                x=[0.0],
                label="e-",
            )

    def test_allows_a_zero_temperature_where_there_is_none(self):
        """Vacuum cells carry no plasma, so their temperature is meaningless."""
        phase_space = pic_thomson.from_moments(
            [[self.DENSITY, 0.0]],
            [[self.TEMPERATURE, 0.0]] * u.eV,
            t=[0.0],
            x=[0.0, 1.0],
            label="e-",
        )
        density = pic_thomson.number_density(phase_space.f, phase_space.v)
        np.testing.assert_allclose(density[0, 0], self.DENSITY, rtol=1e-6)
        assert density[0, 1] == 0.0

    def test_reports_an_empty_species(self):
        with pytest.raises(ValueError, match="zero density everywhere"):
            pic_thomson.from_moments(
                [[0.0]],
                [[self.TEMPERATURE]] * u.eV,
                t=[0.0],
                x=[0.0],
                label="e-",
            )

    def test_tapering_a_reconstructed_maxwellian_fabricates_a_pedestal(self):
        """
        The taper exists to replace the discontinuity where macroparticle shot
        noise meets the edge of the velocity grid. A reconstructed distribution
        has no shot noise and no such edge, so tapering it only invents a
        pedestal -- which is why these phase spaces want ``taper_threshold=None``.
        """
        phase_space = self.build()
        with pytest.warns(RuntimeWarning, match="fabricating a pedestal"):
            pic_thomson.condition_phase_space(phase_space)

        untapered = pic_thomson.condition_phase_space(phase_space, taper_threshold=None)
        _, _, width = self.moments(untapered)
        expected = np.sqrt((self.TEMPERATURE * u.eV).to_value(u.J) / const.m_e.si.value)
        np.testing.assert_allclose(width, expected, rtol=1e-6)

    def test_feeds_the_driver(self):
        """A fluid species has to be usable exactly like a kinetic one."""
        pytest.importorskip("numba")
        times = np.linspace(0.0, 1e-9, 3)
        positions = np.linspace(0.0, 1e-3, 4)
        shape = (times.size, positions.size)
        electrons = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 300.0) * u.eV,
            np.zeros(shape),
            t=times,
            x=positions,
            label="e-",
        )
        ions = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 30.0) * u.eV,
            np.zeros(shape),
            t=times,
            x=positions,
            label="p+",
            mass=const.m_p,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            spectra = pic_thomson.spectra_from_phase_spaces(
                electrons=electrons,
                ions=[ions],
                position=positions[1],
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(480, 580, 120) * u.nm,
                electron_conditioning={"taper_threshold": None},
                ion_conditioning={"taper_threshold": None},
                progress=False,
            )
        assert spectra.n_time == times.size
        assert np.all(np.isfinite(spectra.epw))
        np.testing.assert_allclose(spectra.electron_density, 1e25, rtol=1e-6)


# ---------------------------------------------------------------------------
# Hybrid WarpX: electrons that exist only as moments
# ---------------------------------------------------------------------------

HYBRID_CELLS = 128
HYBRID_LENGTH = 2.0e-3  # m
HYBRID_DENSITY = 1.0e25  # m^-3
HYBRID_TEMPERATURE = 100.0  # eV
HYBRID_CURRENT = 3.0e4  # A/m^2, deposited by the ions


class FakeMeshDataset:
    """A yt dataset carrying mesh fields but no particles."""

    def __init__(self, fields, dimensions=(HYBRID_CELLS, 1, 1)):
        self.field_list = [("boxlib", name) for name in fields]
        self.particle_types = ()
        self.domain_dimensions = np.asarray(dimensions)


def fake_hybrid_fields(
    monkeypatch,
    *,
    fields=("rho", "Te", "jz", "Bx", "By"),
    density=HYBRID_DENSITY,
    temperature=HYBRID_TEMPERATURE,
    current=HYBRID_CURRENT,
    n_spatial=1,
):
    """
    Stand in for the yt layer with uniform hybrid moments, so the reader's
    arithmetic can be checked against values worked out by hand.
    """
    shape = (HYBRID_CELLS,) * n_spatial
    resolved = [2] if n_spatial == 1 else [0, 2][:n_spatial]
    spacing = dict.fromkeys(resolved, HYBRID_LENGTH / HYBRID_CELLS)
    charge = const.e.si.value

    def frame(_yt, plotfile, names):
        index = int(str(plotfile).rsplit("diag1", 1)[1])
        values = {
            "rho": np.full(shape, charge * density),
            "Te": np.full(shape, temperature),
            "jz": np.full(shape, current),
            "Bx": np.zeros(shape),
            "By": np.zeros(shape),
            "Bz": np.zeros(shape),
        }
        left = np.zeros(3)
        right = np.array([1.0, 1.0, 1.0])
        for k in resolved:
            right[k] = HYBRID_LENGTH
        return (
            {name: values[name] for name in names},
            index * 1e-12,
            (left, right),
            spacing,
            resolved,
        )

    monkeypatch.setattr(
        pic_thomson, "_load_yt", lambda: FakeYt(FakeMeshDataset(fields))
    )
    monkeypatch.setattr(pic_thomson, "_warpx_field_frame", frame)


class TestHybridElectrons:
    """
    A hybrid run has no electron macroparticles: quasineutrality fixes the
    density, the closure or the energy equation fixes the temperature, and the
    current fixes the drift. Each of those has to come out exactly right.
    """

    def read(self, tmp_path, **kwargs):
        diags = make_warpx_plotfiles(tmp_path, n_frames=kwargs.pop("n_frames", 2))
        return pic_thomson.read_warpx_hybrid_electrons(diags, progress=False, **kwargs)

    def test_density_comes_from_quasineutrality(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path)
        density = pic_thomson.number_density(phase_space.f, phase_space.v)
        np.testing.assert_allclose(density, HYBRID_DENSITY, rtol=1e-6)
        assert phase_space.is_electron
        assert phase_space.meta["reference_density"] == 1.0

    def test_temperature_comes_from_the_solved_field(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path)
        f = phase_space.f[0, :, 0]
        v = phase_space.v
        total = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / total
        width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / total)
        expected = np.sqrt(
            (HYBRID_TEMPERATURE * u.eV).to_value(u.J) / const.m_e.si.value
        )
        np.testing.assert_allclose(width, expected, rtol=1e-6)
        assert phase_space.meta["temperature_source"] == "Te"

    def test_drift_comes_from_the_deposited_ion_current(self, tmp_path, monkeypatch):
        r"""
        In 1-D, :math:`(\nabla \times \vec{B})_z` vanishes identically, so the
        total current along the axis is zero and the electrons must exactly
        counterstream the ions: :math:`u_e = j_z / (e n_e)`.
        """
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path)
        f = phase_space.f[0, :, 0]
        v = phase_space.v
        mean = np.trapezoid(f * v, v) / np.trapezoid(f, v)
        expected = HYBRID_CURRENT / (const.e.si.value * HYBRID_DENSITY)
        np.testing.assert_allclose(mean, expected, rtol=1e-6)

    def test_one_dimensional_ampere_needs_no_magnetic_field(
        self, tmp_path, monkeypatch
    ):
        """
        The curl term is identically zero along the axis of a 1-D run, so the
        reader should not ask for B at all -- and the two drift routes must
        agree exactly.
        """
        fake_hybrid_fields(monkeypatch, fields=("rho", "Te", "jz"))
        ampere = self.read(tmp_path, drift="ampere")
        ion_current = self.read(tmp_path, drift="ion_current")
        assert ampere.meta["magnetic_fields"] == []
        np.testing.assert_array_equal(ampere.f, ion_current.f)

    def test_drift_can_be_switched_off(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path, drift="zero")
        f = phase_space.f[0, :, 0]
        v = phase_space.v
        mean = np.trapezoid(f * v, v) / np.trapezoid(f, v)
        assert abs(mean) < 1e-6 * np.sqrt(
            (HYBRID_TEMPERATURE * u.eV).to_value(u.J) / const.m_e.si.value
        )

    def test_falls_back_to_the_barotropic_closure(self, tmp_path, monkeypatch):
        r"""
        Runs that do not solve an electron energy equation have only the
        closure :math:`T_e = T_{e0} (n_e/n_0)^{\gamma-1}`.
        """
        fake_hybrid_fields(monkeypatch, fields=("rho", "jz"))
        closure = {"T_e0": 40.0, "n0_ref": HYBRID_DENSITY / 8.0, "gamma": 5.0 / 3.0}
        phase_space = self.read(tmp_path, closure=closure)

        f = phase_space.f[0, :, 0]
        v = phase_space.v
        total = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / total
        width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / total)
        # n_e / n_0 = 8, and 8^(2/3) = 4.
        expected_temperature = 40.0 * 4.0
        expected = np.sqrt(
            (expected_temperature * u.eV).to_value(u.J) / const.m_e.si.value
        )
        np.testing.assert_allclose(width, expected, rtol=1e-6)
        assert phase_space.meta["temperature_source"] == "barotropic closure"

    def test_reports_a_missing_temperature_with_no_closure(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch, fields=("rho", "jz"))
        with pytest.raises(KeyError, match="no 'Te' field and no closure"):
            self.read(tmp_path)

    def test_rejects_an_incomplete_closure(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch, fields=("rho", "jz"))
        with pytest.raises(ValueError, match=r"closure is missing \['gamma'\]"):
            self.read(tmp_path, closure={"T_e0": 40.0, "n0_ref": 1e25})

    def test_rejects_a_run_that_is_not_hybrid(self, tmp_path, monkeypatch):
        """
        An explicit run carries electron macroparticles too, so its total rho
        is near zero rather than e * n_e.
        """
        fake_hybrid_fields(monkeypatch, density=0.0)
        with pytest.raises(ValueError, match="nowhere positive"):
            self.read(tmp_path)

    def test_reduces_to_a_point(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path, position=HYBRID_LENGTH / 2)
        assert phase_space.shape[2] == 1

    def test_shares_a_time_axis_with_the_ions(self, tmp_path, monkeypatch):
        """The driver needs one time axis, and both readers must produce it."""
        fake_hybrid_fields(monkeypatch)
        phase_space = self.read(tmp_path, n_frames=3)
        np.testing.assert_allclose(phase_space.t, [0.0, 1e-12, 2e-12])

    def test_rejects_an_unknown_drift(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        with pytest.raises(ValueError, match="drift must be"):
            self.read(tmp_path, drift="guess")

    def test_refuses_to_guess_the_projection_beyond_one_dimension(
        self, tmp_path, monkeypatch
    ):
        fake_hybrid_fields(monkeypatch, n_spatial=2)
        with pytest.raises(ValueError, match="scattering direction has to be given"):
            self.read(tmp_path)


class TestCurlAlong:
    r"""
    The curl term in Ohm's law is what makes the drift exact in more than one
    dimension, and identically zero along the axis of a 1-D run.
    """

    def test_vanishes_along_the_axis_of_a_one_dimensional_run(self):
        magnetic = {0: np.linspace(0.0, 1.0, 16), 1: np.linspace(1.0, 0.0, 16)}
        result = pic_thomson._curl_along((0.0, 0.0, 1.0), magnetic, {2: 0.1}, [2])
        assert np.all(np.asarray(result) == 0.0)

    def test_names_no_field_it_does_not_differentiate(self):
        # Along z, with only z resolved: nothing survives.
        assert pic_thomson._curl_needs((0.0, 0.0, 1.0), [2]) == set()
        # Along x, with only z resolved: only dBy/dz survives.
        assert pic_thomson._curl_needs((1.0, 0.0, 0.0), [2]) == {1}

    def test_matches_an_analytic_curl(self):
        r"""For :math:`B_y = b z`, :math:`(\nabla \times B)_x = -b`."""
        spacing = 0.25
        z = spacing * np.arange(32)
        slope = 3.0
        magnetic = {1: slope * z}
        result = pic_thomson._curl_along((1.0, 0.0, 0.0), magnetic, {2: spacing}, [2])
        np.testing.assert_allclose(result, -slope, rtol=1e-9)


# ---------------------------------------------------------------------------
# openPMD phase space, as WarpX's ParticleHistogram2D writes it
# ---------------------------------------------------------------------------

OPENPMD_DENSITY = 1.0e24  # m^-3
OPENPMD_LENGTH = 1.0e-3  # m
OPENPMD_DRIFT = 0.05  # v/c
OPENPMD_SPREAD = 0.02  # v/c


def write_openpmd_histogram(
    directory,
    *,
    step,
    time,
    data,
    ordinate_range,
    abscissa_range,
    ordinate_function="(1*uz)/sqrt(1+ux*ux+uy*uy+uz*uz)",
    abscissa_function="z",
    filter_function="",
):
    """
    Write one openPMD file in the layout ``ParticleHistogram2D`` produces.

    Every attribute here was read back off a file a real WarpX run wrote, so a
    reader that satisfies this fixture satisfies the diagnostic.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n_ord, n_abs = data.shape
    ord_lo, ord_hi = ordinate_range
    abs_lo, abs_hi = abscissa_range

    with h5py.File(directory / f"openpmd_{step:06d}.h5", "w") as handle:
        handle.attrs["basePath"] = np.bytes_(b"/data/%T/")
        handle.attrs["meshesPath"] = np.bytes_(b"meshes/")
        handle.attrs["openPMD"] = np.bytes_(b"1.1.0")
        handle.attrs["iterationEncoding"] = np.bytes_(b"fileBased")
        handle.attrs["iterationFormat"] = np.bytes_(b"openpmd_%06T")

        iteration = handle.create_group(f"data/{step}")
        iteration.attrs["time"] = np.float64(time)
        iteration.attrs["dt"] = np.float64(1.0)
        iteration.attrs["timeUnitSI"] = np.float64(1.0)

        mesh = iteration.create_group("meshes")
        record = mesh.create_dataset("data", data=data.astype(np.float64))
        record.attrs["axisLabels"] = np.array(
            [ordinate_function.encode(), abscissa_function.encode()]
        )
        record.attrs["dataOrder"] = np.bytes_(b"C")
        record.attrs["geometry"] = np.bytes_(b"cartesian")
        record.attrs["gridGlobalOffset"] = np.array([ord_lo, abs_lo])
        record.attrs["gridSpacing"] = np.array(
            [(ord_hi - ord_lo) / n_ord, (abs_hi - abs_lo) / n_abs]
        )
        record.attrs["gridUnitSI"] = np.float64(1.0)
        record.attrs["position"] = np.array([0.5, 0.5])
        record.attrs["unitSI"] = np.float64(1.0)
        record.attrs["function_abscissa"] = np.bytes_(abscissa_function.encode())
        record.attrs["function_ordinate"] = np.bytes_(ordinate_function.encode())
        record.attrs["filter"] = np.bytes_(filter_function.encode())


def make_openpmd_run(
    directory, *, n_frames=2, n_ord=128, n_abs=64, ord_range=(-0.2, 0.2)
):
    """
    A drifting Gaussian in v/c, uniform in position, holding a known number of
    physical particles -- the same thing the WarpX test run produced.
    """
    ord_lo, ord_hi = ord_range
    edges = np.linspace(ord_lo, ord_hi, n_ord + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    profile = np.exp(-0.5 * ((centres - OPENPMD_DRIFT) / OPENPMD_SPREAD) ** 2)
    profile /= profile.sum()

    total = OPENPMD_DENSITY * OPENPMD_LENGTH  # particles per m^2 of cross-section
    data = np.outer(profile, np.full(n_abs, total / n_abs))
    for step in range(n_frames):
        write_openpmd_histogram(
            directory,
            step=5 * step,
            time=step * 1.3e-13,
            data=data,
            ordinate_range=ord_range,
            abscissa_range=(0.0, OPENPMD_LENGTH),
        )
    return Path(directory)


class TestOpenPMDReader:
    def test_recovers_density_drift_and_spread(self, tmp_path):
        """
        Every number here is fixed by the fixture, so the reader's axis
        construction, unit conversion, and bin-volume normalisation are all
        pinned at once.
        """
        path = make_openpmd_run(tmp_path / "eps")
        phase_space = pic_thomson.read_openpmd_phase_space(
            path, label="e-", progress=False
        )
        speed_of_light = const.c.si.value

        density = pic_thomson.number_density(phase_space.f, phase_space.v)
        np.testing.assert_allclose(density, OPENPMD_DENSITY, rtol=2e-3)

        f = phase_space.f[0].sum(axis=1)
        v = phase_space.v
        mean = np.trapezoid(f * v, v) / np.trapezoid(f, v)
        width = np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / np.trapezoid(f, v))
        np.testing.assert_allclose(mean, OPENPMD_DRIFT * speed_of_light, rtol=1e-3)
        np.testing.assert_allclose(width, OPENPMD_SPREAD * speed_of_light, rtol=1e-2)

    def test_axes_and_times(self, tmp_path):
        path = make_openpmd_run(tmp_path / "eps", n_frames=3)
        phase_space = pic_thomson.read_openpmd_phase_space(
            path, label="e-", progress=False
        )
        assert phase_space.shape == (3, 128, 64)
        np.testing.assert_allclose(phase_space.t, [0.0, 1.3e-13, 2.6e-13])
        # Bin centres, not edges: half a bin in from each end.
        step = OPENPMD_LENGTH / 64
        np.testing.assert_allclose(phase_space.x[0], step / 2, rtol=1e-9)
        np.testing.assert_allclose(
            phase_space.x[-1], OPENPMD_LENGTH - step / 2, rtol=1e-9
        )

    def test_records_the_deck_expressions(self, tmp_path):
        """
        The expressions are the only record of what the axes mean, so they have
        to survive into the metadata.
        """
        path = make_openpmd_run(tmp_path / "eps")
        phase_space = pic_thomson.read_openpmd_phase_space(
            path, label="e-", progress=False
        )
        assert phase_space.meta["function_abscissa"] == "z"
        assert "sqrt(1+ux*ux+uy*uy+uz*uz)" in phase_space.meta["function_ordinate"]
        assert phase_space.meta["reference_density"] == 1.0

    def test_transverse_area_scales_the_density(self, tmp_path):
        path = make_openpmd_run(tmp_path / "eps")
        unit = pic_thomson.read_openpmd_phase_space(
            path, label="e-", transverse_area=1.0, progress=False
        )
        slab = pic_thomson.read_openpmd_phase_space(
            path, label="e-", transverse_area=1e-4, progress=False
        )
        np.testing.assert_allclose(slab.f, unit.f * 1e4, rtol=1e-12)

    def test_reduces_to_a_point(self, tmp_path):
        path = make_openpmd_run(tmp_path / "eps")
        phase_space = pic_thomson.read_openpmd_phase_space(
            path, label="e-", position=OPENPMD_LENGTH / 2, progress=False
        )
        assert phase_space.shape[2] == 1

    def test_selects_requested_files(self, tmp_path):
        path = make_openpmd_run(tmp_path / "eps", n_frames=4)
        phase_space = pic_thomson.read_openpmd_phase_space(
            path, label="e-", timesteps=[0, 2], progress=False
        )
        assert phase_space.shape[0] == 2

    def test_proper_velocity_carries_the_jacobian(self, tmp_path):
        r"""
        An axis in :math:`u = \gamma v/c` is not uniform in :math:`v`, so the
        density per unit velocity has to pick up :math:`\gamma^3/c`.
        """
        path = make_openpmd_run(tmp_path / "eps", ord_range=(-2.0, 2.0))
        proper = pic_thomson.read_openpmd_phase_space(
            path, label="e-", velocity_kind="proper", progress=False
        )
        # The Jacobian is exactly what keeps the zeroth moment a density.
        density = pic_thomson.number_density(proper.f, proper.v)
        np.testing.assert_allclose(density, OPENPMD_DENSITY, rtol=2e-3)
        assert proper.v.max() < const.c.si.value

    def test_velocity_on_the_abscissa_is_transposed(self, tmp_path):
        directory = tmp_path / "eps"
        data = np.arange(4 * 3, dtype=np.float64).reshape(4, 3)
        write_openpmd_histogram(
            directory,
            step=0,
            time=0.0,
            data=data,
            ordinate_range=(0.0, OPENPMD_LENGTH),
            abscissa_range=(-0.1, 0.1),
            ordinate_function="z",
            abscissa_function="(1*uz)/sqrt(1+ux*ux+uy*uy+uz*uz)",
        )
        phase_space = pic_thomson.read_openpmd_phase_space(
            directory, label="e-", velocity_axis="abscissa", progress=False
        )
        # (n_time, n_v, n_x) with velocity now the length-3 axis.
        assert phase_space.shape == (1, 3, 4)

    def test_reports_an_empty_directory(self, tmp_path):
        (tmp_path / "eps").mkdir()
        with pytest.raises(FileNotFoundError, match="no files matching"):
            pic_thomson.read_openpmd_phase_space(
                tmp_path / "eps", label="e-", progress=False
            )

    def test_rejects_a_changing_grid(self, tmp_path):
        directory = tmp_path / "eps"
        make_openpmd_run(directory, n_frames=1)
        write_openpmd_histogram(
            directory,
            step=5,
            time=1e-13,
            data=np.ones((128, 64)),
            ordinate_range=(-0.5, 0.5),
            abscissa_range=(0.0, OPENPMD_LENGTH),
        )
        with pytest.raises(ValueError, match="binned on a different grid"):
            pic_thomson.read_openpmd_phase_space(directory, label="e-", progress=False)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"velocity_axis": "middle"}, "velocity_axis must be"),
            ({"velocity_kind": "momentum"}, "velocity_kind must be"),
        ],
    )
    def test_rejects_bad_settings(self, tmp_path, kwargs, match):
        path = make_openpmd_run(tmp_path / "eps", n_frames=1)
        with pytest.raises(ValueError, match=match):
            pic_thomson.read_openpmd_phase_space(
                path, label="e-", progress=False, **kwargs
            )


class TestHistogram2DDeckBlock:
    SETTINGS = {
        "position_range": (0.0, 1.0e-3),
        "velocity_range": (-4.0e7, 4.0e7),
        "n_position_bins": 64,
        "n_velocity_bins": 128,
    }

    def test_names_every_required_option(self):
        block, _ = histogram2d_deck_block("eps", "electrons", **self.SETTINGS)
        for required in (
            "eps.type = ParticleHistogram2D",
            "eps.species = electrons",
            "eps.bin_number_abs = 64",
            "eps.bin_number_ord = 128",
            "eps.histogram_function_abs(t,x,y,z,ux,uy,uz,w)",
            "eps.histogram_function_ord(t,x,y,z,ux,uy,uz,w)",
        ):
            assert required in block

    def test_states_the_value_function_explicitly(self):
        """
        WarpX reads value_function into m_do_parser_value and then never checks
        it, so leaving it out makes the kernel call a null ParserExecutor and
        fill the histogram with uninitialised memory. Stating the documented
        default is the workaround.
        """
        block, _ = histogram2d_deck_block("eps", "electrons", **self.SETTINGS)
        assert "eps.value_function(t,x,y,z,ux,uy,uz,w) = w" in block

    def test_keeps_the_output_readable_without_openpmd_api(self):
        block, _ = histogram2d_deck_block("eps", "electrons", **self.SETTINGS)
        assert "eps.openpmd_backend = h5" in block

    def test_velocity_expression_is_the_lab_frame_projection(self):
        """
        WarpX gives the parser ux as gamma*vx/c, so dividing the projection by
        gamma is what turns it into the velocity itself.
        """
        block, _ = histogram2d_deck_block(
            "eps", "electrons", scatter_direction=(0, 0, 1), **self.SETTINGS
        )
        assert "(1*uz)/sqrt(1+ux*ux+uy*uy+uz*uz)" in block
        # Components k does not touch are dropped rather than written as 0*ux.
        assert "0*ux" not in block

    def test_normalises_the_scattering_direction(self):
        block, _ = histogram2d_deck_block(
            "eps", "electrons", scatter_direction=(0, 3, 4), **self.SETTINGS
        )
        assert "0.6*uy + 0.8*uz" in block

    def test_velocity_bounds_are_in_units_of_c(self):
        block, _ = histogram2d_deck_block("eps", "electrons", **self.SETTINGS)
        expected = 4.0e7 / const.c.si.value
        line = next(entry for entry in block.splitlines() if "bin_max_ord" in entry)
        np.testing.assert_allclose(float(line.split("=")[1]), expected, rtol=1e-9)

    def test_slab_becomes_a_filter_and_an_area(self):
        block, area = histogram2d_deck_block(
            "eps",
            "electrons",
            transverse_slab={"y": (0.0, 50e-6)},
            **self.SETTINGS,
        )
        assert "eps.filter_function(t,x,y,z,ux,uy,uz,w) = (y>-5e-05)*(y<5e-05)" in block
        np.testing.assert_allclose(area, 1e-4)

    def test_a_two_axis_slab_multiplies_the_area(self):
        _, area = histogram2d_deck_block(
            "eps",
            "electrons",
            transverse_slab={"x": (0.0, 1e-3), "y": (0.0, 2e-3)},
            **self.SETTINGS,
        )
        np.testing.assert_allclose(area, (2e-3) * (4e-3))

    def test_unfiltered_is_one_square_metre(self):
        """WarpX gives the directions a run does not resolve an extent of 1 m."""
        _, area = histogram2d_deck_block("eps", "electrons", **self.SETTINGS)
        assert area == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"position_axis": "q"}, "position_axis must be"),
            ({"scatter_direction": (0, 0, 0)}, "zero vector"),
            ({"scatter_direction": (1, 0)}, "must be a 3-vector"),
            ({"transverse_slab": {"q": (0.0, 1.0)}}, "keys must be"),
            ({"transverse_slab": {"y": (0.0, -1.0)}}, "half-width must be positive"),
        ],
    )
    def test_rejects_bad_settings(self, kwargs, match):
        settings = {**self.SETTINGS, **kwargs}
        with pytest.raises(ValueError, match=match):
            histogram2d_deck_block("eps", "electrons", **settings)

    def test_rejects_inverted_ranges(self):
        with pytest.raises(ValueError, match=r"\(low, high\)"):
            histogram2d_deck_block(
                "eps",
                "electrons",
                position_range=(1.0e-3, 0.0),
                velocity_range=(-4.0e7, 4.0e7),
            )

    def test_the_block_and_the_reader_agree_on_the_grid(self, tmp_path):
        """
        The deck fixes the axes and the reader reconstructs them; if the two
        ever disagree the spectra are wrong with nothing to show for it. Round
        trip the generated numbers through a file to pin them together.
        """
        block, area = histogram2d_deck_block(
            "eps",
            "electrons",
            scatter_direction=(0, 0, 1),
            transverse_slab={"y": (0.0, 50e-6)},
            **self.SETTINGS,
        )
        settings: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith("eps.") and "=" in line:
                key, _, value = line.partition("=")
                settings[key.strip().removeprefix("eps.")] = value.strip()

        n_ord = int(settings["bin_number_ord"])
        n_abs = int(settings["bin_number_abs"])
        directory = tmp_path / "eps"
        write_openpmd_histogram(
            directory,
            step=0,
            time=0.0,
            data=np.ones((n_ord, n_abs)),
            ordinate_range=(
                float(settings["bin_min_ord"]),
                float(settings["bin_max_ord"]),
            ),
            abscissa_range=(
                float(settings["bin_min_abs"]),
                float(settings["bin_max_abs"]),
            ),
        )
        phase_space = pic_thomson.read_openpmd_phase_space(
            directory, label="e-", transverse_area=area, progress=False
        )

        assert phase_space.shape == (1, n_ord, n_abs)
        # The reader's axes must span exactly what the deck asked for, to
        # within the half bin that separates a centre from an edge.
        v_step = float(np.diff(phase_space.v)[0])
        x_step = float(np.diff(phase_space.x)[0])
        np.testing.assert_allclose(
            phase_space.v[0] - v_step / 2, self.SETTINGS["velocity_range"][0], rtol=1e-9
        )
        np.testing.assert_allclose(
            phase_space.v[-1] + v_step / 2,
            self.SETTINGS["velocity_range"][1],
            rtol=1e-9,
        )
        np.testing.assert_allclose(
            phase_space.x[0] - x_step / 2,
            self.SETTINGS["position_range"][0],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            phase_space.x[-1] + x_step / 2,
            self.SETTINGS["position_range"][1],
            rtol=1e-9,
        )


class TestHybridTwoDimensional:
    """
    A 2-D XZ run is stored on grid axes 0 and 1 while its physical directions
    are x and z, so the domain edges cannot be indexed by physical direction --
    that returns the unresolved third axis, which WarpX gives an extent of 1 m.
    """

    def read(self, tmp_path, monkeypatch, **kwargs):
        fake_hybrid_fields(monkeypatch, n_spatial=2)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        return pic_thomson.read_warpx_hybrid_electrons(
            diags,
            scatter_direction=(0, 0, 1),
            axis=1,
            progress=False,
            **kwargs,
        )

    def test_spatial_axis_spans_the_resolved_direction(self, tmp_path, monkeypatch):
        phase_space = self.read(tmp_path, monkeypatch)
        assert phase_space.x.size == HYBRID_CELLS
        step = HYBRID_LENGTH / HYBRID_CELLS
        # The diagnostic axis must span the deck's z extent, not the 1 m that
        # WarpX gives the direction the run does not resolve.
        np.testing.assert_allclose(phase_space.x[0], step / 2, rtol=1e-9)
        np.testing.assert_allclose(
            phase_space.x[-1], HYBRID_LENGTH - step / 2, rtol=1e-9
        )

    def test_moments_survive_the_transverse_slab(self, tmp_path, monkeypatch):
        phase_space = self.read(tmp_path, monkeypatch)
        density = pic_thomson.number_density(phase_space.f, phase_space.v)
        np.testing.assert_allclose(density, HYBRID_DENSITY, rtol=1e-6)

    def test_slab_and_chord_agree_on_a_uniform_field(self, tmp_path, monkeypatch):
        """Averaging, not summing: the density must not depend on cells kept."""
        slab = self.read(tmp_path, monkeypatch, transverse_reduction="slab")
        chord = self.read(tmp_path, monkeypatch, transverse_reduction="chord")
        np.testing.assert_allclose(
            pic_thomson.number_density(slab.f, slab.v),
            pic_thomson.number_density(chord.f, chord.v),
            rtol=1e-9,
        )

    def test_transverse_position_selects_a_slab(self, tmp_path, monkeypatch):
        phase_space = self.read(
            tmp_path,
            monkeypatch,
            transverse_position=[HYBRID_LENGTH / 4],
            slab_halfwidth=HYBRID_LENGTH / 16,
        )
        assert phase_space.meta["transverse_reduction"] == "slab"
        assert phase_space.x.size == HYBRID_CELLS


class TestPerSpeciesVelocityScaling:
    r"""
    A hybrid run's electrons come from `from_moments` at the real electron mass
    and are already physical; its ions carry whatever reduced mass the
    simulation gave them. One factor for both would be wrong.
    """

    R = 18.36  # m_p / (100 m_e)

    def build(self):
        times = np.linspace(0.0, 1e-9, 2)
        positions = np.linspace(0.0, 1e-3, 3)
        shape = (times.size, positions.size)
        electrons = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 200.0) * u.eV,
            t=times,
            x=positions,
            label="e-",
            # alpha is about 5.5 here, so the EPW resonance sits near 6.3
            # thermal speeds and the default headroom of 6 stops just short.
            velocity_headroom=12.0,
        )
        ions = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 20.0) * u.eV,
            t=times,
            x=positions,
            label="p+",
            mass=100 * const.m_e,
        )
        return electrons, ions, positions

    def run(self, **kwargs):
        electrons, ions, positions = self.build()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            return pic_thomson.spectra_from_phase_spaces(
                electrons=electrons,
                ions=[ions],
                position=positions[1],
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(480, 580, 100) * u.nm,
                iaw_wavelengths=np.linspace(528, 536, 100) * u.nm,
                electron_conditioning={"taper_threshold": None},
                ion_conditioning={"taper_threshold": None},
                progress=False,
                **kwargs,
            )

    def test_scaling_the_ions_leaves_the_satellites_alone(self):
        """
        The electron-plasma-wave resonance is set by n_e and the electron
        distribution, neither of which an ion mass convention touches. Only the
        centre of the window moves, because the ion feature lives there.
        """
        pytest.importorskip("numba")
        plain = self.run()
        scaled = self.run(ion_velocity_scale_factor=self.R)
        # The spectrogram stores its axis in metres.
        wavelengths = plain.epw_wavelengths * 1e9
        wings = np.abs(wavelengths - 532.0) > 10.0
        for spectra in (plain, scaled):
            assert np.any(spectra.epw[0][wings] > 0)

        def renormalised(spectra):
            wing = spectra.epw[0][wings]
            return wing / np.trapezoid(wing, wavelengths[wings])

        # Not bit-identical, and should not be: the ions enter the satellites
        # through their susceptibility in epsilon = 1 + chi_e + chi_i, which is
        # near zero at the resonance and so amplifies even a small term. The
        # residual is a few parts in 1e5, against the factor-of-4 change the
        # same rescaling makes to the ion feature.
        np.testing.assert_allclose(renormalised(scaled), renormalised(plain), rtol=1e-3)
        # ...and the satellite sits in the same bin.
        assert np.argmax(scaled.epw[0][wings]) == np.argmax(plain.epw[0][wings])

    def test_scaling_the_ions_narrows_the_ion_feature(self):
        r"""
        The ion-acoustic width goes as :math:`\sqrt{Z k T_e / m_i}`, so mapping
        a 100 m_e ion onto a proton must narrow it by about :math:`\sqrt{R}`.
        """
        pytest.importorskip("numba")
        plain = self.run()
        scaled = self.run(ion_velocity_scale_factor=self.R)

        def width(spectrogram):
            f = spectrogram.iaw[0]
            lam = spectrogram.iaw_wavelengths * 1e9
            total = np.trapezoid(f, lam)
            mean = np.trapezoid(f * lam, lam) / total
            return np.sqrt(np.trapezoid(f * (lam - mean) ** 2, lam) / total)

        assert width(scaled) < width(plain)

    def test_a_global_factor_still_applies_to_both(self):
        pytest.importorskip("numba")
        both = self.run(velocity_scale_factor=self.R)
        assert both.meta["velocity_scale_factor"] == self.R
        assert both.meta["ion_velocity_scale_factor"] == self.R
        ions_only = self.run(ion_velocity_scale_factor=self.R)
        # Scaling the electrons too must move the electron feature.
        assert not np.allclose(both.epw, ions_only.epw)

    def test_ion_factor_is_recorded(self):
        pytest.importorskip("numba")
        spectra = self.run(ion_velocity_scale_factor=self.R)
        assert spectra.meta["velocity_scale_factor"] is None
        assert spectra.meta["ion_velocity_scale_factor"] == self.R


class TestHybridPositionReduction:
    """
    Reducing the moments before building the distribution has to give the same
    answer as reducing after -- the point is only that it does not allocate the
    whole (n_time, n_v, n_x) block on the way.
    """

    def test_matches_reducing_after(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)
        target = HYBRID_LENGTH / 3

        whole = pic_thomson.read_warpx_hybrid_electrons(diags, progress=False)
        after = whole.at_position(target)
        before = pic_thomson.read_warpx_hybrid_electrons(
            diags, position=target, v=whole.v, progress=False
        )

        assert before.shape[2] == 1
        np.testing.assert_array_equal(before.x, after.x)
        np.testing.assert_allclose(before.f, after.f, rtol=1e-12)
        assert before.meta["sampled_index"] == after.meta["sampled_index"]

    def test_reports_a_position_off_the_axis(self, tmp_path, monkeypatch):
        fake_hybrid_fields(monkeypatch)
        diags = make_warpx_plotfiles(tmp_path, n_frames=1)
        with pytest.raises(ValueError, match="outside the diagnostic axis"):
            pic_thomson.read_warpx_hybrid_electrons(
                diags, position=10 * HYBRID_LENGTH, progress=False
            )


class TestAlreadyReducedPhaseSpace:
    """
    Readers reduce to the probe position themselves -- they have to, or they
    build the whole block first -- and each snaps to its own grid. Handing those
    single-point phase spaces back to the driver with the position the caller
    asked for must work.
    """

    def make(self, x):
        return pic_thomson.from_arrays(
            f=np.ones((2, 8, 1)),
            v=np.linspace(-1e6, 1e6, 8),
            x=[x],
            t=[0.0, 1e-9],
            label="e-",
        )

    def test_any_position_selects_the_only_point(self):
        phase_space = self.make(1.234e-3)
        index, sliced = phase_space.slice_position(9.9e-3)
        assert index == 0
        assert sliced.shape == (2, 8)

    def test_at_position_is_the_identity(self):
        phase_space = self.make(1.234e-3)
        again = phase_space.at_position(5.0e-3)
        np.testing.assert_array_equal(again.f, phase_space.f)
        np.testing.assert_array_equal(again.x, phase_space.x)

    def test_a_profile_still_rejects_a_position_off_the_grid(self):
        profile = pic_thomson.from_arrays(
            f=np.ones((2, 8, 4)),
            v=np.linspace(-1e6, 1e6, 8),
            x=np.linspace(0.0, 1e-3, 4),
            t=[0.0, 1e-9],
            label="e-",
        )
        with pytest.raises(ValueError, match="outside the spatial axis"):
            profile.slice_position(5.0e-3)

    def test_species_reduced_to_slightly_different_points_still_run(self):
        """The whole point: two readers snap to grids that differ by a cell."""
        pytest.importorskip("numba")
        times = np.array([0.0, 1e-9])
        shape = (2, 1)
        electrons = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 200.0) * u.eV,
            t=times,
            x=[9.6991e-4],
            label="e-",
        )
        ions = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 20.0) * u.eV,
            t=times,
            x=[9.7100e-4],  # a different reader, a different grid
            label="p+",
            mass=100 * const.m_e,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            spectra = pic_thomson.spectra_from_phase_spaces(
                electrons=electrons,
                ions=[ions],
                position=9.7022e-4,
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(480, 580, 80) * u.nm,
                electron_conditioning={"taper_threshold": None},
                ion_conditioning={"taper_threshold": None},
                progress=False,
            )
        assert np.all(np.isfinite(spectra.epw))


class TestNotchBoundaries:
    r"""
    A notch blocks a closed interval of wavelengths. Locating its edges with
    ``argmin`` rounded each to the nearest bin centre, and the half-open slice
    that followed left the upper endpoint bin unblocked on top of that: on the
    0.4 nm grid used for OSIRIS EPW windows a requested [530, 534] came out as
    [530.2, 533.4]. That matters because the bin immediately outside a notch
    holds the skirt of the central feature, which is orders of magnitude above
    the EPW satellites and takes the whole window once each row is normalised
    to unit area.
    """

    @staticmethod
    def spectrum(wavelengths, notches):
        pytest.importorskip("numba")
        electrons = species_phase_space("e-", 100, const.m_e)
        ions = species_phase_space("p+", 50, const.m_p)
        conditioned = [
            # No taper: these tests are about which wavelength bins the notch
            # covers, and the fixture's grid would trip the pedestal check.
            pic_thomson.condition_phase_space(
                ps.at_position(0.5e-3), taper_threshold=None
            )
            for ps in (electrons, ions)
        ]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            _, spectrum = thomson.arbitrary_forwardmodel(
                wavelengths=wavelengths,
                probe_wavelength=PROBE_WAVELENGTH,
                e_velocity_axes=[conditioned[0].v] * u.m / u.s,
                i_velocity_axes=[conditioned[1].v] * u.m / u.s,
                efn=[conditioned[0].f[0, :, 0]] * u.s / u.m,
                ifn=[conditioned[1].f[0, :, 0]] * u.s / u.m,
                efract=np.array([1.0]),
                ifract=np.array([1.0]),
                n=REFERENCE_DENSITY,
                ion_species=["p+"],
                probe_vec=np.asarray(PROBE_VEC, dtype=float),
                scatter_vec=np.asarray(SCATTER_VEC, dtype=float),
                scattered_power=True,
                notches=notches,
            )
        return np.asarray(spectrum, dtype=float)

    def test_blocks_exactly_the_requested_interval(self):
        # 0.4008 nm bins, deliberately offset so that neither notch edge lands
        # on a bin centre -- which is the case argmin used to get wrong.
        wavelengths = np.linspace(432, 632, 500) * u.nm
        grid = wavelengths.to_value(u.nm)
        spectrum = self.spectrum(wavelengths, np.array([[530.0, 534.0]]) * u.nm)

        inside = (grid >= 530.0) & (grid <= 534.0)
        assert np.all(spectrum[inside] == 0.0)
        # In particular the bin nearest the upper edge, which the half-open
        # slice used to leave in.
        assert grid[inside][-1] == pytest.approx(533.804, abs=1e-3)

    def test_leaves_everything_outside_it_alone(self):
        wavelengths = np.linspace(432, 632, 500) * u.nm
        grid = wavelengths.to_value(u.nm)
        spectrum = self.spectrum(wavelengths, np.array([[530.0, 534.0]]) * u.nm)

        outside = (grid < 530.0) | (grid > 534.0)
        # Nothing beyond the interval was zeroed: argmin could round an edge
        # outwards and take a bin the filter does not cover.
        assert np.all(spectrum[outside] > 0.0)

    def test_an_edge_landing_on_a_bin_centre_is_included(self):
        # 0.5 nm bins, so 530.0 and 534.0 are exactly bin centres.
        wavelengths = np.linspace(430, 630, 401) * u.nm
        grid = wavelengths.to_value(u.nm)
        spectrum = self.spectrum(wavelengths, np.array([[530.0, 534.0]]) * u.nm)

        assert spectrum[np.argmin(np.abs(grid - 530.0))] == 0.0
        assert spectrum[np.argmin(np.abs(grid - 534.0))] == 0.0
        assert spectrum[np.argmin(np.abs(grid - 529.5))] > 0.0
        assert spectrum[np.argmin(np.abs(grid - 534.5))] > 0.0

    def test_several_notches_are_all_applied(self):
        wavelengths = np.linspace(432, 632, 500) * u.nm
        grid = wavelengths.to_value(u.nm)
        spectrum = self.spectrum(
            wavelengths, np.array([[500.0, 505.0], [530.0, 534.0]]) * u.nm
        )
        for lo, hi in ((500.0, 505.0), (530.0, 534.0)):
            assert np.all(spectrum[(grid >= lo) & (grid <= hi)] == 0.0)


class TestSmoothingWidth:
    r"""
    A boxcar is a convolution, so it adds ``iterations * (window^2 - 1) / 12``
    bins^2 of variance to every slice whatever that slice's own width. In
    velocity units that is an additive temperature, and the forward model reads
    the thermal speed straight off the distribution. A window given in bins is
    therefore only meaningful next to the data; ``smoothing_width`` gives it in
    thermal speeds instead.
    """

    @staticmethod
    def narrow_phase_space(n_v=1024, half_width=40.0):
        """A Maxwellian occupying a small part of a wide velocity grid."""
        sigma = sigma_of(40.0, const.m_e)
        v = np.linspace(-half_width * sigma, half_width * sigma, n_v)
        return pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma), n_time=2, n_x=2),
            v=v,
            x=np.linspace(0.0, 1e-3, 2),
            t=np.linspace(0.0, 1e-9, 2),
            label="e-",
        )

    @staticmethod
    def width_of(phase_space, step=0, cell=0):
        f = phase_space.f[step, :, cell]
        v = phase_space.v
        norm = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / norm
        return float(np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / norm))

    def test_a_window_in_bins_inflates_the_temperature(self):
        phase_space = self.narrow_phase_space()
        before = self.width_of(phase_space)
        with pytest.warns(RuntimeWarning, match="additive temperature"):
            conditioned = pic_thomson.condition_phase_space(
                phase_space,
                smoothing_window=40,
                smoothing_iterations=3,
                taper_threshold=None,
            )
        # T scales as the square of the width, so this is a factor of a few in
        # temperature -- exactly the effect that has to be caught.
        assert self.width_of(conditioned) / before > 1.5

    def test_a_width_in_thermal_speeds_does_not(self):
        phase_space = self.narrow_phase_space()
        before = self.width_of(phase_space)
        conditioned = pic_thomson.condition_phase_space(
            phase_space,
            smoothing_width=0.25,
            smoothing_iterations=3,
            taper_threshold=None,
        )
        assert self.width_of(conditioned) / before == pytest.approx(1.0, abs=0.02)

    def test_the_window_follows_the_data_not_the_grid(self):
        """
        The same plasma dumped on a grid twice as wide must get half the window
        in bins -- that is the whole point of asking in thermal speeds.
        """
        windows = []
        for half_width in (20.0, 40.0):
            conditioned = pic_thomson.condition_phase_space(
                self.narrow_phase_space(half_width=half_width),
                smoothing_width=0.25,
                smoothing_iterations=2,
                taper_threshold=None,
            )
            windows.append(conditioned.meta["conditioning"]["smoothing_window"])
        assert windows[0] == pytest.approx(2 * windows[1], rel=0.05)

    def test_the_resolved_window_is_recorded(self):
        conditioned = pic_thomson.condition_phase_space(
            self.narrow_phase_space(),
            smoothing_width=0.5,
            smoothing_iterations=1,
            taper_threshold=None,
        )
        conditioning = conditioned.meta["conditioning"]
        assert conditioning["smoothing_width"] == 0.5
        assert conditioning["smoothing_window"] > 1

    def test_window_and_width_together_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            pic_thomson.condition_phase_space(
                self.narrow_phase_space(),
                smoothing_window=10,
                smoothing_width=0.25,
                smoothing_iterations=1,
            )

    def test_smoothing_is_sized_by_the_narrowest_slice(self):
        """
        Every slice gets a window sized against its own width, so a cold cell
        beside a hot one is not smeared to match the hot one.
        """
        sigma = sigma_of(40.0, const.m_e)
        v = np.linspace(-40 * sigma, 40 * sigma, 1024)
        cold = maxwellian(v, sigma)
        hot = maxwellian(v, 8 * sigma)
        f = np.stack([cold, hot], axis=-1)[np.newaxis, :, :]
        phase_space = pic_thomson.from_arrays(
            f=f,
            v=v,
            x=np.array([0.0, 1e-3]),
            t=np.array([0.0]),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(
            phase_space,
            smoothing_width=0.25,
            smoothing_iterations=3,
            taper_threshold=None,
        )
        assert self.width_of(conditioned, cell=0) / self.width_of(
            phase_space, cell=0
        ) == pytest.approx(1.0, abs=0.02)

    def test_no_smoothing_means_no_window_to_resolve(self):
        # Nothing is smoothed, so nothing needs sizing -- and an unpopulated
        # phase space must not raise on the way past.
        empty = replace(self.narrow_phase_space(), f=np.zeros((2, 1024, 2)))
        conditioned = pic_thomson.condition_phase_space(
            empty, smoothing_iterations=0, taper_threshold=None
        )
        assert conditioned.meta["conditioning"]["smoothing_window"] == 1

    def test_a_species_absent_from_the_point_is_not_an_alarm(self):
        """
        A population with no macroparticles at the sampled point -- a piston
        that has not arrived -- has no width to size anything against. That is
        ordinary, and must not warn.
        """
        empty = replace(self.narrow_phase_space(), f=np.zeros((2, 1024, 2)))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            conditioned = pic_thomson.condition_phase_space(
                empty,
                tail_model=None,
                smoothing_width=0.25,
                smoothing_iterations=2,
                max_taper_width=0.5,
            )
        assert conditioned.meta["conditioning"]["smoothing_window"] == 1
        assert conditioned.meta["conditioning"]["max_taper_bins"] == 1

    def test_a_slice_the_grid_cannot_resolve_still_warns(self):
        # A single occupied bin has weight but no measurable spread, which is
        # not ordinary -- it means the grid cannot resolve the distribution.
        f = np.zeros((1, 1024, 1))
        f[0, 512, 0] = 1.0
        spike = replace(
            self.narrow_phase_space(), f=f, t=np.array([0.0]), x=np.array([0.0])
        )
        with pytest.warns(RuntimeWarning, match="under one velocity bin"):
            conditioned = pic_thomson.condition_phase_space(
                spike,
                smoothing_width=0.25,
                smoothing_iterations=2,
                taper_threshold=None,
            )
        assert conditioned.meta["conditioning"]["smoothing_window"] == 1

    def test_smooth_vdf_stays_silent_unless_asked(self):
        phase_space = self.narrow_phase_space()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            pic_thomson.smooth_vdf(phase_space.f, window=40, iterations=3)


class TestEPWTailCheck:
    r"""
    The EPW satellite sits at the Bohm--Gross resonance, so it reads :math:`f_e`
    at :math:`\sqrt{\alpha^2 + 3}` thermal widths. A PIC histogram is populated
    only as far as its last macroparticle, typically three to five widths --
    reaching :math:`n` widths needs of order :math:`e^{n^2/2}` particles per
    cell. Past that the taper and the floor supply whatever the resonance of
    :math:`|1 - \chi_e/\epsilon|^2` then amplifies into a satellite, and that
    satellite moves when ``max_taper_bins`` moves. It is not a measurement and
    the spectrogram has to say so.
    """

    @staticmethod
    def electrons(reach, T_eV=100.0, n_v=2048):
        """A Maxwellian truncated at *reach* thermal widths."""
        sigma = sigma_of(T_eV, const.m_e)
        v = np.linspace(-reach * sigma, reach * sigma, n_v)
        return pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma), n_time=3, n_x=4),
            v=v,
            x=np.linspace(0.0, 1e-3, 4),
            t=np.linspace(0.0, 2e-9, 3),
            label="e-",
        )

    def spectra(self, reach, T_eV=100.0, **kwargs):
        pytest.importorskip("numba")
        return run_driver(
            electrons=self.electrons(reach, T_eV=T_eV),
            electron_conditioning={"taper_threshold": None},
            ion_conditioning={"taper_threshold": None},
            **kwargs,
        )

    def test_reports_what_the_satellite_needs(self):
        spectra = self.spectra(8.0)
        # sqrt(alpha_conventional^2 + 3), with the model's alpha being sqrt(2)
        # times the conventional one.
        expected = np.sqrt(spectra.alpha_epw**2 / 2 + 3)
        assert spectra.epw_tail_required.shape == (1, spectra.n_time)
        np.testing.assert_allclose(spectra.epw_tail_required[0], expected, rtol=1e-12)

    def test_reports_how_far_the_histogram_reaches(self):
        spectra = self.spectra(8.0)
        assert spectra.epw_tail_ratio.shape == (1, spectra.n_time)
        np.testing.assert_allclose(spectra.epw_tail_ratio, 8.0, rtol=0.02)

    def test_a_reachable_resonance_is_left_alone(self):
        spectra = self.spectra(8.0)
        assert np.all(spectra.epw_resolved)
        assert np.all(np.isfinite(spectra.epw))

    def test_an_unreachable_resonance_is_kept_but_priced(self):
        """
        Since `extend_vdf_tail` fits the tail rather than tapering it, a
        resonance past the sampled data is an extrapolation with a stated
        uncertainty, not an artefact -- so it is kept, and the uncertainty is
        what says whether to believe it.
        """
        # Colder electrons at the same density put the resonance at 4.4
        # widths, and this histogram stops at 4.
        with pytest.warns(RuntimeWarning, match="beyond the last velocity"):
            spectra = self.spectra(4.0, T_eV=20.0)
        assert not np.any(spectra.epw_resolved)
        assert np.all(np.isfinite(spectra.epw))
        assert np.all(spectra.epw_tail_uncertainty >= 1.0)
        # The check does not touch the IAW feature, which reads the bulk.
        assert np.all(np.isfinite(spectra.alpha_epw))

    def test_nothing_extrapolated_is_priced_at_one(self):
        spectra = self.spectra(8.0)
        assert np.all(spectra.epw_resolved)
        np.testing.assert_allclose(spectra.epw_tail_uncertainty, 1.0)

    def test_the_mask_can_still_be_asked_for(self):
        with pytest.warns(RuntimeWarning, match="beyond the last velocity"):
            spectra = self.spectra(4.0, T_eV=20.0, mask_unresolved_epw=True)
        assert not np.any(spectra.epw_resolved)
        assert np.all(np.isnan(spectra.epw))

    def test_the_warning_names_the_sampling_as_the_cause(self):
        with pytest.warns(RuntimeWarning, match="sampled no 'e-' that fast"):
            self.spectra(4.0, T_eV=20.0)

    def test_a_reconstructed_tail_is_named_as_such(self):
        r"""
        `from_moments` has no last macroparticle -- its grid stops where
        ``velocity_headroom`` was told to stop -- so the remedy is a wider grid,
        and the spectrogram records that the population was reconstructed.
        """
        pytest.importorskip("numba")
        times = np.linspace(0.0, 1e-9, 2)
        positions = np.linspace(0.0, 1e-3, 3)
        shape = (times.size, positions.size)
        electrons = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 20.0) * u.eV,
            t=times,
            x=positions,
            label="e-",
            velocity_headroom=4.0,
        )
        ions = pic_thomson.from_moments(
            np.full(shape, 1e25),
            np.full(shape, 20.0) * u.eV,
            t=times,
            x=positions,
            label="p+",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            with pytest.warns(RuntimeWarning, match="widen velocity_headroom"):
                spectra = pic_thomson.spectra_from_phase_spaces(
                    electrons=electrons,
                    ions=[ions],
                    position=positions[1],
                    probe_wavelength=PROBE_WAVELENGTH,
                    epw_wavelengths=EPW_WINDOW,
                    electron_conditioning={"taper_threshold": None},
                    ion_conditioning={"taper_threshold": None},
                    progress=False,
                )
        assert spectra.meta["epw_tail_analytic"] == ["e-"]

    def test_an_absent_population_cannot_fail_the_check(self):
        """
        A population below the presence threshold is not handed to the forward
        model at all, so its tail cannot fabricate anything.
        """
        pytest.importorskip("numba")
        present = self.electrons(8.0)
        absent = replace(
            self.electrons(2.0),
            f=self.electrons(2.0).f * 1e-6,
            label="e- halo",
        )
        spectra = run_driver(
            electrons=[present, absent],
            electron_conditioning={"taper_threshold": None},
            ion_conditioning={"taper_threshold": None},
        )
        assert np.all(spectra.epw_resolved)
        assert np.all(np.isfinite(spectra.epw))

    def test_the_diagnostic_survives_a_round_trip(self, tmp_path):
        spectra = self.spectra(8.0)
        path = tmp_path / "spectra.h5"
        spectra.to_hdf5(path)
        restored = pic_thomson.ThomsonSpectrogram.from_hdf5(path)
        np.testing.assert_allclose(restored.epw_tail_ratio, spectra.epw_tail_ratio)
        np.testing.assert_allclose(
            restored.epw_tail_required, spectra.epw_tail_required
        )
        np.testing.assert_array_equal(restored.epw_resolved, spectra.epw_resolved)
        np.testing.assert_allclose(
            restored.epw_tail_uncertainty, spectra.epw_tail_uncertainty
        )

    def test_a_spectrogram_without_the_diagnostic_is_unassessed(self):
        """
        Files written before the check existed, and spectrograms built by hand,
        must read back as "not assessed" rather than as a verdict.
        """
        spectrogram = synthetic_spectrogram()
        assert spectrogram.epw_tail_ratio.shape == (1, spectrogram.n_time)
        assert np.all(np.isnan(spectrogram.epw_tail_ratio))
        assert np.all(np.isnan(spectrogram.epw_tail_required))
        assert np.all(spectrogram.epw_resolved)


class TestTaperWidth:
    """
    ``max_taper_bins`` has the same trouble as ``smoothing_window``: how much
    width a rolloff of N bins adds depends on how wide the code's velocity grid
    happens to be, not on the plasma. ``max_taper_width`` asks in thermal
    speeds instead.
    """

    @staticmethod
    def phase_space(half_width):
        sigma = sigma_of(40.0, const.m_p)
        v = np.linspace(-half_width * sigma, half_width * sigma, 1024)
        return pic_thomson.from_arrays(
            f=block(maxwellian(v, sigma), n_time=2, n_x=2),
            v=v,
            x=np.linspace(0.0, 1e-3, 2),
            t=np.linspace(0.0, 1e-9, 2),
            label="p+",
        )

    def test_the_rolloff_follows_the_data_not_the_grid(self):
        rolloffs = []
        for half_width in (20.0, 40.0):
            conditioned = pic_thomson.condition_phase_space(
                self.phase_space(half_width),
                tail_model=None,
                max_taper_width=0.5,
                pedestal_warning=None,
            )
            rolloffs.append(conditioned.meta["conditioning"]["max_taper_bins"])
        # Half the grid spacing, so twice the bins -- give or take the rounding
        # of each to a whole number.
        assert rolloffs[0] == pytest.approx(2 * rolloffs[1], abs=1)

    def test_it_keeps_the_second_moment(self):
        phase_space = self.phase_space(40.0)
        v = phase_space.v

        def width(f):
            norm = np.trapezoid(f, v)
            mean = np.trapezoid(f * v, v) / norm
            return float(np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / norm))

        before = width(phase_space.f[0, :, 0])
        conditioned = pic_thomson.condition_phase_space(
            phase_space, tail_model=None, max_taper_width=0.5, pedestal_warning=None
        )
        assert width(conditioned.f[0, :, 0]) / before == pytest.approx(1.0, abs=0.05)

    def test_bins_and_width_together_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            pic_thomson.condition_phase_space(
                self.phase_space(40.0),
                tail_model=None,
                max_taper_bins=8,
                max_taper_width=0.5,
            )


class TestExtendVDFTail:
    r"""
    The tail beyond the last macroparticle is not measured, and the EPW
    satellite reads it. `taper_vdf_edges` filled it with a half-cosine to zero,
    whose amplitude, slope and extent were all free choices;
    `extend_vdf_tail` fills it with the distribution's own fitted Maxwellian,
    which has none.
    """

    SIGMA = 1.0

    @staticmethod
    def grid(half_width=40.0, n_v=1024):
        return np.linspace(-half_width, half_width, n_v)

    @classmethod
    def sampled(cls, n_particles, v, sigma=None, rng=None, drift=0.0):
        """A histogram of *n_particles* drawn from a Maxwellian, as PIC gives."""
        rng = rng or np.random.default_rng(4)
        sigma = cls.SIGMA if sigma is None else sigma
        dv = v[1] - v[0]
        edges = np.append(v - dv / 2, v[-1] + dv / 2)
        draws = rng.normal(drift, sigma, int(n_particles))
        return np.histogram(draws, bins=edges)[0].astype(float)[np.newaxis, :, None]

    @staticmethod
    def truth(v, n_particles, sigma=1.0):
        dv = v[1] - v[0]
        return (
            np.exp(-(v**2) / (2 * sigma**2))
            / (sigma * np.sqrt(2 * np.pi))
            * n_particles
            * dv
        )

    def test_recovers_the_temperature_it_was_drawn_at(self):
        v = self.grid()
        _, info = pic_thomson.extend_vdf_tail(self.sampled(1e5, v), v)
        assert info["tail_over_core_width"] == pytest.approx(1.0, abs=0.01)
        assert info["fallback_fraction"] == 0.0

    def test_extrapolates_well_past_the_last_particle(self):
        """
        The point of the exercise: be right where the satellite reads, which is
        several times beyond anything the run sampled.
        """
        v = self.grid()
        extended, info = pic_thomson.extend_vdf_tail(self.sampled(1e5, v), v)
        truth = self.truth(v, 1e5)
        assert info["join_thermal_speeds"] < 4.0
        # Right to within a factor of two at twelve thermal speeds, which is
        # three times beyond the last macroparticle. The taper's answer there
        # was whatever max_taper_bins happened to be.
        for probe in (5.0, 8.0, 12.0):
            index = int(np.argmin(np.abs(v - probe)))
            assert 0.5 < extended[0, index, 0] / truth[index] < 2.0

    def test_the_reported_uncertainty_bounds_the_error(self):
        r"""
        ``tail_width_error`` is the standard error on :math:`\sigma_t`, and
        :math:`\delta \ln f = (x/\sigma_t)^2 \delta\sigma_t/\sigma_t`. A run too
        thin to constrain the tail has to say so rather than quietly extrapolate.
        """
        v = self.grid()
        extended, info = pic_thomson.extend_vdf_tail(self.sampled(1e4, v), v)
        index = int(np.argmin(np.abs(v - 12.0)))
        observed = extended[0, index, 0] / self.truth(v, 1e4)[index]
        allowed = np.exp(144.0 * info["tail_width_error"])
        assert 1 / allowed <= observed <= allowed
        # And a better-sampled run must be quoted as more certain.
        _, better = pic_thomson.extend_vdf_tail(self.sampled(1e6, v), v)
        assert better["tail_width_error"] < info["tail_width_error"]

    def test_a_resolved_suprathermal_tail_survives(self):
        """
        A fitted tail temperature is worth having only if it can differ from the
        core; otherwise the core temperature would do.
        """
        v = self.grid()
        rng = np.random.default_rng(11)
        dv = v[1] - v[0]
        edges = np.append(v - dv / 2, v[-1] + dv / 2)
        draws = np.concatenate(
            [rng.normal(0.0, 1.0, 500_000), rng.normal(0.0, 2.5, 25_000)]
        )
        f = np.histogram(draws, bins=edges)[0].astype(float)[np.newaxis, :, None]
        _, info = pic_thomson.extend_vdf_tail(f, v)
        assert info["tail_over_core_width"] > 1.1

    def test_it_falls_back_to_the_core_when_there_is_no_tail_to_fit(self):
        v = self.grid()
        extended, info = pic_thomson.extend_vdf_tail(self.sampled(60, v), v)
        # Sixty particles leave nothing to fit on at least one side.
        assert info["fallback_fraction"] > 0.0
        assert np.all(np.isfinite(extended))
        assert np.all(extended >= 0)

    def test_it_anchors_on_the_fit_not_on_the_join_bin(self):
        """
        The join is the outermost bin still holding ``min_counts``, so it is a
        selected upward fluctuation. ``join_step`` records the disagreement, and
        it must sit below one -- anchoring there would bias the whole tail high.
        """
        v = self.grid()
        _, info = pic_thomson.extend_vdf_tail(self.sampled(1e5, v), v)
        assert 0.4 < info["join_step"] < 1.0

    def test_more_particles_buys_more_measured_tail(self):
        """
        What the fraction-of-peak threshold could not do: a count-based join has
        to move outwards when the run is better sampled.
        """
        v = self.grid()
        joins = [
            pic_thomson.extend_vdf_tail(self.sampled(n, v), v)[1]["join_thermal_speeds"]
            for n in (1e4, 1e6)
        ]
        assert joins[1] > joins[0] + 0.5

    def test_a_drifting_slice_is_measured_about_its_own_mean(self):
        v = self.grid(half_width=60.0)
        _, info = pic_thomson.extend_vdf_tail(self.sampled(1e5, v, drift=8.0), v)
        assert info["tail_over_core_width"] == pytest.approx(1.0, abs=0.05)

    def test_an_empty_slice_is_untouched(self):
        v = self.grid()
        f = np.zeros((1, v.size, 1))
        extended, info = pic_thomson.extend_vdf_tail(f, v)
        np.testing.assert_array_equal(extended, f)
        assert info["join_thermal_speeds"] is None

    def test_an_analytic_distribution_is_left_alone(self):
        """
        Nothing was counted, so there is no particle quantum and no ragged edge
        to join at. Extending it would be inventing a join.
        """
        v = self.grid()
        f = block(maxwellian(v, 1.0), n_time=1, n_x=1)
        extended, info = pic_thomson.extend_vdf_tail(f, v)
        np.testing.assert_array_equal(extended, f)
        assert info["counted_fraction"] == 0.0

    def test_the_smoothed_distribution_is_extended_but_the_raw_one_is_counted(self):
        """
        Smoothing mixes bins and destroys the particle quantum, so the counts
        have to come from the histogram as the code wrote it.
        """
        v = self.grid()
        raw = self.sampled(1e5, v)
        smoothed = pic_thomson.smooth_vdf(raw, window=8, iterations=2)
        _, with_counts = pic_thomson.extend_vdf_tail(smoothed, v, counts=raw)
        _, without = pic_thomson.extend_vdf_tail(smoothed, v)
        # A boxcar spreads single particles over its width, so the smallest
        # positive value is no longer one particle and the join lands somewhere
        # else entirely.
        assert with_counts["counted_fraction"] == 1.0
        assert without["join_thermal_speeds"] != pytest.approx(
            with_counts["join_thermal_speeds"], rel=0.05
        )

    def test_mismatched_counts_are_refused(self):
        v = self.grid()
        with pytest.raises(ValueError, match="same histogram"):
            pic_thomson.extend_vdf_tail(
                self.sampled(1e4, v), v, counts=np.zeros((1, 10, 1))
            )

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [({"min_counts": 0}, "min_counts"), ({"fit_from": -1.0}, "fit_from")],
    )
    def test_rejects_impossible_settings(self, kwargs, match):
        v = self.grid()
        with pytest.raises(ValueError, match=match):
            pic_thomson.extend_vdf_tail(self.sampled(1e4, v), v, **kwargs)

    def test_condition_phase_space_uses_it_by_default(self):
        v = self.grid()
        phase_space = pic_thomson.from_arrays(
            f=self.sampled(1e5, v),
            v=v,
            x=np.zeros(1),
            t=np.zeros(1),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(phase_space)
        conditioning = conditioned.meta["conditioning"]
        assert conditioning["tail"]["model"] == "maxwellian"
        assert conditioning["taper_threshold"] is None
        # No floor: it would put a flat pedestal at ~12 thermal speeds, which is
        # exactly where the satellite reads at alpha ~ 12.
        assert conditioning["floor"] == 0.0
        assert conditioned.f.min() >= 0.0

    def test_selecting_the_taper_instead_still_works(self):
        v = self.grid()
        phase_space = pic_thomson.from_arrays(
            f=self.sampled(1e5, v),
            v=v,
            x=np.zeros(1),
            t=np.zeros(1),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(
            phase_space, tail_model=None, max_taper_bins=20, pedestal_warning=None
        )
        conditioning = conditioned.meta["conditioning"]
        assert conditioning["tail"] is None
        assert conditioning["taper_threshold"] == 0.005
        assert conditioning["floor"] == pic_thomson.DEFAULT_FLOOR

    def test_an_unknown_tail_model_is_refused(self):
        with pytest.raises(ValueError, match="tail_model must be"):
            pic_thomson.condition_phase_space(simple_phase_space(), tail_model="kappa")


class TestAutomaticNotch:
    r"""
    A stray-light mask fixed in wavelength does not stay over a feature the flow
    is Doppler-shifting: at :math:`2\times10^6` m/s the shift is
    :math:`\lambda^2 k u / 2\pi c \approx 3.7` nm, most of a 4 nm notch, and the
    bin immediately outside holds the skirt of a feature orders of magnitude
    above the satellites. ``epw_notches="auto"`` sizes it from the central
    feature as measured in the IAW window, every timestep.
    """

    def run(self, **kwargs):
        pytest.importorskip("numba")
        return run_driver(iaw_wavelengths=IAW_WINDOW, **kwargs)

    def test_it_records_what_it_masked(self):
        spectra = self.run(epw_notches="auto")
        applied = np.asarray(spectra.meta["epw_notch_applied"], dtype=float)
        assert applied.shape == (spectra.n_time, 2)
        assert np.all(applied[:, 1] > applied[:, 0])
        # And it lands on the probe, which is where the central feature is for
        # a plasma with no bulk flow.
        centres = applied.mean(axis=1)
        np.testing.assert_allclose(centres, PROBE_WAVELENGTH.to_value(u.nm), atol=1.0)

    def test_the_masked_bins_are_zero(self):
        spectra = self.run(epw_notches="auto")
        applied = np.asarray(spectra.meta["epw_notch_applied"], dtype=float)
        grid = spectra.epw_wavelengths * 1e9
        for step in range(spectra.n_time):
            lo, hi = applied[step]
            inside = (grid >= lo) & (grid <= hi)
            if inside.any():
                assert np.all(spectra.epw[step][inside] == 0.0)

    def test_a_fixed_notch_is_still_honoured(self):
        spectra = self.run(epw_notches=[528, 536] * u.nm)
        grid = spectra.epw_wavelengths * 1e9
        inside = (grid >= 528) & (grid <= 536)
        assert np.all(spectra.epw[0][inside] == 0.0)
        assert np.all(np.isnan(np.asarray(spectra.meta["epw_notch_applied"])))

    def test_auto_needs_a_window_to_measure_the_feature_in(self):
        with pytest.raises(ValueError, match="measures in the IAW window"):
            run_driver(epw_notches="auto")

    def test_an_unknown_keyword_is_refused(self):
        with pytest.raises(ValueError, match="epw_notches must be"):
            self.run(epw_notches="widest")


class TestContainingInterval:
    """The measurement behind the automatic notch."""

    def test_it_finds_a_gaussian(self):
        x = np.linspace(-10.0, 10.0, 4001)
        y = np.exp(-(x**2) / 2)
        lo, hi = pic_thomson._containing_interval(y, x, 0.9973)  # +-3 sigma
        assert lo == pytest.approx(-3.0, abs=0.05)
        assert hi == pytest.approx(3.0, abs=0.05)

    def test_it_follows_a_shifted_feature(self):
        x = np.linspace(-10.0, 10.0, 4001)
        lo, hi = pic_thomson._containing_interval(
            np.exp(-((x - 4.0) ** 2) / 2), x, 0.9973
        )
        assert (lo + hi) / 2 == pytest.approx(4.0, abs=0.05)

    def test_an_empty_spectrum_has_no_interval(self):
        x = np.linspace(-10.0, 10.0, 101)
        assert pic_thomson._containing_interval(np.zeros_like(x), x, 0.999) is None


class TestSatelliteOffset:
    r"""
    Where Bohm--Gross puts the satellite, which is what stops the automatic
    stray-light mask from eating it.
    """

    def test_it_matches_the_plasma_frequency_shift_at_large_alpha(self):
        r"""
        As :math:`\alpha \to \infty` the thermal correction vanishes and the
        satellite sits at the plasma-frequency shift.
        """
        density = 1e25
        omega_pe = pic_thomson._plasma_frequency(density)
        expected = (532e-9) ** 2 * omega_pe / (2 * np.pi * const.c.si.value) * 1e9
        offset = pic_thomson._satellite_offset(1e4, density, 532 * u.nm)
        assert offset == pytest.approx(expected, rel=1e-3)

    def test_the_thermal_correction_pushes_it_out(self):
        near = pic_thomson._satellite_offset(np.sqrt(2.0), 1e25, 532 * u.nm)
        far = pic_thomson._satellite_offset(1e4, 1e25, 532 * u.nm)
        # sqrt(1 + 6/alpha^2) is exactly 2 at alpha = sqrt(2).
        assert near / far == pytest.approx(2.0, rel=1e-6)

    @pytest.mark.parametrize(
        ("alpha", "density"), [(np.nan, 1e25), (0.0, 1e25), (2.0, 0.0)]
    )
    def test_no_resonance_no_offset(self, alpha, density):
        assert np.isnan(pic_thomson._satellite_offset(alpha, density, 532 * u.nm))

    def test_the_mask_is_capped_by_it(self):
        """
        Where the plasma is only weakly collective the central feature and the
        electron feature merge, and a containment-sized mask swallows 30 nm of
        the window -- the satellites with it.
        """
        pytest.importorskip("numba")
        spectra = run_driver(iaw_wavelengths=IAW_WINDOW, epw_notches="auto")
        applied = np.asarray(spectra.meta["epw_notch_applied"], dtype=float)
        widths = applied[:, 1] - applied[:, 0]
        offsets = np.array(
            [
                pic_thomson._satellite_offset(a, n, PROBE_WAVELENGTH)
                for a, n in zip(
                    spectra.alpha_iaw, spectra.electron_density, strict=True
                )
            ]
        )
        good = np.isfinite(widths) & np.isfinite(offsets)
        assert good.any()
        # Half-width never past the default fraction of the satellite offset.
        assert np.all(widths[good] / 2 <= 0.2 * offsets[good] + 1e-9)


class TestStalePlotfiles:
    """
    WarpX renames a plotfile it is about to overwrite to ``<name>.old.<pid>``
    rather than deleting it, so a run that was killed and relaunched, or
    restarted, leaves stale copies in ``diags/``. They sort immediately after
    the plotfile they duplicate, so a plain ``<prefix>*`` glob returns them as
    extra timesteps carrying another run's data.
    """

    def test_old_directories_are_not_timesteps(self, tmp_path):
        diags = make_warpx_plotfiles(tmp_path, n_frames=3)
        (diags / "diag1000001.old.2859663").mkdir()
        found = pic_thomson._warpx_plotfiles(diags, "diag1", None)
        assert [p.name for p in found] == [
            "diag1000000",
            "diag1000001",
            "diag1000002",
        ]

    def test_other_diagnostics_sharing_the_prefix_are_not_taken(self, tmp_path):
        """``diag1`` must not swallow ``diag1_fields``."""
        diags = make_warpx_plotfiles(tmp_path, n_frames=2)
        (diags / "diag1_fields000000").mkdir()
        found = pic_thomson._warpx_plotfiles(diags, "diag1", None)
        assert [p.name for p in found] == ["diag1000000", "diag1000001"]

    def test_an_empty_directory_still_reports_clearly(self, tmp_path):
        diags = tmp_path / "diags"
        diags.mkdir()
        (diags / "diag1000000.old.1").mkdir()
        with pytest.raises(FileNotFoundError, match=r"diag1<step> plotfiles"):
            pic_thomson._warpx_plotfiles(diags, "diag1", None)

    def test_the_density_cross_check_skips_them_too(self, tmp_path):
        diags = make_warpx_plotfiles(tmp_path, n_frames=2, prefix="diag_fields")
        (diags / "diag_fields000001.old.7").mkdir()
        found = pic_thomson._warpx_plotfile_directories(diags, "diag_fields")
        assert [p.name for p in found] == ["diag_fields000000", "diag_fields000001"]


class TestAlphaScan:
    r"""
    `TestMaxwellianConsistency` checks the pipeline against
    `~plasmapy.diagnostics.thomson.spectral_density` at one operating point.
    A synthetic diagnostic is used across regimes, so the agreement is pinned
    over a range of :math:`\alpha` here, and with metrics that mean something
    for a narrow feature.

    Pointwise ratios do not: above :math:`\alpha \sim 3` the satellites are
    narrow, and half a nanometre of offset between two otherwise correct
    resonances reads as a factor of a hundred while the power under them agrees
    to a few percent. Integrated band power and peak position are what the
    measurement is, so those are what is checked.
    """

    PROBE = 532 * u.nm
    PROBE_VEC = np.array([1.0, 0.0, 0.0])
    SCATTER_VEC = np.array([0.0, 1.0, 0.0])
    T_E = 100.0
    T_I = 50.0
    SPLIT = 5.0  # nm about the probe dividing the centre from the wings

    @classmethod
    def spectra(cls, n_e):
        """The same plasma through the VDF model and the analytic one."""
        pytest.importorskip("numba")
        sigma_e = sigma_of(cls.T_E, const.m_e)
        sigma_i = sigma_of(cls.T_I, const.m_p)
        v_e = np.linspace(-12 * sigma_e, 12 * sigma_e, 8001)
        v_i = np.linspace(-12 * sigma_i, 12 * sigma_i, 8001)

        def phase_space(v, sigma, label):
            return pic_thomson.condition_phase_space(
                pic_thomson.from_arrays(
                    f=maxwellian(v, sigma)[np.newaxis, :, np.newaxis],
                    v=v,
                    x=np.zeros(1),
                    t=np.zeros(1),
                    label=label,
                )
            )

        electrons = phase_space(v_e, sigma_e, "e-")
        ions = phase_space(v_i, sigma_i, "p+")
        wavelengths = np.linspace(380, 760, 4000) * u.nm

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*numba_scipy.*")
            _, vdf = thomson.arbitrary_forwardmodel(
                wavelengths=wavelengths,
                probe_wavelength=cls.PROBE,
                e_velocity_axes=[electrons.v] * u.m / u.s,
                i_velocity_axes=[ions.v] * u.m / u.s,
                efn=[electrons.f[0, :, 0]] * u.s / u.m,
                ifn=[ions.f[0, :, 0]] * u.s / u.m,
                n=n_e * u.m**-3,
                ion_species=["p+"],
                probe_vec=cls.PROBE_VEC,
                scatter_vec=cls.SCATTER_VEC,
            )
        alpha, analytic = thomson.spectral_density(
            wavelengths,
            cls.PROBE,
            n_e * u.m**-3,
            T_e=cls.T_E * u.eV,
            T_i=np.array([cls.T_I]) * u.eV,
            ions=["p+"],
            probe_vec=cls.PROBE_VEC,
            scatter_vec=cls.SCATTER_VEC,
        )
        grid = wavelengths.to_value(u.nm)

        def unit(spectrum):
            spectrum = as_array(spectrum)
            return spectrum / np.trapezoid(spectrum, grid)

        return float(np.mean(np.asarray(alpha))), grid, unit(vdf), unit(analytic)

    @pytest.mark.parametrize(
        ("n_e", "alpha", "power_tolerance", "position_tolerance"),
        [
            (1.4e23, 0.31, 0.01, 0.15),
            (1.5e24, 1.01, 0.01, 0.15),
            (6.2e24, 2.05, 0.03, 0.35),
            (2.5e25, 4.12, 0.06, 1.20),
        ],
    )
    def test_band_power_and_satellite_position(
        self, n_e, alpha, power_tolerance, position_tolerance
    ):
        got_alpha, grid, vdf, analytic = self.spectra(n_e)
        assert got_alpha == pytest.approx(alpha, abs=0.05)

        blue = grid < self.PROBE.to_value(u.nm) - self.SPLIT
        centre = np.abs(grid - self.PROBE.to_value(u.nm)) <= self.SPLIT
        for name, band in (("blue wing", blue), ("centre", centre)):
            a = np.trapezoid(vdf[band], grid[band])
            b = np.trapezoid(analytic[band], grid[band])
            assert a == pytest.approx(b, rel=power_tolerance), (
                f"{name} power at alpha = {alpha}: {a:.4f} against {b:.4f}"
            )

        peak_vdf = grid[blue][np.argmax(vdf[blue])]
        peak_analytic = grid[blue][np.argmax(analytic[blue])]
        assert abs(peak_vdf - peak_analytic) < position_tolerance

    def test_the_electron_feature_all_but_vanishes_by_alpha_eight(self):
        r"""
        Not a tolerance but a fact worth pinning: by :math:`\alpha \approx 8`
        the satellites carry under a thousandth of the scattered power, so the
        electron channel stops being a measurement of anything whatever the
        pipeline does with it.
        """
        _, grid, vdf, analytic = self.spectra(1.0e26)
        blue = grid < self.PROBE.to_value(u.nm) - self.SPLIT
        assert np.trapezoid(analytic[blue], grid[blue]) < 1e-3
        assert np.trapezoid(vdf[blue], grid[blue]) < 1e-3


class TestPlotfileOrdering:
    r"""
    WarpX pads the step number in a plotfile name to six digits but does not
    truncate it, so a run past step 999999 writes seven-digit names that do not
    sort lexically against the six-digit ones. ``diag11002384`` (step 1002384)
    lands before ``diag1111376`` (step 111376), and the frames come back
    interleaved with a time axis that runs backwards in places.
    """

    def test_seven_digit_steps_sort_after_six(self, tmp_path):
        diags = tmp_path / "diags"
        steps = [0, 55688, 111376, 1002384, 1058072, 2784400]
        for step in steps:
            (diags / f"diag1{step:06d}").mkdir(parents=True)
        found = pic_thomson._warpx_plotfiles(diags, "diag1", None)
        assert [p.name for p in found] == [f"diag1{step:06d}" for step in steps]

    def test_a_lexical_sort_would_have_got_it_wrong(self, tmp_path):
        """Guard the guard: the fixture has to actually exercise the bug."""
        diags = tmp_path / "diags"
        names = ["diag1000000", "diag1111376", "diag11002384"]
        for name in names:
            (diags / name).mkdir(parents=True)
        assert sorted(names) != names
        found = pic_thomson._warpx_plotfiles(diags, "diag1", None)
        assert [p.name for p in found] == names

    def test_timesteps_index_the_step_ordered_list(self, tmp_path):
        diags = tmp_path / "diags"
        for step in (0, 111376, 1002384):
            (diags / f"diag1{step:06d}").mkdir(parents=True)
        found = pic_thomson._warpx_plotfiles(diags, "diag1", [1])
        assert [p.name for p in found] == ["diag1111376"]


class TestRescaleVDFDrift:
    r"""
    A run with kinetic electrons at the real :math:`m_e` and a reduced ion mass
    has two velocity scales in one array: an electron thermal speed that is
    already physical, and a bulk flow shared with the ions that is
    :math:`\sqrt{R}` too fast. `rescale_velocity_axis` cannot fix one without
    ruining the other. Translating each slice fixes the drift and leaves every
    central moment -- the temperature included -- exactly alone.
    """

    R = 18.3615
    T_EV = 100.0

    @classmethod
    def drifting(cls, drift, n_v=4001, span=12.0):
        sigma = sigma_of(cls.T_EV, const.m_e)
        v = np.linspace(drift - span * sigma, drift + span * sigma, n_v)
        return v, maxwellian(v, sigma, drift), sigma

    @staticmethod
    def moments(f, v):
        n = np.trapezoid(f, v)
        mean = np.trapezoid(f * v, v) / n
        return mean, float(np.sqrt(np.trapezoid(f * (v - mean) ** 2, v) / n))

    def test_the_drift_lands_on_target(self):
        scale = np.sqrt(self.R)
        v, f, _ = self.drifting(3e6)
        shifted, _ = pic_thomson.rescale_vdf_drift(
            f[np.newaxis, :, np.newaxis], v, scale
        )
        mean, _ = self.moments(shifted[0, :, 0], v)
        assert mean == pytest.approx(3e6 / scale, rel=1e-4)

    def test_the_temperature_is_untouched(self):
        v, f, _ = self.drifting(3e6)
        _, before = self.moments(f, v)
        shifted, _ = pic_thomson.rescale_vdf_drift(
            f[np.newaxis, :, np.newaxis], v, np.sqrt(self.R)
        )
        _, after = self.moments(shifted[0, :, 0], v)
        assert after == pytest.approx(before, rel=1e-4)

    def test_a_stationary_distribution_is_unchanged(self):
        v, f, _ = self.drifting(0.0)
        shifted, info = pic_thomson.rescale_vdf_drift(
            f[np.newaxis, :, np.newaxis], v, np.sqrt(self.R)
        )
        np.testing.assert_allclose(shifted[0, :, 0], f, atol=1e-30)
        assert info["max_shift_m_per_s"] == 0.0

    def test_each_slice_moves_by_its_own_drift(self):
        """
        The point of doing it per slice: a shock has a stationary upstream and a
        fast-moving front in the same array.
        """
        sigma = sigma_of(self.T_EV, const.m_e)
        v = np.linspace(-20 * sigma, 20 * sigma, 4001)
        f = np.stack(
            [maxwellian(v, sigma, drift) for drift in (0.0, 2e6, 5e6)], axis=-1
        )[np.newaxis, :, :]
        scale = np.sqrt(self.R)
        shifted, _ = pic_thomson.rescale_vdf_drift(f, v, scale)
        for cell, drift in enumerate((0.0, 2e6, 5e6)):
            mean, width = self.moments(shifted[0, :, cell], v)
            assert mean == pytest.approx(drift / scale, abs=0.01 * sigma)
            assert width == pytest.approx(sigma, rel=1e-3)

    def test_it_warns_when_the_shift_runs_off_the_grid(self):
        # A grid sized tightly around a fast-drifting population has nowhere to
        # put the distribution once it is moved back towards zero.
        sigma = sigma_of(self.T_EV, const.m_e)
        drift = 40 * sigma
        v = np.linspace(drift - 3 * sigma, drift + 3 * sigma, 2001)
        f = maxwellian(v, sigma, drift)[np.newaxis, :, np.newaxis]
        with pytest.warns(RuntimeWarning, match="off the end of its velocity grid"):
            pic_thomson.rescale_vdf_drift(f, v, np.sqrt(self.R))

    def test_an_empty_slice_is_returned_unchanged(self):
        v = np.linspace(-1e7, 1e7, 501)
        f = np.zeros((1, v.size, 1))
        shifted, info = pic_thomson.rescale_vdf_drift(f, v, 2.0)
        np.testing.assert_array_equal(shifted, f)
        assert info["max_shift_m_per_s"] == 0.0

    def test_a_non_positive_scale_is_refused(self):
        v, f, _ = self.drifting(1e6)
        with pytest.raises(ValueError, match="scale must be positive"):
            pic_thomson.rescale_vdf_drift(f[np.newaxis, :, np.newaxis], v, 0.0)

    def test_condition_phase_space_takes_it(self):
        v, f, sigma = self.drifting(3e6)
        phase_space = pic_thomson.from_arrays(
            f=f[np.newaxis, :, np.newaxis],
            v=v,
            x=np.zeros(1),
            t=np.zeros(1),
            label="e-",
        )
        conditioned = pic_thomson.condition_phase_space(
            phase_space,
            drift_scale_factor=self.R,
            tail_model=None,
            taper_threshold=None,
        )
        mean, width = self.moments(conditioned.f[0, :, 0], conditioned.v)
        assert mean == pytest.approx(3e6 / np.sqrt(self.R), rel=1e-3)
        assert width == pytest.approx(sigma, rel=1e-3)
        assert conditioned.meta["conditioning"]["drift"][
            "drift_scale"
        ] == pytest.approx(np.sqrt(self.R))

    def test_the_two_scalings_are_mutually_exclusive(self):
        v, f, _ = self.drifting(1e6)
        phase_space = pic_thomson.from_arrays(
            f=f[np.newaxis, :, np.newaxis],
            v=v,
            x=np.zeros(1),
            t=np.zeros(1),
            label="e-",
        )
        with pytest.raises(ValueError, match="not both"):
            pic_thomson.condition_phase_space(
                phase_space, velocity_scale_factor=self.R, drift_scale_factor=self.R
            )

    def test_the_driver_leaves_alpha_alone(self):
        r"""
        The whole point: scaling the electron velocity axis drags
        :math:`\alpha` by :math:`\sqrt{R}`, and scaling only the drift does not.
        """
        pytest.importorskip("numba")
        electrons = species_phase_space("e-", 100, const.m_e, drift=2e6)
        ions = species_phase_space("p+", 50, const.m_p, drift=2e6)
        plain = run_driver(electrons=electrons, ions=[ions])
        drifted = run_driver(
            electrons=electrons, ions=[ions], electron_drift_scale_factor=self.R
        )
        np.testing.assert_allclose(drifted.alpha_epw, plain.alpha_epw, rtol=1e-3)

        with warnings.catch_warnings():
            # Compressing the axis by sqrt(R) squeezes the distribution into a
            # fraction of its bins, and the forward model divides by the zeros
            # that leaves. That is the treatment being argued against, and the
            # numerical complaint is part of the argument.
            warnings.filterwarnings("ignore", message=".*encountered in divide.*")
            scaled = run_driver(
                electrons=electrons, ions=[ions], velocity_scale_factor=self.R
            )
        assert np.nanmedian(scaled.alpha_epw) > 3 * np.nanmedian(plain.alpha_epw)


class TestNonuniformVelocityAxis:
    """
    Readers that bin proper velocity hand back a non-uniform v axis, and the
    forward model differentiates with fixed-spacing stencils. Conditioning has
    to leave every species on a uniform axis, whether or not it is rescaled.
    """

    @staticmethod
    def phase_space():
        proper = np.linspace(-1.0, 1.0, 801)
        v = proper * const.c.si.value / np.sqrt(1 + proper**2)
        sigma = 2e7
        f = np.exp(-((v - 3e6) ** 2) / (2 * sigma**2))[None, :, None]
        return pic_thomson.from_arrays(
            f=f,
            v=v,
            x=np.array([0.0]),
            t=np.array([0.0]),
            label="e-",
            meta={"reference_density": 1.0},
        )

    @pytest.mark.parametrize(
        "settings",
        [{}, {"drift_scale_factor": 18.36}, {"velocity_scale_factor": 18.36}],
    )
    def test_conditioned_axis_is_uniform(self, settings) -> None:
        conditioned = pic_thomson.condition_phase_space(
            self.phase_space(), tail_model=None, taper_threshold=None, **settings
        )
        steps = np.diff(conditioned.v)
        np.testing.assert_allclose(steps, steps[0], rtol=1e-6)

    def test_resampling_keeps_the_moments(self) -> None:
        raw = self.phase_space()
        conditioned = pic_thomson.condition_phase_space(
            raw, tail_model=None, taper_threshold=None
        )
        assert conditioned.meta["conditioning"]["resampled_to_uniform"] > raw.v.size

        def moments(ps):
            g = ps.f[0, :, 0] / np.trapezoid(ps.f[0, :, 0], ps.v)
            mean = np.trapezoid(g * ps.v, ps.v)
            return mean, np.sqrt(np.trapezoid(g * (ps.v - mean) ** 2, ps.v))

        # Linear interpolation of a distribution sampled at the raw core
        # spacing dv adds about dv^2 / 12 to its variance: 1e-4 of the width
        # here, 2e-4 of the temperature.
        np.testing.assert_allclose(moments(conditioned), moments(raw), rtol=3e-4)
        np.testing.assert_allclose(
            np.trapezoid(conditioned.f[0, :, 0], conditioned.v), 1.0, rtol=1e-12
        )


class TestVacuumThreshold:
    """
    A reader that already produces SI records ``reference_density = 1`` m^-3,
    so a threshold relative to it was 0.01 m^-3 and never fired: frames with a
    handful of macroparticles at the probe got a spectrum instead of NaN.
    """

    @staticmethod
    def species(densities, label, *, is_electron, sigma):
        v = np.linspace(-6 * sigma, 6 * sigma, 401)
        shape = np.exp(-(v**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
        f = np.asarray(densities)[:, None, None] * shape[None, :, None]
        return pic_thomson.from_arrays(
            f=f,
            v=v,
            x=np.array([0.0]),
            t=np.arange(len(densities)) * 1e-12,
            label=label,
            is_electron=is_electron,
            meta={"reference_density": 1.0},
        )

    def run(self, **kwargs):
        densities = [1e25, 1e25, 1e20]  # the last frame is all but empty
        electrons = self.species(densities, "e-", is_electron=True, sigma=4e6)
        ions = self.species(densities, "p+", is_electron=False, sigma=7e4)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pic_thomson.spectra_from_phase_spaces(
                electrons,
                ions,
                position=0.0,
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(500, 564, 200) * u.nm,
                iaw_wavelengths=np.linspace(531, 533, 200) * u.nm,
                electron_conditioning={"skip": True},
                ion_conditioning={"skip": True},
                **kwargs,
            )

    def test_near_empty_frame_is_vacuum_for_si_input(self) -> None:
        spectrogram = self.run()
        assert np.isfinite(spectrogram.epw[:2]).all()
        assert np.isnan(spectrogram.epw[2]).all()
        assert spectrogram.meta["vacuum_density"] == pytest.approx(1e23)

    def test_explicit_vacuum_density_wins(self) -> None:
        spectrogram = self.run(vacuum_density=1e19 * u.m**-3)
        assert np.isfinite(spectrogram.epw).all()
        assert spectrogram.meta["vacuum_density"] == pytest.approx(1e19)


class TestElectronMixtureAlpha:
    """
    The forward model reports the unweighted mean over electron populations of
    sqrt(2) w_pe / (k sigma_p). The plasma's alpha comes from summing inverse
    squared Debye lengths: alpha^2 = sum_p f_p alpha_p^2.
    """

    sigma_cold = 4e6

    def run(self, fractions, widths):
        v = np.linspace(-1.2e8, 1.2e8, 4001)
        total = 1e25
        species = []
        for fraction, sigma in zip(fractions, widths, strict=True):
            shape = np.exp(-(v**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma)
            species.append(
                pic_thomson.from_arrays(
                    f=(fraction * total * shape)[None, :, None],
                    v=v,
                    x=np.array([0.0]),
                    t=np.array([0.0]),
                    label="e-",
                    is_electron=True,
                    meta={"reference_density": 1.0},
                )
            )
        sigma_i = 7e4
        vi = np.linspace(-6 * sigma_i, 6 * sigma_i, 801)
        ion_f = np.exp(-(vi**2) / (2 * sigma_i**2)) / (np.sqrt(2 * np.pi) * sigma_i)
        ions = pic_thomson.from_arrays(
            f=(total * ion_f)[None, :, None],
            v=vi,
            x=np.array([0.0]),
            t=np.array([0.0]),
            label="p+",
            meta={"reference_density": 1.0},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pic_thomson.spectra_from_phase_spaces(
                species,
                ions,
                position=0.0,
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(480, 584, 300) * u.nm,
                iaw_wavelengths=np.linspace(531, 533, 100) * u.nm,
                electron_conditioning={"skip": True},
                ion_conditioning={"skip": True},
            )

    def test_mixture_alpha_is_density_weighted(self) -> None:
        cold = self.run([1.0], [self.sigma_cold])
        hot = 5 * self.sigma_cold
        fractions = (0.8, 0.2)
        mixture = self.run(fractions, [self.sigma_cold, hot])
        expected = np.sqrt(fractions[0] + fractions[1] * (self.sigma_cold / hot) ** 2)
        ratio = mixture.alpha_epw[0] / cold.alpha_epw[0]
        assert ratio == pytest.approx(expected, rel=1e-3)

    def test_each_population_reads_the_same_phase_velocity(self) -> None:
        hot = 5 * self.sigma_cold
        mixture = self.run((0.8, 0.2), [self.sigma_cold, hot])
        cold_reach, hot_reach = mixture.epw_tail_required[:, 0]
        # The resonance is one phase velocity; in each population's widths
        # it is that velocity over sigma_p.
        assert cold_reach / hot_reach == pytest.approx(hot / self.sigma_cold, rel=1e-3)


class TestFramesAreIndependent:
    """
    A frame's conditioning must not depend on which other frames were read.
    One smoothing window sized against the narrowest slice of the whole block
    made a frame's spectrum change by up to 2.7 dex with how a run was chunked.
    """

    @staticmethod
    def block():
        rng = np.random.default_rng(3)
        sigma = sigma_of(40.0, const.m_e)
        v = np.linspace(-40 * sigma, 40 * sigma, 1024)
        frames = [
            maxwellian(v, sigma),
            maxwellian(v, 6 * sigma),
            maxwellian(v, 2 * sigma),
        ]
        f = np.stack([rng.poisson(2e4 * frame / frame.max()) for frame in frames])
        return pic_thomson.from_arrays(
            f=f[:, :, np.newaxis].astype(float),
            v=v,
            x=np.array([0.0]),
            t=np.arange(3) * 1e-12,
            label="e-",
        )

    @pytest.mark.parametrize(
        "settings",
        [
            {"smoothing_width": 0.25, "smoothing_iterations": 2},
            {
                "smoothing_width": 0.25,
                "smoothing_iterations": 2,
                "tail_model": None,
                "max_taper_width": 0.5,
            },
        ],
    )
    def test_conditioning_a_block_equals_each_frame_alone(self, settings) -> None:
        whole = self.block()
        together = pic_thomson.condition_phase_space(whole, **settings)
        for frame in range(whole.f.shape[0]):
            alone = pic_thomson.condition_phase_space(
                replace(
                    whole, f=whole.f[frame : frame + 1], t=whole.t[frame : frame + 1]
                ),
                **settings,
            )
            np.testing.assert_allclose(
                together.f[frame], alone.f[0], rtol=1e-12, atol=1e-30
            )


class TestParticleQuantum:
    """
    extend_vdf_tail turns a raw histogram into particle counts. The smallest
    positive value is one particle only when every value is a whole multiple
    of it; a fractional, shaped deposit (OSIRIS) needs the shot noise instead.
    """

    @staticmethod
    def histogram(n_particles, *, shaped, n_columns=8, weight=2.5):
        rng = np.random.default_rng(7)
        n_v = 400
        out = np.zeros((n_columns, n_v, 1))
        for column in range(n_columns):
            position = rng.normal(n_v / 2, 18.0, n_particles)
            if shaped:
                # Linear (cloud-in-cell) deposit: each particle split between
                # its two nearest bins.
                low = np.floor(position).astype(int)
                share = position - low
                np.add.at(out[column, :, 0], low, weight * (1 - share))
                np.add.at(out[column, :, 0], low + 1, weight * share)
            else:
                np.add.at(out[column, :, 0], np.rint(position).astype(int), weight)
        v = np.linspace(-1e7, 1e7, n_v)
        return out, v

    def counts(self, f, v, **kwargs):
        _, info = pic_thomson.extend_vdf_tail(f, v, **kwargs)
        return info

    def test_whole_multiples_use_the_smallest_value(self) -> None:
        f, v = self.histogram(20000, shaped=False)
        info = self.counts(f, v)
        assert info["quantum_source"]["quantised"] == f.shape[0]
        assert info["quantum_source"]["shot_noise"] == 0

    def test_a_shaped_deposit_is_counted_from_its_noise(self) -> None:
        n_particles = 20000
        f, v = self.histogram(n_particles, shaped=True)
        info = self.counts(f, v)
        assert info["quantum_source"]["shot_noise"] == f.shape[0]
        raw = f[:, :, 0].T
        smallest = np.where(raw > 0, raw, np.inf).min(axis=0)
        quantum, _, _ = pic_thomson._particle_quantum(
            raw, smallest, np.ones(raw.shape[1], dtype=bool), None
        )
        inferred = raw.sum(axis=0) / quantum
        # The smallest value of a shaped deposit is a sliver of a particle and
        # overstates the count 5-35 times here; the noise estimate stays
        # within about a factor of two.
        assert np.all(raw.sum(axis=0) / smallest > 3 * n_particles)
        assert np.all((inferred > 0.5 * n_particles) & (inferred < 2.5 * n_particles))

    def test_a_known_quantum_overrides_the_estimate(self) -> None:
        f, v = self.histogram(20000, shaped=True)
        info = self.counts(f, v, quantum=2.5)
        assert info["quantum_source"] == {"given": f.shape[0]}


class TestCacheSignature:
    """A cache key must not depend on how the running numpy spells a scalar."""

    settings = {
        "species": "e",
        "mass": np.float64(9.1093837015e-31),
        "scatter_direction": [np.float64(0.0), np.float64(0.0), np.float64(1.0)],
        "n_velocity_bins": 512,
        "timesteps": None,
    }

    def test_numpy_scalars_and_plain_floats_agree(self) -> None:
        plain = {
            **self.settings,
            "mass": 9.1093837015e-31,
            "scatter_direction": [0.0, 0.0, 1.0],
        }
        assert pic_thomson._cache_signature(self.settings) == (
            pic_thomson._cache_signature(plain)
        )

    def test_an_old_numpy2_repr_key_still_matches(self) -> None:
        # What numpy 2 wrote, whichever numpy is running now.
        legacy = (
            "[('mass', np.float64(9.1093837015e-31)), ('n_velocity_bins', 512), "
            "('scatter_direction', [np.float64(0.0), np.float64(0.0), "
            "np.float64(1.0)]), ('species', 'e'), ('timesteps', None)]"
        )
        assert pic_thomson._cache_matches(legacy, self.settings)

    def test_different_settings_do_not_match(self) -> None:
        other = {**self.settings, "n_velocity_bins": 256}
        assert not pic_thomson._cache_matches(
            pic_thomson._cache_signature(self.settings), other
        )


class TestAbsoluteIntensity:
    """
    The forward model scales every spectrum to unit area. A time series needs
    the opposite: brightness that means the same thing in every frame.
    """

    def run(self, densities, **kwargs):
        electrons = TestVacuumThreshold.species(
            densities, "e-", is_electron=True, sigma=4e6
        )
        ions = TestVacuumThreshold.species(
            densities, "p+", is_electron=False, sigma=7e4
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return pic_thomson.spectra_from_phase_spaces(
                electrons,
                ions,
                position=0.0,
                probe_wavelength=532 * u.nm,
                epw_wavelengths=np.linspace(500, 564, 400) * u.nm,
                iaw_wavelengths=np.linspace(531, 533, 200) * u.nm,
                electron_conditioning={"skip": True},
                ion_conditioning={"skip": True},
                vacuum_density=1.0 * u.m**-3,
                **kwargs,
            )

    def test_each_frame_on_its_own_has_unit_area(self) -> None:
        spectra = self.run([1e25, 3e25], normalise_each_frame=True)
        x = spectra.epw_wavelengths * 1e9
        np.testing.assert_allclose(np.trapezoid(spectra.epw, x, axis=1), 1.0)

    def test_default_is_density_times_the_structure_factor(self) -> None:
        densities = [1e25, 3e25]
        absolute = self.run(densities)
        shape = self.run(densities, normalise_each_frame=True)
        x = absolute.epw_wavelengths * 1e9
        area = np.trapezoid(absolute.epw, x, axis=1)
        # Same shape frame by frame...
        np.testing.assert_allclose(absolute.epw, shape.epw * area[:, None], rtol=1e-9)
        # ...and a brightness history: three times the density scatters more.
        assert area[1] / area[0] > 1.5

    def test_automatic_notch_is_one_filter_on_the_probe(self) -> None:
        spectra = self.run([1e25, 3e25, 2e25], epw_notches="auto")
        applied = np.asarray(spectra.meta["epw_notch_applied"], dtype=float)
        assert np.all(applied == applied[0])
        assert applied[0].mean() == pytest.approx(532.0, abs=1e-9)
        x = spectra.epw_wavelengths * 1e9
        blocked = (x >= applied[0, 0]) & (x <= applied[0, 1])
        assert blocked.any()
        assert np.all(spectra.epw[:, blocked] == 0)
