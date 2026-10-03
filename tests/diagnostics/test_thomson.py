"""
Tests for Thomson scattering analysis functions
"""

import copy

import astropy.constants as const
import astropy.units as u
import numpy as np
import pytest
from lmfit import Parameter, Parameters

from plasmapy.diagnostics import thomson
from plasmapy.particles import Particle, particle_mass
from plasmapy.particles.particle_collections import ParticleList


def example_instr_func(w):
    """
    Example instrument function for use in testing
    """
    sigma = 0.5 * u.nm
    arg = (w / sigma).to(u.dimensionless_unscaled).value
    inst = np.exp(-(arg**2))
    inst *= 1 / np.sum(inst)
    return inst


def example_invalid_instr_func_bad_type(w):
    """
    Example instrument function for use in testing

    This instrument function is invalid because it does not return a plain
    np.ndarray.
    """
    sigma = 0.5 * u.nm
    arg = (w / sigma).to(u.dimensionless_unscaled)
    inst = np.exp(-(arg**2))
    inst *= 1 / np.sum(inst)
    return inst * u.m


def example_invalid_instr_func_bad_shape(w):
    """
    Example instrument function for use in testing

    This instrument function is invalid because it returns an array of a
    different shape than the provided wavelength array
    """
    sigma = 0.5 * u.nm
    arg = (w / sigma).to(u.dimensionless_unscaled).value
    inst = np.exp(-(arg**2))
    inst *= 1 / np.sum(inst)
    return inst[2:]


# A list of invalid instrument functions
invalid_instr_func_list = [
    (example_invalid_instr_func_bad_type),
    (example_invalid_instr_func_bad_shape),
]


def width_at_value(x, y, val):
    """
    Calculates the width of a curve at a given value.
    """
    above = np.where(y > val, x, np.nan)
    return np.abs(np.nanmax(above) - np.nanmin(above))


def spectral_density_args_kwargs(kwargs):
    """
    Separate positional arguments and keyword arguments
    for the spectral_density function from a dictionary of both that is
    easy to use in parametrized tests.
    """
    # Pull out the non-keyword arguments
    args = (
        kwargs["wavelengths"],
        kwargs["probe_wavelength"],
        kwargs["n"],
    )

    del kwargs["wavelengths"]
    del kwargs["probe_wavelength"]
    del kwargs["n"]

    return args, kwargs


def args_to_lite_args(kwargs):  # noqa: C901
    """
    Converts a dict of args for the spectral density function and converts
    them to input for the lite function.

    Used to facilitate testing the two functions against each other.
    """
    keys = list(kwargs.keys())

    if "wavelengths" in keys:
        kwargs["wavelengths"] = kwargs["wavelengths"].to(u.m).value
    if "probe_wavelength" in keys:
        kwargs["probe_wavelength"] = kwargs["probe_wavelength"].to(u.m).value
    if "n" in keys:
        kwargs["n"] = kwargs["n"].to(u.m**-3).value
    if "T_e" in keys:
        kwargs["T_e"] = (kwargs["T_e"] / const.k_B).to(u.K).value
    if "T_i" in keys:
        kwargs["T_i"] = (kwargs["T_i"] / const.k_B).to(u.K).value
    if "electron_vel" in keys:
        kwargs["electron_vel"] = kwargs["electron_vel"].to(u.m / u.s).value
    if "ion_vel" in keys:
        kwargs["ion_vel"] = kwargs["ion_vel"].to(u.m / u.s).value

    if kwargs["T_e"].size == 1:
        kwargs["T_e"] = np.array(
            [
                kwargs["T_e"],
            ],
        )
    if kwargs["T_i"].size == 1:
        kwargs["T_i"] = np.array(
            [
                kwargs["T_i"],
            ],
        )

    if not isinstance(kwargs["ions"], list):
        kwargs["ions"] = [
            kwargs["ions"],
        ]

    ion_z = np.zeros(len(kwargs["ions"]))
    ion_mass = np.zeros(len(kwargs["ions"]))
    for i, particle in enumerate(kwargs["ions"]):
        if not isinstance(particle, Particle):
            particle = Particle(particle)  # noqa: PLW2901
        ion_z[i] = particle.charge_number
        ion_mass[i] = particle_mass(particle).to(u.kg).value
    kwargs["ion_z"] = ion_z
    kwargs["ion_mass"] = ion_mass
    del kwargs["ions"]

    return kwargs


@pytest.fixture
def single_species_collective_args():
    """
    Standard args

    Includes both kwargs and args: separated by the function

    spectral_density_args_kwargs

    """
    return {
        "wavelengths": np.arange(520, 545, 0.01) * u.nm,
        "probe_wavelength": 532 * u.nm,
        "n": 5e17 * u.cm**-3,
        "T_e": 10 * u.eV,
        "T_i": 10 * u.eV,
        "efract": np.array([1.0]),
        "ifract": np.array([1.0]),
        "ions": "C-12 5+",
        "electron_vel": np.array([[0, 0, 0]]) * u.km / u.s,
        "ion_vel": np.array([[0, 0, 0]]) * u.km / u.s,
        "probe_vec": np.array([1, 0, 0]),
        "scatter_vec": np.array([0, 1, 0]),
    }


@pytest.fixture
def single_species_collective_spectrum(single_species_collective_args):
    """
    Generates an example Thomson scattering spectrum in the collective regime
    """
    wavelengths = single_species_collective_args["wavelengths"]

    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)

    alpha, Skw = thomson.spectral_density(*args, **kwargs)

    return (alpha, wavelengths, Skw)


@pytest.mark.slow
def test_single_species_collective_spectrum(single_species_collective_spectrum) -> None:
    """
    Compares the generated spectrum to previously determined values
    """
    alpha, wavelength, Skw = single_species_collective_spectrum

    # Check that alpha is correct
    assert np.isclose(alpha, 1.801, atol=0.01), (
        f"Collective case alpha returns {alpha} instead of expected 1.801"
    )

    i_width = width_at_value(wavelength.value, Skw.value, 2e-13)
    e_width = width_at_value(wavelength.value, Skw.value, 0.2e-13)

    # Check that the widths of the ion and electron features match expectations
    assert np.isclose(i_width, 0.1599, 1e-3), (
        f"Collective case ion feature width is {i_width}instead of expected 0.1599"
    )

    assert np.isclose(e_width, 17.7899, 1e-3), (
        "Collective case electron "
        f"feature width is {e_width} "
        "instead of expected 17.7899"
    )


@pytest.mark.parametrize(
    ("notch", "notch_num"),
    [
        # one notch
        (np.array([531, 533]) * u.nm, 1),
        # two notches
        (np.array([np.array([520, 525]), np.array([530, 540])]) * u.nm, 2),
    ],
)
def test_notched_spectrum(notch, notch_num, single_species_collective_args) -> None:
    """
    Compares notched and unnotched spectra
    """
    # make a copy of the input args
    args_fixture_copy = copy.copy(single_species_collective_args)
    wavelengths = single_species_collective_args["wavelengths"]

    # Compute spectrum with no notch included
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)
    alpha_unnotched, Skw_unnotched = thomson.spectral_density(*args, **kwargs)

    # Compute same spectrum with notch
    args, kwargs = spectral_density_args_kwargs(args_fixture_copy)

    kwargs["notch"] = notch
    alpha_notched, Skw_notched = thomson.spectral_density(*args, **kwargs)

    # Check that notch does not affect alpha
    assert np.isclose(alpha_notched, alpha_unnotched)

    if notch_num == 1:
        # Record wavelength array indices corresponding to notch
        x0 = np.argwhere(wavelengths > notch[0])[0][0]
        x1 = np.argwhere(wavelengths > notch[1])[0][0]
        # Check that regions outside the notch are the same for both Skws
        assert np.allclose(Skw_notched[:x0], Skw_unnotched[:x0])
        assert np.allclose(Skw_notched[x1:], Skw_unnotched[x1:])

        # Check that region inside the notch is 0 for notched Skw
        assert np.allclose(Skw_notched[x0:x1], np.zeros(x1 - x0))
    elif notch_num == 2:
        x0 = np.argwhere(wavelengths > notch[0, 0])[0][0]
        x1 = np.argwhere(wavelengths > notch[0, 1])[0][0]
        x2 = np.argwhere(wavelengths > notch[1, 0])[0][0]
        x3 = np.argwhere(wavelengths > notch[1, 1])[0][0]

        # Check that regions outside the notches are the same for both Skws
        assert np.allclose(Skw_notched[:x0], Skw_unnotched[:x0])
        assert np.allclose(Skw_notched[x1:x2], Skw_unnotched[x1:x2])
        assert np.allclose(Skw_notched[x3:], Skw_unnotched[x3:])

        # Check that region inside the notches is 0 for notched Skw
        assert np.allclose(Skw_notched[x0:x1], np.zeros(x1 - x0))
        assert np.allclose(Skw_notched[x2:x3], np.zeros(x3 - x2))


@pytest.mark.parametrize(
    ("notch"),
    [
        (np.array([533, 531]) * u.nm),  # Elements not in monotonically increasing order
        (np.array([530, 531, 533]) * u.nm),  # Not exactly 2 elements
    ],
)
def test_notch_errors(notch, single_species_collective_args) -> None:
    """
    Check notch input validation
    """
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)
    kwargs["notch"] = notch
    with pytest.raises(ValueError):
        # prefix unused unpacked values with _
        _alpha, _Skw = thomson.spectral_density(*args, **kwargs)


@pytest.mark.slow
def test_spectral_density_minimal_arguments(single_species_collective_args) -> None:
    """
    Check that spectral density runs with minimal arguments
    """
    single_species_collective_args["wavelengths"]
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)

    # Delete the arguments that have default values
    optional_keys = [
        "efract",
        "ifract",
        "ions",
        "electron_vel",
        "ion_vel",
        "probe_vec",
        "scatter_vec",
        "instr_func",
        "notch",
    ]
    for key in optional_keys:
        if key in kwargs:
            del kwargs[key]

    _alpha, _Skw = thomson.spectral_density(*args, **kwargs)


def test_single_species_collective_lite(single_species_collective_args) -> None:
    # Make a copy of the input args
    args_fixture_copy = copy.copy(single_species_collective_args)
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)
    alpha1, Skw1 = thomson.spectral_density(*args, **kwargs)

    lite_kwargs = args_to_lite_args(args_fixture_copy)
    args, kwargs = spectral_density_args_kwargs(lite_kwargs)
    alpha2, Skw2 = thomson.spectral_density.lite(*args, **kwargs)

    assert np.isclose(alpha1, alpha2)

    assert np.allclose(Skw1.to(u.s / u.rad).value, Skw2)


def test_spectral_density_lite_minimal_arguments(
    single_species_collective_args,
) -> None:
    lite_kwargs = args_to_lite_args(single_species_collective_args)
    args, kwargs = spectral_density_args_kwargs(lite_kwargs)

    # Delete the arguments that have default values
    optional_keys = [
        "instr_func_arr",
    ]
    for key in optional_keys:
        if key in kwargs:
            del kwargs[key]

    _alpha, _Skw = thomson.spectral_density.lite(*args, **kwargs)


@pytest.fixture
def multiple_species_collective_args():
    """
    Standard args

    Includes both kwargs and args: separated by the function

    spectral_density_args_kwargs

    """
    kwargs = {
        "wavelengths": np.arange(520, 545, 0.01) * u.nm,
        "probe_wavelength": 532 * u.nm,
        "n": 5e17 * u.cm**-3,
        "T_e": np.array([10, 50]) * u.eV,
    }
    kwargs["T_i"] = np.array([25, 25]) * u.eV
    kwargs["ions"] = [Particle("p+"), Particle("C-12 5+")]
    kwargs["probe_vec"] = np.array([1, 0, 0])
    kwargs["scatter_vec"] = np.array([0, 1, 0])
    kwargs["efract"] = np.array([0.4, 0.6])
    kwargs["ifract"] = np.array([0.7, 0.3])
    kwargs["electron_vel"] = np.array([[0, 0, 0], [100, 0, 0]]) * u.km / u.s
    kwargs["ion_vel"] = np.array([[-100, 0, 0], [0, 100, 0]]) * u.km / u.s

    return kwargs


def test_efract_sum_error(single_species_collective_args) -> None:
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)
    kwargs["efract"] = np.array([2.0])  # Sum is not 1
    with pytest.raises(ValueError):
        _alpha, _Skw = thomson.spectral_density(*args, **kwargs)


def test_ifract_sum_error(single_species_collective_args) -> None:
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)
    kwargs["ifract"] = np.array([0.5, 1.2])  # Sum is not 1
    with pytest.raises(ValueError):
        _alpha, _Skw = thomson.spectral_density(*args, **kwargs)


@pytest.fixture
def multiple_species_collective_spectrum(multiple_species_collective_args):
    """
    Generates an example Thomson scattering spectrum for multiple ion species
    that also have drift velocities. Parameters are set to be in the
    collective regime where ion species are important.
    """
    wavelengths = multiple_species_collective_args["wavelengths"]

    args, kwargs = spectral_density_args_kwargs(multiple_species_collective_args)

    alpha, Skw = thomson.spectral_density(*args, **kwargs)

    return (alpha, wavelengths, Skw)


def test_multiple_species_collective_spectrum(
    multiple_species_collective_spectrum,
) -> None:
    """
    Compares the generated spectrum to previously determined values
    """
    _alpha, wavelength, Skw = multiple_species_collective_spectrum

    # Compute the width and max of the spectrum, and the wavelength
    # of the max (sensitive to ion vel)
    max_skw = np.nanmax(Skw.value)
    width = width_at_value(wavelength.value, Skw.value, 2e-12)

    max_wavelength = wavelength.value[np.argmax(Skw.value)]

    # Check width
    assert np.isclose(width, 0.1499, 1e-2), (
        f"Multiple ion species case spectrum width is {width} instead of expected 0.17"
    )

    # Check max value
    assert np.isclose(max_skw, 6e-12, 1e-11), (
        f"Multiple ion species case spectrum max is {max_skw} instead of expected 6e-12"
    )

    # Check max peak location
    assert np.isclose(max_wavelength, 532, 1e-2), (
        "Multiple ion species case spectrum peak wavelength is "
        f"{max_wavelength} instead of expected 532"
    )


@pytest.fixture
def single_species_non_collective_args():
    """
    Standard args

    Includes both kwargs and args: separated by the function

    spectral_density_args_kwargs

    """
    kwargs = {
        "wavelengths": np.arange(500, 570, 0.01) * u.nm,
        "probe_wavelength": 532 * u.nm,
        "n": 5e15 * u.cm**-3,
        "T_e": 100 * u.eV,
    }
    kwargs["T_i"] = np.array([10]) * u.eV
    kwargs["efract"] = np.array([1.0])
    kwargs["ifract"] = np.array([1.0])
    kwargs["ions"] = ["H+"]
    kwargs["electron_vel"] = np.array([[0, 0, 0]]) * u.km / u.s
    kwargs["ion_vel"] = np.array([[0, 0, 0]]) * u.km / u.s
    kwargs["probe_vec"] = np.array([1, 0, 0])
    kwargs["scatter_vec"] = np.array([0, 1, 0])

    return kwargs


@pytest.fixture
def single_species_non_collective_spectrum(single_species_non_collective_args):
    """
    Generates an example Thomson scattering spectrum in the non-collective
    regime
    """
    wavelengths = single_species_non_collective_args["wavelengths"]

    args, kwargs = spectral_density_args_kwargs(single_species_non_collective_args)

    alpha, Skw = thomson.spectral_density(*args, **kwargs)

    return (alpha, wavelengths, Skw)


@pytest.mark.slow
def test_single_species_non_collective_spectrum(
    single_species_non_collective_spectrum,
) -> None:
    """
    Compares the generated spectrum to previously determined values
    """
    alpha, wavelength, Skw = single_species_non_collective_spectrum

    # Check that alpha is correct
    assert np.isclose(alpha, 0.05707, atol=0.01), (
        f"Non-collective case alpha returns {alpha} instead of expected 0.05707"
    )

    e_width = width_at_value(wavelength.value, Skw.value, 0.2e-13)

    # Check that the widths of the electron feature matches expectations
    assert np.isclose(e_width, 22.6699, 1e-3), (
        "Non-collective case electron "
        f"feature width is {e_width} "
        "instead of expected 22.6699"
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "msg"),
    [
        # Ion species provided but empty
        (
            {"ions": []},
            ValueError,
            "At least one ion species needs to be defined.",
        ),
        # Inconsistent number of ion parameters
        (
            {
                "ifract": [0.5, 0.5],
                "T_i": 5 * u.eV,
            },
            ValueError,
            "Inconsistent number of ion species in ifract",
        ),
        (
            {"ifract": [0.5, 0.5], "ion_vel": np.array([[100, 0, 0]]) * u.km / u.s},
            ValueError,
            "Inconsistent number of ion species in ifract",
        ),
        # Inconsistent number of electron parameters
        (
            {
                "efract": [0.5, 0.5],
                "T_e": np.array(
                    [
                        5,
                    ],
                )
                * u.eV,
            },
            ValueError,
            "number of electron populations",
        ),
        (
            {
                "efract": [0.5, 0.5],
                "electron_vel": np.array([[100, 0, 0]]) * u.km / u.s,
            },
            ValueError,
            "number of electron populations",
        ),
        # List of strings
        (
            {
                "ions": [
                    "p+",
                ],
            },
            None,
            None,
        ),
        # List of Particles
        (
            {
                "ions": [
                    Particle("p+"),
                ],
            },
            None,
            None,
        ),
        # Particle list
        ({"ions": ParticleList(["p+"])}, None, None),
        # ValueError when an ion is negative
        (
            {"ions": ParticleList(["p-"])},
            ValueError,
            "All ions must be positively charged.",
        ),
        # ValueError when an ion charge information is not provided
        (
            {"ions": ParticleList(["He"])},
            ValueError,
            "All ions must be positively charged.",
        ),
    ],
)
def test_spectral_density_input_errors(
    kwargs,
    error,
    msg,
    single_species_collective_args,
) -> None:
    """
    Validate errors with invalid argument and keyword arguments in
    spectral_density
    """
    args = single_species_collective_args

    # Replace any modified keys
    for key, value in kwargs.items():
        args[key] = value

    # Separate the arguments into args and kwargs for spectral_density
    args, kwargs = spectral_density_args_kwargs(args)

    if error is None:
        _alpha, _Skw = thomson.spectral_density(*args, **kwargs)
    else:
        with pytest.raises(error) as excinfo:
            _alpha, _Skw = thomson.spectral_density(*args, **kwargs)

            # If msg is not None, check that this string is a subset of the
            # error message
            if msg is not None:
                assert msg in str(excinfo.value)


@pytest.mark.slow
def test_split_populations() -> None:
    """
    Make sure that splitting a single population of ions or electrons
    into two identical halves returns the same result.
    """
    wavelengths = np.arange(520, 545, 0.01) * u.nm
    probe_wavelength = 532 * u.nm
    n = 5e17 * u.cm**-3
    probe_vec = np.array([1, 0, 0])
    scatter_vec = np.array([0, 1, 0])

    # Combined
    T_e = np.array([10]) * u.eV
    T_i = np.array([10]) * u.eV
    ions = ["H+"]
    ifract = np.array([1.0])
    efract = np.array([1.0])

    _alpha, Skw0 = thomson.spectral_density(
        wavelengths,
        probe_wavelength,
        n,
        T_e=T_e,
        T_i=T_i,
        ifract=ifract,
        efract=efract,
        ions=ions,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
    )

    # Split e and i populations into two parts
    # this should not change the results since the parts are identical
    T_e = np.array([10, 10]) * u.eV
    T_i = np.array([10, 10]) * u.eV
    ions = ["H+", "H+"]
    ifract = np.array([0.2, 0.8])
    efract = np.array([0.8, 0.2])

    _alpha, Skw1 = thomson.spectral_density(
        wavelengths,
        probe_wavelength,
        n,
        T_e=T_e,
        T_i=T_i,
        ifract=ifract,
        efract=efract,
        ions=ions,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
    )

    # Calculate the deviation between the two spectra
    # (any differences should be in the noise)
    deviation = (Skw0 - Skw1) / Skw0 * 100

    assert np.all(deviation < 1e-6), "Failed split populations test"


def test_thomson_with_instrument_function(single_species_collective_args) -> None:
    """
    Generates an example Thomson scattering spectrum with an instrument
    function applied
    """
    wavelengths = single_species_collective_args["wavelengths"]

    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)

    _alpha, Skw_with = thomson.spectral_density(
        *args,
        **kwargs,
        instr_func=example_instr_func,
    )

    _alpha, Skw_without = thomson.spectral_density(*args, **kwargs)

    # Assert that the instrument function has made the IAW peak wider
    w1 = width_at_value(wavelengths.value, Skw_with.value, 2e-13)
    w2 = width_at_value(wavelengths.value, Skw_without.value, 2e-13)
    assert w1 > w2


@pytest.mark.parametrize("instr_func", invalid_instr_func_list)
def test_thomson_with_invalid_instrument_function(
    instr_func,
    single_species_collective_args,
) -> None:
    """
    Verifies that an exception is raised if the provided instrument function
    is invalid.
    """
    args, kwargs = spectral_density_args_kwargs(single_species_collective_args)

    kwargs["instr_func"] = instr_func

    with pytest.raises(ValueError):
        _alpha, _Skw_with = thomson.spectral_density(*args, **kwargs)


def test_param_to_array_fcns() -> None:
    """
    Tests a few low-level routines used to convert lmfit scalar parameters
    into array input for `spectral_density` based on a naming convention
    """
    params = Parameters()

    # Create two groups of test variables, one of scalars and one of vectors
    prefix = "T_e"
    for i in range(3):
        params.add(f"{prefix}_{i}", value=2)

    prefix = "ion_vel"
    for i in range(2):
        for j in ("x", "y", "z"):
            params.add(f"{prefix}_{j}_{i}", value=2)

    arr = thomson._params_to_array(params, "T_e", vector=False)
    assert arr.shape == (3,)
    assert np.mean(arr) == 2

    arr = thomson._params_to_array(params, "ion_vel", vector=True)
    assert arr.shape == (2, 3)
    assert np.mean(arr) == 2


def run_fit(  # noqa: C901
    wavelengths,
    params,
    settings,
    noise_amp: float = 0.05,
    fit_method: str = "differential_evolution",
    fit_kws: dict[str, object] | None = None,
    max_iter=None,
    require_redchi: float = 1.0,
    # If false, don't perform the actual fit but instead just create the Model
    run_fit: bool = True,  # noqa: FBT001, FBT002
) -> None:
    """
    Take a Parameters object, generate some synthetic data near it,
    perturb the initial values, then try a fit.

    Note: `ions` is passed in settings here (instead of parameters)
    because we need the full ions list to make the spectrum. They are then
    moved to parameters later in this function.
    """
    if fit_kws is None:
        fit_kws = {}

    wavelengths = (wavelengths * u.m).to(u.nm)

    skeys = list(settings.keys())
    pkeys = list(params.keys())

    # Fill any missing required parameters
    if "efract_0" not in pkeys:
        params.add("efract_0", value=1.0, vary=False)

    if "ifract_0" not in pkeys:
        params.add("ifract_0", value=1.0, vary=False)

    if "electron_speed" not in pkeys:
        params.add("electron_speed_0", value=0.0, vary=False)

    if "ion_speed" not in pkeys:
        params.add("ion_speed_0", value=0.0, vary=False)

    # LOAD FROM PARAMS
    n = params["n"]
    T_e = thomson._params_to_array(params, "T_e")
    T_i = thomson._params_to_array(params, "T_i")
    efract = thomson._params_to_array(params, "efract")
    ifract = thomson._params_to_array(params, "ifract")
    electron_speed = thomson._params_to_array(params, "electron_speed")
    ion_speed = thomson._params_to_array(params, "ion_speed")

    if "instr_func" not in skeys:
        settings["instr_func"] = None

    if "notch" not in skeys:
        settings["notch"] = None

    # LOAD FROM SETTINGS
    ions = settings["ions"]
    probe_vec = settings["probe_vec"]
    scatter_vec = settings["scatter_vec"]
    probe_wavelength = settings["probe_wavelength"]
    instr_func = settings["instr_func"]
    notch = settings["notch"]

    electron_vdir = settings.get("electron_vdir", np.ones([len(T_e), 3]))
    ion_vdir = settings.get("ion_vdir", np.ones([len(T_i), 3]))

    electron_vel = electron_speed[:, np.newaxis] * electron_vdir
    ion_vel = ion_speed[:, np.newaxis] * ion_vdir

    # Create the synthetic data
    _alpha, Skw = thomson.spectral_density(
        wavelengths,
        probe_wavelength * u.m,
        n * u.m**-3,
        T_e=T_e * u.eV,
        T_i=T_i * u.eV,
        ifract=ifract,
        efract=efract,
        ions=ions,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
        electron_vel=electron_vel * u.m / u.s,
        ion_vel=ion_vel * u.m / u.s,
        instr_func=instr_func,
        notch=notch,
    )

    data = Skw

    data *= 1 + np.random.normal(  # noqa: NPY002
        loc=0,
        scale=noise_amp,
        size=wavelengths.size,
    )
    data *= 1 / np.nanmax(data)

    # Randomly choose the starting values of the parameters within the
    # search space (to make the algorithm do some work!)
    for p in list(params.keys()):
        if params[p].vary:
            params[p].value = float(
                np.random.uniform(  # noqa: NPY002
                    low=params[p].min,
                    high=params[p].max,
                    size=1,
                )[0],
            )

    # Make the model, then perform the fit
    model = thomson.spectral_density_model(
        wavelengths.to(u.m).value,
        settings,
        params,
    )

    if run_fit:
        result = model.fit(
            data,
            params,
            wavelengths=wavelengths.to(u.m).value,
            method=fit_method,
            max_nfev=max_iter,
            fit_kws=fit_kws,
        )

        # Assert that the fit reduced chi2 is under the requirement specified
        assert result.redchi < require_redchi


def spectral_density_model_settings_params(kwargs):
    """
    Separate a settings dict and a parameters object from a provided
    dictionary.

    This is useful for testing the spectral_density_model function.

    The dictionary needs to hold a Parameter object for Parameters.

    """
    if "wavelengths" in kwargs:
        wavelengths = kwargs["wavelengths"]
    else:
        raise ValueError("Kwargs must include 'wavelengths'")

    settings = {}
    setting_names = [
        "probe_wavelength",
        "probe_vec",
        "scatter_vec",
        "ions",
        "electron_vdir",
        "ion_vdir",
        "instr_func",
        "notch",
    ]

    params = Parameters()

    for k, v in kwargs.items():
        # If key is a setting, add the value to the settings
        if k == "wavelengths":
            pass

        elif k in setting_names:
            settings[k] = v

        # If v is a parameter, add to the params
        elif isinstance(v, Parameter):
            params.add(v)

        else:
            raise ValueError(f"Invalid key: {k}")

    return wavelengths, params, settings


@pytest.fixture
def epw_single_species_settings_params():
    """
    Standard input for the spectral_density_model function

    Includes both settings and params: separated by the function

    spectral_density_model_settings_params

    """
    probe_wavelength = 532 * u.nm
    scattering_angle = np.deg2rad(63)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])
    notch = np.array([531, 533]) * u.nm

    kwargs = {"probe_wavelength": probe_wavelength.to(u.m).value}
    kwargs["probe_vec"] = np.array([1, 0, 0])
    kwargs["scatter_vec"] = scatter_vec
    kwargs["notch"] = notch
    kwargs["ions"] = ["H+"]

    kwargs["n"] = Parameter(
        "n",
        value=2e17 * 1e6,
        vary=True,
        min=8e16 * 1e6,
        max=6e17 * 1e6,
    )
    kwargs["T_e_0"] = Parameter("T_e_0", value=10, vary=True, min=5, max=20)
    kwargs["T_i_0"] = Parameter("T_i_0", value=20, vary=False, min=5, max=70)

    w0 = probe_wavelength.value
    kwargs["wavelengths"] = (
        (np.linspace(w0 - 40, w0 + 40, num=512) * u.nm).to(u.m).value
    )

    return kwargs


@pytest.fixture
def epw_multi_species_settings_params():
    """
    Standard input for the spectral_density_model function

    Includes both settings and params: separated by the function

    spectral_density_model_settings_params

    """
    probe_wavelength = 532 * u.nm
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(63)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])
    notch = np.array([531, 533]) * u.nm

    kwargs = {"probe_wavelength": probe_wavelength.to(u.m).value}

    kwargs["probe_vec"] = probe_vec
    kwargs["scatter_vec"] = scatter_vec
    kwargs["notch"] = notch
    kwargs["ions"] = ["H+"]

    kwargs["n"] = Parameter(
        "n",
        value=2e17 * 1e6,
        vary=True,
        min=8e16 * 1e6,
        max=6e17 * 1e6,
    )
    kwargs["T_e_0"] = Parameter("T_e_0", value=10, vary=True, min=5, max=20)
    kwargs["T_e_1"] = Parameter("T_e_1", value=35, vary=True, min=5, max=20)
    kwargs["T_i_0"] = Parameter("T_i_0", value=20, vary=False, min=5, max=70)
    kwargs["efract_0"] = Parameter("efract_0", value=0.5, vary=False)
    kwargs["efract_1"] = Parameter("efract_1", value=0.5, vary=False)

    w0 = probe_wavelength.value
    kwargs["wavelengths"] = (
        (np.linspace(w0 - 40, w0 + 40, num=512) * u.nm).to(u.m).value
    )

    return kwargs


@pytest.fixture
def iaw_single_species_settings_params():
    """
    Standard input for the spectral_density_model function

    Includes both settings and params: separated by the function

    spectral_density_model_settings_params

    """
    probe_wavelength = 532 * u.nm
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(90)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

    kwargs = {
        "probe_wavelength": probe_wavelength.to(u.m).value,
        "probe_vec": probe_vec,
        "scatter_vec": scatter_vec,
        "ions": ["H+"],
        "ion_vdir": np.array([[1, 0, 0]]),
        "electron_vdir": np.array([[1, 0, 0]]),
        "n": Parameter("n", value=2e17 * 1e6, vary=False),
        "T_e_0": Parameter("T_e_0", value=10, vary=False, min=5, max=20),
        "T_i_0": Parameter("T_i_0", value=20, vary=True, min=5, max=70),
        "ifract_0": Parameter("ifract_0", value=1.0, vary=False),
        "ion_speed_0": Parameter("ion_speed_0", value=0, vary=False),
        "electron_speed_0": Parameter("electron_speed_0", value=0, vary=False),
    }

    w0 = probe_wavelength.value
    kwargs["wavelengths"] = (np.linspace(w0 - 5, w0 + 5, num=512) * u.nm).to(u.m).value

    return kwargs


@pytest.fixture
def iaw_multi_species_settings_params():
    """
    Standard input for the spectral_density_model function

    Includes both settings and params: separated by the function

    spectral_density_model_settings_params

    """
    probe_wavelength = 532 * u.nm
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(63)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

    kwargs = {
        "probe_wavelength": probe_wavelength.to(u.m).value,
        "probe_vec": probe_vec,
        "scatter_vec": scatter_vec,
        "ions": ["H+", "H+", "C-12 +4"],
        "ion_vdir": np.array([[0.5, 0.5, 0]]),
        "electron_vdir": np.array([[0, 0.2, 0.7]]),
        "n": Parameter("n", value=1e19 * 1e6, vary=False),
        "T_e_0": Parameter("T_e_0", value=500, vary=False, min=5, max=1000),
        "T_i_0": Parameter("T_i_0", value=200, vary=True, min=5, max=1000),
        "T_i_1": Parameter("T_i_1", value=500, vary=True, min=5, max=1000),
        "T_i_2": Parameter("T_i_2", value=400, vary=False, min=5, max=1000),
        "ifract_0": Parameter("ifract_0", value=0.4, vary=False, min=0.2, max=0.8),
        "ifract_1": Parameter("ifract_1", value=0.3, vary=False, min=0.2, max=0.8),
        "ifract_2": Parameter("ifract_2", value=0.3, vary=False, min=0.2, max=0.8),
        "ion_speed_0": Parameter("ion_speed_0", value=0, vary=False),
        "ion_speed_1": Parameter("ion_speed_1", value=1e5, vary=True, min=0, max=5e5),
        "ion_speed_2": Parameter("ion_speed_2", value=2e5, vary=False, min=0, max=5e5),
        "electron_speed_0": Parameter("electron_speed_0", value=0, vary=False),
    }

    w0 = probe_wavelength.value
    kwargs["wavelengths"] = (np.linspace(w0 - 5, w0 + 5, num=512) * u.nm).to(u.m).value

    return kwargs


@pytest.fixture
def noncollective_single_species_settings_params():
    """
    Standard input for the spectral_density_model function

    Includes both settings and params: separated by the function

    spectral_density_model_settings_params

    """
    probe_wavelength = 532 * u.nm
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(30)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])

    kwargs = {
        "probe_wavelength": probe_wavelength.to(u.m).value,
        "probe_vec": probe_vec,
        "scatter_vec": scatter_vec,
        "ions": ["H+"],
        "ion_vdir": np.array([[1, 0, 0]]),
        "electron_vdir": np.array([[1, 0, 0]]),
        "n": Parameter(
            "n",
            value=2e17 * 1e6,
            vary=True,
            min=8e16 * 1e6,
            max=6e17 * 1e6,
        ),
        "T_e_0": Parameter("T_e_0", value=10, vary=True, min=5, max=20),
        "T_i_0": Parameter("T_i_0", value=120, vary=False, min=5, max=70),
        "efract_0": Parameter("efract_0", value=1.0, vary=False),
        "electron_speed_0": Parameter("electron_speed_0", value=0, vary=False),
    }

    w0 = probe_wavelength.value
    kwargs["wavelengths"] = (
        (np.linspace(w0 - 60, w0 + 60, num=512) * u.nm).to(u.m).value
    )

    return kwargs


@pytest.mark.slow
def test_fit_epw_single_species(epw_single_species_settings_params) -> None:
    wavelengths, params, settings = spectral_density_model_settings_params(
        epw_single_species_settings_params,
    )

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_epw_multi_species(epw_multi_species_settings_params) -> None:
    wavelengths, params, settings = spectral_density_model_settings_params(
        epw_multi_species_settings_params,
    )

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_iaw_single_species(iaw_single_species_settings_params) -> None:
    wavelengths, params, settings = spectral_density_model_settings_params(
        iaw_single_species_settings_params,
    )

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_iaw_instr_func(iaw_single_species_settings_params) -> None:
    """
    Tests fitting with an instrument function
    """
    wavelengths, params, settings = spectral_density_model_settings_params(
        iaw_single_species_settings_params,
    )

    settings["instr_func"] = example_instr_func

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_ion_mu_and_z(iaw_single_species_settings_params) -> None:
    """
    Tests fitting with ion parameters explicitly set and allowed to vary
    """
    wavelengths, params, settings = spectral_density_model_settings_params(
        iaw_single_species_settings_params,
    )

    for i, ion in enumerate(settings["ions"]):
        _ion = Particle(ion)
        mass = _ion.mass.to(u.kg).value
        Z = _ion.charge_number
        params.add(f"ion_mu_{i!s}", value=mass, vary=True, min=0.5, max=10)
        params.add(f"ion_z_{i!s}", value=Z, vary=True, min=1, max=10)

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_iaw_multi_species(iaw_multi_species_settings_params) -> None:
    wavelengths, params, settings = spectral_density_model_settings_params(
        iaw_multi_species_settings_params,
    )

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_noncollective_single_species(
    noncollective_single_species_settings_params,
) -> None:
    wavelengths, params, settings = spectral_density_model_settings_params(
        noncollective_single_species_settings_params,
    )

    run_fit(wavelengths, params, settings)


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_fit_with_instr_func(epw_single_species_settings_params) -> None:
    """

    Check that fitting works with an instrument function.

    It specifically tests the case where a notch is being used in the filter,
    because this can cause a potential error with the instrument function.

    """
    wavelengths, params, settings = spectral_density_model_settings_params(
        epw_single_species_settings_params,
    )

    settings["instr_func"] = example_instr_func
    settings["notch"] = np.array([531, 533]) * 1e-9 * u.nm

    # Warns that data should not include any NaNs
    # This is taken care of in run_fit by deleting the notch region rather than
    # replacing it with np.nan
    with pytest.warns(UserWarning, match="If an instrument function is included,"):
        run_fit(
            wavelengths,
            params,
            settings,
            run_fit=False,
        )

    # Run the same fit using np.delete instead of np.nan values
    run_fit(wavelengths, params, settings)


@pytest.mark.parametrize("instr_func", invalid_instr_func_list)
def test_fit_with_invalid_instr_func(
    instr_func,
    iaw_single_species_settings_params,
) -> None:
    """
    Verifies that an exception is raised if the provided instrument function
    is invalid.
    """
    wavelengths, params, settings = spectral_density_model_settings_params(
        iaw_single_species_settings_params,
    )

    settings["instr_func"] = instr_func

    with pytest.raises(ValueError):
        run_fit(wavelengths, params, settings)


@pytest.mark.slow
def test_fit_with_minimal_parameters() -> None:
    # Create example data for fitting
    probe_wavelength = 532 * u.nm
    probe_vec = np.array([1, 0, 0])
    scattering_angle = np.deg2rad(90)
    scatter_vec = np.array([np.cos(scattering_angle), np.sin(scattering_angle), 0])
    w0 = probe_wavelength.value
    wavelengths = np.linspace(w0 - 5, w0 + 5, num=512) * u.nm

    ions = [Particle("H+")]
    n = 2e17 * u.cm**-3
    T_i = 20 * u.eV
    T_e = 10 * u.eV

    _alpha, Skw = thomson.spectral_density(
        wavelengths,
        probe_wavelength,
        n,
        T_e=T_e,
        T_i=T_i,
        ions=ions,
        probe_vec=probe_vec,
        scatter_vec=scatter_vec,
    )
    data = Skw.value

    data *= 1 + np.random.normal(  # noqa: NPY002
        loc=0,
        scale=0.1,
        size=wavelengths.size,
    )
    data *= 1 / np.nanmax(data)

    # Create settings and params using only the minimal parameters
    # intentionally leave out a few required values to check to make sure an
    # exception is raised

    settings = {"probe_vec": probe_vec, "scatter_vec": scatter_vec, "ions": ions}
    params = Parameters()

    params.add("T_e_0", value=T_e.value, vary=False, min=5, max=20)
    params.add("T_i_0", value=T_i.value, vary=True, min=5, max=70)
    params.add("ion_mu_0", value=1, vary=False)
    params.add("ion_z_0", value=ions[0].charge_number, vary=False)

    # Try creating model: will raise exception because required values
    # are missing in settings, eg. 'probe_wavelength'
    with pytest.raises(ValueError):
        model = thomson.spectral_density_model(wavelengths, settings, params)

    # Add back in the required values
    settings["probe_wavelength"] = probe_wavelength.to(u.m).value

    # Still raises an exception because T_e_0 is still missing
    with pytest.raises(ValueError):
        model = thomson.spectral_density_model(wavelengths, settings, params)

    params.add("n", value=n.to(u.m**-3).value, vary=False)

    # Make the model, then perform the fit
    model = thomson.spectral_density_model(wavelengths.to(u.m).value, settings, params)

    result = model.fit(  # noqa: F841
        data,
        params,
        wavelengths=wavelengths.to(u.m).value,
        method="differential_evolution",
        max_nfev=2000,
    )


@pytest.mark.parametrize(
    ("control", "error", "msg"),
    [
        # Required settings
        (
            {"probe_wavelength": None},
            ValueError,
            "not provided in settings, but is required",
        ),
        (
            {"scatter_vec": None},
            ValueError,
            "not provided in settings, but is required",
        ),
        ({"probe_vec": None}, ValueError, "not provided in settings, but is required"),
        (
            {"ions": None},
            ValueError,
            "not provided in settings, but is required",
        ),
        # Required parameters
        ({"n": None}, ValueError, "was not provided in parameters, but is required."),
        (
            {"T_e_0": None},
            ValueError,
            "was not provided in parameters, but is required.",
        ),
        # Two ion temps are required for this multi-ion example
        (
            {"T_i_0": None},
            ValueError,
            "was not provided in parameters, but is required.",
        ),
        (
            {"T_i_1": None},
            ValueError,
            "was not provided in parameters, but is required.",
        ),
        # If speed is not zero, vdir must be set
        (
            {
                "electron_speed_0": Parameter("electron_speed_0", 1e5),
                "electron_vdir": None,
            },
            ValueError,
            "electron_vdir must be set if electron_speeds",
        ),
        (
            {"ion_speed_0": Parameter("ion_speed_0", 1e5), "ion_vdir": None},
            ValueError,
            "ion_vdir must be set if ion_speeds",
        ),
    ],
)
def test_model_input_validation(
    control,
    error,
    msg,
    iaw_multi_species_settings_params,
) -> None:
    kwargs = iaw_multi_species_settings_params
    # We'll need to switch from print() to using logging library
    print(list(control.keys()))  # noqa: T201

    # Remove or replace values in kwargs
    for k, v in control.items():
        if v is None:
            del kwargs[k]
        else:
            kwargs[k] = v

    wavelengths, params, settings = spectral_density_model_settings_params(kwargs)

    if error is None:
        thomson.spectral_density_model(
            wavelengths,
            settings,
            params,
        )

    else:
        with pytest.raises(error) as excinfo:
            thomson.spectral_density_model(wavelengths, settings, params)

            # If msg is not None, check that this string is a subset of the
            # error message
            if msg is not None:
                print(excinfo.value)  # noqa: T201
                assert msg in str(excinfo.value)


class TestArbitraryChi:
    r"""
    `~plasmapy.diagnostics.thomson.arbitrary_chi` evaluates the principal value
    in closed form. Checked here against an independent brute-force quadrature,
    and against the large-:math:`\xi` asymptote a truncated integration window
    cannot reach.
    """

    u_axis = np.linspace(-6, 6, 513)
    unit = {"k": 1.0, "v_th": 1.0, "n": 1.0, "particle_m": 1.0, "particle_q": 1.0}

    @classmethod
    def coefficient(cls) -> float:
        """The prefactor arbitrary_chi applies to the integral, for `unit`."""
        wpl2 = 1.6022e-19**2 / (1.6605e-27 * 8.8541878e-12)
        return -wpl2 / np.sqrt(2)

    @staticmethod
    def brute_force(f, u_axis, x):
        """PV as int_0^S [g(x+s) - g(x-s)] / s ds on a step of h/2000."""
        f_prime = thomson.arbitrary_derivative(f, u_axis, 1)
        h = u_axis[1] - u_axis[0]
        span = u_axis[-1] - u_axis[0]
        out = []
        for point in x:
            reach = span + abs(point)
            s = np.linspace(0, reach, int(reach / (h / 2000)) + 1)[1:]
            g_plus = np.interp(point + s, u_axis, f_prime, left=0, right=0)
            g_minus = np.interp(point - s, u_axis, f_prime, left=0, right=0)
            y = (g_plus - g_minus) / s
            out.append(np.trapezoid(y, s) + s[0] * y[0])
        return np.array(out)

    @pytest.mark.parametrize("noisy", [False, True])
    def test_principal_value_matches_brute_force(self, noisy) -> None:
        f = np.exp(-(self.u_axis**2))
        if noisy:
            # A PIC-like histogram: shot noise, and empty bins at the edges.
            rng = np.random.default_rng(1)
            f = np.clip(f * (1 + 0.2 * rng.standard_normal(f.size)), 0, None)
            f[:20] = 0
            f[-20:] = 0
        x = np.array([-5.7, -2.1, -0.37, 0.0, 0.81, 1.9, 4.4])
        chi = thomson.arbitrary_chi(f, self.u_axis, xi=x, **self.unit)
        expected = self.brute_force(f, self.u_axis, x)
        np.testing.assert_allclose(
            chi.real / self.coefficient(), expected, rtol=1e-6, atol=1e-8
        )

    @pytest.mark.parametrize("xi", [20.0, 60.0, 200.0])
    def test_far_phase_velocity_sees_the_whole_distribution(self, xi) -> None:
        """
        PV of f'(u)/(u - xi) tends to (int f du) / xi^2 + 3 (int u^2 f du) / xi^4.

        The old quadrature integrated over xi plus or minus the axis span, so
        once |xi| exceeded the span -- every ion in an EPW window -- it missed
        part or all of the distribution, however many points it used.
        """
        f = np.exp(-(self.u_axis**2))
        chi = thomson.arbitrary_chi(f, self.u_axis, xi=np.array([xi]), **self.unit)
        norm = np.trapezoid(f, self.u_axis)
        second = np.trapezoid(self.u_axis**2 * f, self.u_axis)
        asymptote = norm / xi**2 + 3 * second / xi**4
        np.testing.assert_allclose(
            chi.real[0] / self.coefficient(), asymptote, rtol=1e-4
        )

    def test_odd_and_even_point_counts_agree(self) -> None:
        """The quadrature parameters are kept for compatibility and do nothing."""
        f = np.exp(-(self.u_axis**2))
        x = np.linspace(-3, 3, 41)
        a = thomson.arbitrary_chi(f, self.u_axis, xi=x, nPoints=1000, **self.unit)
        b = thomson.arbitrary_chi(f, self.u_axis, xi=x, nPoints=1001, **self.unit)
        np.testing.assert_array_equal(a, b)

    def test_derivative_refuses_a_nonuniform_axis(self) -> None:
        u_axis = np.sinh(np.linspace(-2, 2, 513))
        with pytest.raises(ValueError, match="uniformly spaced"):
            thomson.arbitrary_derivative(np.exp(-(u_axis**2)), u_axis, 1)


class TestArbitraryMatchesSpectralDensity:
    """
    On Maxwellian input the arbitrary-VDF forward model must reproduce
    `~plasmapy.diagnostics.thomson.spectral_density`, including for a mixture of
    ion species. Two faults used to break this: the ion term was weighted by
    ``Z * ifract`` instead of ``Z**2 * ifract / Zbar`` -- identical for one
    species, 24-39% off for a hydrogen-carbon mixture -- and the ion mass was
    passed in units of m_p to a function that converts with the amu, 0.72% light.
    """

    @staticmethod
    def spectra(ions, ifract):
        wavelengths = np.linspace(529, 535, 601) * u.nm
        n = 5e25 * u.m**-3
        T_e, T_i = 100.0, 50.0
        kB = const.e.si.value
        sigma_e = np.sqrt(T_e * kB / const.m_e.si.value)
        v_e = np.linspace(-3e7, 3e7, 4001)
        efn = [np.exp(-(v_e**2) / (2 * sigma_e**2)) / (np.sqrt(2 * np.pi) * sigma_e)]
        ifn, v_i = [], []
        for ion in ions:
            sigma = np.sqrt(T_i * kB / Particle(ion).mass.si.value)
            v = np.linspace(-12 * sigma, 12 * sigma, 4001)
            v_i.append(v)
            ifn.append(np.exp(-(v**2) / (2 * sigma**2)) / (np.sqrt(2 * np.pi) * sigma))
        geometry = {
            "probe_vec": np.array([1, 0, 0]),
            "scatter_vec": np.array([0, 1, 0]),
        }
        _, arbitrary = thomson.arbitrary_forwardmodel(
            wavelengths=wavelengths,
            probe_wavelength=532 * u.nm,
            e_velocity_axes=[v_e] * u.m / u.s,
            i_velocity_axes=v_i * u.m / u.s,
            efn=efn * u.s / u.m,
            ifn=ifn * u.s / u.m,
            n=n,
            efract=np.array([1.0]),
            ifract=np.array(ifract),
            ion_species=list(ions),
            **geometry,
        )
        _, reference = thomson.spectral_density(
            wavelengths,
            532 * u.nm,
            n,
            T_e=T_e * u.eV,
            T_i=np.full(len(ions), T_i) * u.eV,
            efract=np.array([1.0]),
            ifract=np.array(ifract),
            ions=list(ions),
            **geometry,
        )
        x = wavelengths.value
        arbitrary = np.asarray(arbitrary)
        reference = np.asarray(getattr(reference, "value", reference))
        return (
            arbitrary / np.trapezoid(arbitrary, x),
            reference / np.trapezoid(reference, x),
        )

    @pytest.mark.parametrize(
        ("ions", "ifract"),
        [(["p+"], [1.0]), (["p+", "C 6+"], [0.5, 0.5]), (["p+", "C 6+"], [0.9, 0.1])],
    )
    def test_ion_feature_matches(self, ions, ifract) -> None:
        arbitrary, reference = self.spectra(ions, ifract)
        assert np.max(np.abs(arbitrary - reference)) / reference.max() < 1e-3

    def test_caller_ion_list_is_left_alone(self) -> None:
        ions = ["p+", "C 6+"]
        self.spectra(ions, [0.5, 0.5])
        assert ions == ["p+", "C 6+"]


class TestAutodiffChi:
    """The torch path evaluates the same closed form as the numpy one."""

    @staticmethod
    def stencils(n):
        """The numpy derivative stencils, as matrices on a unit-spaced axis."""
        unit = np.arange(n, dtype=float)
        eye = np.eye(n)
        return tuple(
            np.stack(
                [
                    thomson.arbitrary_derivative(eye[:, j], unit, order)
                    for j in range(n)
                ],
                axis=1,
            )
            for order in (1, 2)
        )

    def test_matches_the_numpy_path_and_differentiates(self) -> None:
        torch = pytest.importorskip("torch")
        u_axis = np.linspace(-6, 6, 257)
        rng = np.random.default_rng(2)
        f = np.exp(-(u_axis**2)) * (1 + 0.1 * rng.standard_normal(u_axis.size))
        f[:15] = 0
        f[-15:] = 0
        xi = np.array([-40.0, -5.5, -1.3, 0.0, 0.7, 2.9, 25.0])
        args = {"k": 1.0, "v_th": 1.0, "n": 1.0, "particle_m": 1.0, "particle_q": 1.0}
        expected = thomson.arbitrary_chi(f, u_axis, xi=xi, **args)
        f_torch = torch.tensor(f, dtype=torch.float64, requires_grad=True)
        matrices = tuple(torch.tensor(m) for m in self.stencils(u_axis.size))
        chi = thomson.autodiff_chi(
            f_torch,
            matrices,
            torch.tensor(u_axis),
            xi=torch.tensor(xi),
            **args,
        )
        np.testing.assert_allclose(chi.detach().numpy().ravel(), expected, rtol=1e-9)
        chi.real.sum().backward()
        assert torch.isfinite(f_torch.grad).all()
