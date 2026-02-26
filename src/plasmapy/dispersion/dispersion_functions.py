"""
For calculating the plasma dispersion function :math:`Z(ζ)` and its
derivative :math:`Z′(ζ)`.
"""

__all__ = ["plasma_dispersion_func", "plasma_dispersion_func_deriv"]


import astropy.units as u
import numpy as np
from scipy.special import wofz as faddeeva_function


def plasma_dispersion_func(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the plasma dispersion function.

    The plasma dispersion function is defined as:

    .. math::
        Z(ζ) = π^{-0.5} \int_{-∞}^{+∞}
        \frac{e^{-x^2}}{x-ζ} dx

    where the argument is a complex number :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        The real or complex value to be provided as an argument to the
        plasma dispersion function.

    Returns
    -------
    |array_like| or |Quantity|
        The real or complex value of plasma dispersion function
        evaluated at ``zeta``.

    Raises
    ------
    ~astropy.units.UnitsError
        If ``zeta`` is a |Quantity| but is not dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_func_deriv`

    Notes
    -----
    In plasma wave theory, the plasma dispersion function appears
    frequently when the background medium has a Maxwellian
    distribution function.  The argument of this function then refers
    to the ratio of a wave's phase velocity to a thermal velocity.

    Examples
    --------
    >>> from plasmapy.dispersion import plasma_dispersion_func
    >>> plasma_dispersion_func(0)
    np.complex128(1.77245385...j)
    >>> plasma_dispersion_func(1 + 1j)
    np.complex128(-0.36905845...+0.54014504...j)
    >>> plasma_dispersion_func([0.3, 0.7 + 2.3j])
    array([-0.56526333+1.61990085j, -0.09995023+0.37685142j])
    """
    try:
        return 1j * np.sqrt(np.pi) * faddeeva_function(zeta)
    except u.UnitTypeError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_func should be a real or "
            "complex number or array, or a dimensionless Quantity."
        ) from wrong_type


def plasma_dispersion_func_deriv(
    zeta: complex | np.ndarray | u.Quantity[u.dimensionless_unscaled],
) -> complex | np.ndarray | u.Quantity[u.dimensionless_unscaled]:
    r"""
    Calculate the derivative of the plasma dispersion function.

    The derivative of the plasma dispersion function is:

    .. math::
        Z'(ζ) = π^{-1/2} \int_{-∞}^{+∞} \frac{e^{-x^2}}{(x-ζ)^2} dx

    where the argument :math:`ζ` is a complex number
    :cite:p:`fried:1961`.

    Parameters
    ----------
    zeta : |array_like| or |Quantity|
        Argument of plasma dispersion function.

    Returns
    -------
    complex, `~numpy.ndarray`, or |Quantity|
        First derivative of plasma dispersion function.

    Raises
    ------
    ~astropy.units.UnitsError
        If the argument is a `~astropy.units.Quantity` but is not
        dimensionless.

    See Also
    --------
    `~plasmapy.dispersion.dispersion_functions.plasma_dispersion_func`

    Examples
    --------
    >>> plasma_dispersion_func_deriv(0)
    np.complex128(-2+0j)
    >>> plasma_dispersion_func_deriv(1j)
    np.complex128(-0.48425568771737604+0j)
    >>> plasma_dispersion_func_deriv(-1.52 + 0.47j)
    np.complex128(0.165871331...+0.4458797880...j)
    """
    try:
        return -2 * (1 + zeta * plasma_dispersion_func(zeta))
    except u.UnitsError as wrong_units:
        raise u.UnitsError(
            "The argument to plasma_dispersion_func_deriv "
            "must be dimensionless if it is a Quantity."
        ) from wrong_units
    except TypeError as wrong_type:
        raise TypeError(
            "The argument to plasma_dispersion_func_deriv "
            "must be one of the following types: complex, float, "
            "int, ndarray, or a dimensionless Quantity."
        ) from wrong_type


# -----------------------------------------------------------------------------
# Compatibility helpers (formerly in plasmapy.dispersion.dispersionfunction)
# -----------------------------------------------------------------------------

# NOTE:
# Historically, PlasmaPy exposed `plasma_dispersion_func` and
# `plasma_dispersion_func_deriv` from `plasmapy.dispersion.dispersionfunction`,
# which also provided "lite" wrappers and bound them via decorators.
#
# This repository variant keeps everything in this single module to simplify
# maintenance while retaining the lite wrappers for backward compatibility.


import warnings

try:
    from plasmapy.utils.decorators import bind_lite_func, preserve_signature
except Exception:  # pragma: no cover
    # Fallbacks: keep import-time robustness if these decorators are unavailable.
    def preserve_signature(func):  # type: ignore
        return func

    def bind_lite_func(_lite_func):  # type: ignore
        def _decorator(func):
            return func
        return _decorator

try:
    from plasmapy.utils.exceptions import PlasmaPyFutureWarning
except Exception:  # pragma: no cover
    class PlasmaPyFutureWarning(FutureWarning):
        """Fallback PlasmaPyFutureWarning."""
        pass


@preserve_signature
def plasma_dispersion_func_lite(zeta):
    """Lite wrapper for :func:`plasma_dispersion_func` (deprecated)."""
    warnings.warn(
        (
            "plasma_dispersion_func_lite is deprecated and will be removed in a "
            "future release. Use plasmapy.dispersion.dispersion_functions."
            "plasma_dispersion_func instead."
        ),
        PlasmaPyFutureWarning,
        stacklevel=2,
    )
    return plasma_dispersion_func(zeta)


@bind_lite_func(plasma_dispersion_func_lite)
def plasma_dispersion_func_deprecated_location(
    zeta: complex | np.ndarray | u.Quantity,
) -> complex | np.ndarray | u.Quantity:
    """Deprecated alias matching the old import location."""
    warnings.warn(
        (
            "The old import location `plasmapy.dispersion.dispersionfunction."
            "plasma_dispersion_func` is deprecated. Use "
            "plasmapy.dispersion.dispersion_functions.plasma_dispersion_func instead."
        ),
        PlasmaPyFutureWarning,
        stacklevel=2,
    )
    return plasma_dispersion_func(zeta)


@preserve_signature
def plasma_dispersion_func_deriv_lite(zeta):
    """Lite wrapper for :func:`plasma_dispersion_func_deriv` (deprecated)."""
    warnings.warn(
        (
            "plasma_dispersion_func_deriv_lite is deprecated and will be removed in a "
            "future release. Use plasmapy.dispersion.dispersion_functions."
            "plasma_dispersion_func_deriv instead."
        ),
        PlasmaPyFutureWarning,
        stacklevel=2,
    )
    return plasma_dispersion_func_deriv(zeta)


@bind_lite_func(plasma_dispersion_func_deriv_lite)
def plasma_dispersion_func_deriv_deprecated_location(
    zeta: complex | np.ndarray | u.Quantity,
) -> complex | np.ndarray | u.Quantity:
    """Deprecated alias matching the old import location."""
    warnings.warn(
        (
            "The old import location `plasmapy.dispersion.dispersionfunction."
            "plasma_dispersion_func_deriv` is deprecated. Use "
            "plasmapy.dispersion.dispersion_functions.plasma_dispersion_func_deriv instead."
        ),
        PlasmaPyFutureWarning,
        stacklevel=2,
    )
    return plasma_dispersion_func_deriv(zeta)

# Extend public exports for compatibility
try:
    __all__ = list(__all__) + [
        'plasma_dispersion_func_lite',
        'plasma_dispersion_func_deriv_lite',
        'plasma_dispersion_func_deprecated_location',
        'plasma_dispersion_func_deriv_deprecated_location',
    ]
except Exception:
    pass
