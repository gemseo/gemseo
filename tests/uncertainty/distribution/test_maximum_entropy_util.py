# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the utils of the maximum-entropy distributions."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

import pytest
from mpmath import mp
from numpy import inf
from numpy import sign
from numpy.testing import assert_allclose
from scipy.stats import truncnorm

from gemseo.uncertainty.distribution import MaximumEntropyNumericalSettings
from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    compute_maximum_entropy_moments,
)
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    find_maximum_entropy_family,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_mean,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_tilt,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    _compute_reduced_moments,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    compute_truncated_normal_maximum_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    compute_truncated_normal_parameters,
)
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable

support_minimum: Final[float] = 2.0
"""The minimum of the support."""

support_maximum: Final[float] = 5.0
"""The maximum of the support."""

default_mu: Final[float] = 3.2
"""The default mean of the random variable."""

default_sigma: Final[float] = 0.4
"""The default standard deviation of the random variable."""

numerical_settings: Final[MaximumEntropyNumericalSettings] = (
    MaximumEntropyNumericalSettings()
)
"""The default numerical settings, as the util functions have no default values."""

tilt_settings: Final[dict[str, float]] = {
    "small_tilt": numerical_settings.small_tilt,
    "large_tilt": numerical_settings.large_tilt,
}
"""The tilt thresholds of the numerical settings."""

information_names: Final[tuple[str, ...]] = ("minimum", "maximum", "mu", "sigma")
"""The names of the information that resolves into a maximum-entropy distribution."""

presence_to_family: Final[dict[tuple[bool, bool, bool, bool], MaximumEntropyFamily]] = {
    (True, True, False, False): MaximumEntropyFamily.UNIFORM,
    (False, False, True, True): MaximumEntropyFamily.NORMAL,
    (True, False, True, False): MaximumEntropyFamily.EXPONENTIAL,
    (False, True, True, False): MaximumEntropyFamily.REFLECTED_EXPONENTIAL,
    (True, True, True, False): MaximumEntropyFamily.TILTED_UNIFORM,
    (True, False, True, True): MaximumEntropyFamily.TRUNCATED_NORMAL,
    (False, True, True, True): MaximumEntropyFamily.TRUNCATED_NORMAL,
    (True, True, True, True): MaximumEntropyFamily.TRUNCATED_NORMAL,
}
"""The presence of each information and the family it resolves into."""


def _compute_reference_moments(
    exponent: Callable[[mp.mpf], mp.mpf],
    peak: float,
    width: float,
    lower: float,
    upper: float,
) -> tuple[float, float]:
    """Compute the moments of an exponential-form density with mpmath.

    The exponent is shifted by its value at the peak of the density
    so that the integrand never overflows,
    and the integration is split at graded points around the peak
    so that the quadrature resolves a density much narrower than the support;
    a raw quadrature would confidently return wrong moments otherwise.

    Args:
        exponent: The exponent of the density.
        peak: The location of the maximum of the density over the support.
        width: The scale of the density around its peak.
        lower: The lower end of the support, possibly infinite.
        upper: The upper end of the support, possibly infinite.

    Returns:
        The mean and standard deviation of the distribution.
    """
    with mp.workdps(50):
        points = {mp.mpf(peak)} | {
            peak + sign * grading * width
            for grading in (1.0, 3.0, 10.0, 100.0)
            for sign in (1.0, -1.0)
        }
        edges = [
            mp.mpf(lower) if lower > -inf else -mp.inf,
            *sorted(point for point in points if lower < point < upper),
            mp.mpf(upper) if upper < inf else mp.inf,
        ]
        shift = exponent(mp.mpf(peak))

        def compute_weight(x: mp.mpf) -> mp.mpf:
            """Compute the shifted density.

            Args:
                x: The value of the random variable.

            Returns:
                The density, up to the constant factor of the shift.
            """
            return mp.e ** (exponent(x) - shift)

        normalization = mp.quad(compute_weight, edges)
        mean = mp.quad(lambda x: x * compute_weight(x), edges) / normalization
        variance = (
            mp.quad(lambda x: (x - mean) ** 2 * compute_weight(x), edges)
            / normalization
        )
        return float(mean), float(mp.sqrt(variance))


def _compute_tilted_uniform_moments(
    tilt: float, minimum: float, maximum: float
) -> tuple[float, float]:
    """Compute the reference moments of a tilted uniform distribution.

    Args:
        tilt: The tilt of the distribution.
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable.

    Returns:
        The mean and standard deviation of the distribution.
    """
    peak = maximum if tilt >= 0.0 else minimum
    width = 1.0 / abs(tilt) if tilt != 0.0 else maximum - minimum
    return _compute_reference_moments(lambda x: tilt * x, peak, width, minimum, maximum)


def _compute_truncated_normal_moments(
    normal_mu: float,
    normal_sigma: float,
    minimum: float | None,
    maximum: float | None,
) -> tuple[float, float]:
    """Compute the reference moments of a truncated normal distribution.

    Args:
        normal_mu: The mean of the normal distribution to be truncated.
        normal_sigma: The standard deviation of the normal distribution
            to be truncated.
        minimum: The minimum of the random variable.
            If `None`, the random variable is not bounded from below.
        maximum: The maximum of the random variable.
            If `None`, the random variable is not bounded from above.

    Returns:
        The mean and standard deviation of the truncated normal distribution.
    """
    lower = -inf if minimum is None else minimum
    upper = inf if maximum is None else maximum
    peak = min(max(normal_mu, lower), upper)
    if peak == normal_mu:
        width = normal_sigma
    else:
        width = min(normal_sigma, normal_sigma**2 / abs(peak - normal_mu))
    return _compute_reference_moments(
        lambda x: -((x - normal_mu) ** 2) / (2.0 * normal_sigma**2),
        peak,
        width,
        lower,
        upper,
    )


@pytest.mark.parametrize(
    ("presence", "family"),
    presence_to_family.items(),
    ids=[
        "-".join(
            name
            for name, given in zip(information_names, presence, strict=True)
            if given
        )
        for presence in presence_to_family
    ],
)
def test_find_maximum_entropy_family(presence, family) -> None:
    """Check the family of every combination of information."""
    values = (support_minimum, support_maximum, default_mu, default_sigma)
    kwargs = {
        name: value if is_given else None
        for name, value, is_given in zip(
            information_names, values, presence, strict=True
        )
    }
    assert find_maximum_entropy_family(**kwargs, settings=numerical_settings) == family


@pytest.mark.parametrize(
    "tilt",
    [-1e6, -40.0, -3.0, -1e-6, 0.0, 1e-6, 2.5, 40.0, 1e6],
    ids=[
        "very-large-negative",
        "large-negative",
        "moderate-negative",
        "tiny-negative",
        "zero",
        "tiny-positive",
        "moderate-positive",
        "large-positive",
        "very-large-positive",
    ],
)
def test_tilted_uniform_moments(tilt) -> None:
    """Check the moments of a tilted uniform distribution against a quadrature.

    The high-precision reference exercises the series and asymptotic expansions
    down to the double-precision resolution,
    which a low-precision reference could not do,
    as their error is far below its own;
    the extreme tilts check that the asymptotic expansions do not overflow
    while the mean tends to a bound of the support.
    """
    expected_mean, expected_std = _compute_tilted_uniform_moments(
        tilt, support_minimum, support_maximum
    )
    assert compute_tilted_uniform_mean(
        tilt, support_minimum, support_maximum, **tilt_settings
    ) == pytest.approx(expected_mean, rel=1e-12)
    assert compute_tilted_uniform_standard_deviation(
        tilt, support_minimum, support_maximum, **tilt_settings
    ) == pytest.approx(expected_std, rel=1e-12)


@pytest.mark.parametrize(
    "dimensionless_tilt", [5e-3, 2e-2], ids=["below-the-switch", "above-the-switch"]
)
def test_tilted_uniform_moments_near_series_switch(dimensionless_tilt) -> None:
    """Check that the series and the closed-form expressions agree near the switch."""
    tilt = dimensionless_tilt / (support_maximum - support_minimum)
    series_settings = {"small_tilt": 0.1, "large_tilt": 50.0}
    closed_form_settings = {"small_tilt": 1e-12, "large_tilt": 50.0}
    assert compute_tilted_uniform_mean(
        tilt, support_minimum, support_maximum, **series_settings
    ) == pytest.approx(
        compute_tilted_uniform_mean(
            tilt, support_minimum, support_maximum, **closed_form_settings
        ),
        rel=1e-10,
    )
    assert compute_tilted_uniform_standard_deviation(
        tilt, support_minimum, support_maximum, **series_settings
    ) == pytest.approx(
        compute_tilted_uniform_standard_deviation(
            tilt, support_minimum, support_maximum, **closed_form_settings
        ),
        rel=1e-8,
    )


@pytest.mark.parametrize(
    ("mu", "expected_sign"),
    [(2.05, -1.0), (2.5, -1.0), (3.5, 0.0), (4.2, 1.0), (4.95, 1.0)],
    ids=[
        "near-minimum",
        "below-middle",
        "at-middle",
        "above-middle",
        "near-maximum",
    ],
)
def test_compute_tilted_uniform_tilt(mu, expected_sign) -> None:
    """Check that the tilt results in the expected mean.

    The tilt has the sign of the offset of the mean
    from the middle of the support
    and is exactly zero for a mean at the middle.
    """
    tilt = compute_tilted_uniform_tilt(
        support_minimum, support_maximum, mu, settings=numerical_settings
    )
    assert compute_tilted_uniform_mean(
        tilt, support_minimum, support_maximum, **tilt_settings
    ) == pytest.approx(mu, rel=1e-7)
    assert sign(tilt) == expected_sign


mean_near_the_middle: Final[float] = (support_minimum + support_maximum) / 2 + 0.1
"""A mean slightly above the middle of the support."""


def test_tilt_within_the_tilt_tolerance_is_zero() -> None:
    """Check that a mean within the tilt tolerance has a zero tilt."""
    tilt = compute_tilted_uniform_tilt(
        support_minimum,
        support_maximum,
        mean_near_the_middle,
        settings=MaximumEntropyNumericalSettings(tilt_tolerance=0.1),
    )
    assert tilt == 0.0


@pytest.mark.parametrize(
    ("tilt_tolerance", "family"),
    [
        (0.1, MaximumEntropyFamily.UNIFORM),
        (
            MaximumEntropyNumericalSettings().tilt_tolerance,
            MaximumEntropyFamily.TILTED_UNIFORM,
        ),
    ],
    ids=["tolerance-at-the-distance-to-the-middle", "default-tolerance"],
)
def test_tilt_tolerance_sets_the_family(tilt_tolerance, family) -> None:
    """Check that the tilt tolerance decides between uniform and tilted uniform."""
    settings = MaximumEntropyNumericalSettings(tilt_tolerance=tilt_tolerance)
    assert (
        find_maximum_entropy_family(
            support_minimum,
            support_maximum,
            mean_near_the_middle,
            None,
            settings=settings,
        )
        == family
    )


def test_too_few_solver_iterations(snapshot) -> None:
    """Check the error raised when the tilt cannot be bracketed."""
    with assert_exception(ValueError, snapshot):
        compute_tilted_uniform_tilt(
            support_minimum,
            support_maximum,
            support_maximum - 1e-3,
            settings=MaximumEntropyNumericalSettings(maximum_solver_iterations=1),
        )


@pytest.mark.parametrize(
    ("minimum", "maximum", "mu", "sigma"),
    [
        (support_minimum, support_maximum, 3.5, 0.5),
        (support_minimum, support_maximum, 4.0, 0.5),
        (support_minimum, None, 3.0, 0.5),
        (None, support_maximum, 4.0, 0.7),
    ],
    ids=["bounded-at-middle", "bounded-above-middle", "minimum-only", "maximum-only"],
)
def test_compute_truncated_normal_parameters(minimum, maximum, mu, sigma) -> None:
    """Check that the truncated normal parameters result in the expected moments."""
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        minimum, maximum, mu, sigma, settings=numerical_settings
    )
    distribution = truncnorm(
        -inf if minimum is None else (minimum - normal_mu) / normal_sigma,
        inf if maximum is None else (maximum - normal_mu) / normal_sigma,
        loc=normal_mu,
        scale=normal_sigma,
    )
    assert distribution.mean() == pytest.approx(mu, rel=1e-7)
    assert distribution.std() == pytest.approx(sigma, rel=1e-7)


def test_compute_truncated_normal_parameters_error(snapshot) -> None:
    """Check the error raised when the truncated normal fit fails.

    No truncated normal distribution on this semi-infinite support
    has this standard deviation,
    which is twice the distance from the mean to the bound,
    i.e. twice the largest achievable one,
    so the solver fails for every initial curvature.
    The settings reject such information upstream, hence this direct call.
    """
    with assert_exception(ValueError, snapshot):
        compute_truncated_normal_parameters(
            -0.05, None, 0.0, 0.1, settings=numerical_settings
        )


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected"),
    [
        (support_minimum, None, default_mu - support_minimum),
        (None, support_maximum, support_maximum - default_mu),
    ],
    ids=["minimum-only", "maximum-only"],
)
def test_compute_truncated_normal_maximum_standard_deviation_semi_infinite(
    minimum, maximum, expected
) -> None:
    """Check the largest standard deviation for a semi-infinite support.

    This is the standard deviation of the shifted or reflected exponential distribution
    maximizing the entropy for the support and the mean only.
    """
    maximum_sigma = compute_truncated_normal_maximum_standard_deviation(
        minimum, maximum, default_mu, settings=numerical_settings
    )
    assert maximum_sigma == pytest.approx(expected, rel=1e-7)


def test_compute_truncated_normal_maximum_standard_deviation_bounded() -> None:
    """Check the largest standard deviation for a bounded support.

    This is the standard deviation of the tilted uniform distribution
    maximizing the entropy for the support and the mean only,
    checked against a quadrature.
    """
    maximum_sigma = compute_truncated_normal_maximum_standard_deviation(
        support_minimum, support_maximum, default_mu, settings=numerical_settings
    )
    tilt = compute_tilted_uniform_tilt(
        support_minimum, support_maximum, default_mu, settings=numerical_settings
    )
    _, expected = _compute_tilted_uniform_moments(
        tilt, support_minimum, support_maximum
    )
    assert maximum_sigma == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize(
    ("minimum", "maximum", "factor", "family"),
    [
        (
            support_minimum,
            None,
            1.0,
            MaximumEntropyFamily.EXPONENTIAL,
        ),
        (
            None,
            support_maximum,
            1.0,
            MaximumEntropyFamily.REFLECTED_EXPONENTIAL,
        ),
        (
            support_minimum,
            support_maximum,
            1.0,
            MaximumEntropyFamily.TILTED_UNIFORM,
        ),
        (
            support_minimum,
            None,
            0.9,
            MaximumEntropyFamily.TRUNCATED_NORMAL,
        ),
        (
            None,
            support_maximum,
            0.9,
            MaximumEntropyFamily.TRUNCATED_NORMAL,
        ),
        (
            support_minimum,
            support_maximum,
            0.9,
            MaximumEntropyFamily.TRUNCATED_NORMAL,
        ),
    ],
    ids=[
        "at-largest-minimum-only",
        "at-largest-maximum-only",
        "at-largest-minimum-and-maximum",
        "below-largest-minimum-only",
        "below-largest-maximum-only",
        "below-largest-minimum-and-maximum",
    ],
)
def test_largest_standard_deviation_family(minimum, maximum, factor, family) -> None:
    """Check the family at and below the largest standard deviation.

    At the largest standard deviation,
    the truncated normal distribution degenerates
    into the family maximizing the entropy for the support and the mean only.
    Below it, the truncated normal family is kept.
    """
    sigma = compute_truncated_normal_maximum_standard_deviation(
        minimum, maximum, default_mu, settings=numerical_settings
    )
    assert (
        find_maximum_entropy_family(
            minimum, maximum, default_mu, factor * sigma, settings=numerical_settings
        )
        == family
    )


@pytest.mark.parametrize(
    ("minimum", "maximum", "sigma"),
    [
        # With a single bound, the largest standard deviation
        # of a maximum-entropy distribution is the distance from the mean
        # to that bound, namely 1.2 and 1.8.
        (support_minimum, None, 1.5),
        (None, support_maximum, 2.0),
        # With both bounds, that largest standard deviation is about 0.83
        # and that of the distribution concentrated at the bounds is about 1.47;
        # the standard deviations below lie between them and above them.
        # They are literals rather than multiples of a value computed by the tilt
        # solver, whose last digits depend on the platform and would then appear
        # in the messages compared with the snapshots.
        (support_minimum, support_maximum, 1.0),
        (support_minimum, support_maximum, 1.5),
    ],
    ids=[
        "minimum-only",
        "maximum-only",
        "bounded-log-convex",
        "bounded-unachievable",
    ],
)
def test_too_large_standard_deviation(minimum, maximum, sigma, snapshot) -> None:
    """Check the errors raised when the standard deviation is too large.

    Above the largest standard deviation of a maximum-entropy distribution,
    the error explains why no truncated normal distribution can be delivered;
    on a bounded support,
    a standard deviation at or above that of the distribution
    concentrated at the bounds is not achievable by any distribution at all
    and gets its own error.
    """
    with assert_exception(ValueError, snapshot):
        find_maximum_entropy_family(
            minimum, maximum, default_mu, sigma, settings=numerical_settings
        )


@pytest.mark.parametrize(
    ("minimum", "maximum", "mu", "shrinkage"),
    [
        (0.0, 1.0, 0.2, 1.05e-6),
        (0.0, 1.0, 0.2, 1e-4),
        (0.0, 1.0, 0.001, 1.05e-6),
        (0.0, 1.0, 0.001, 1e-2),
        (support_minimum, support_maximum, default_mu, 1e-5),
        (support_minimum, None, default_mu, 1e-5),
        (support_minimum, None, default_mu, 1e-4),
        (None, support_maximum, default_mu, 3e-6),
        (None, support_maximum, default_mu, 1e-3),
    ],
    ids=[
        "unit-middle-tiny-shrinkage",
        "unit-middle-small-shrinkage",
        "unit-near-minimum-tiny-shrinkage",
        "unit-near-minimum-moderate-shrinkage",
        "bounded",
        "minimum-only-small-shrinkage",
        "minimum-only-larger-shrinkage",
        "maximum-only-tiny-shrinkage",
        "maximum-only-larger-shrinkage",
    ],
)
def test_nearly_largest_standard_deviation(minimum, maximum, mu, shrinkage) -> None:
    """Check the fit just below the largest standard deviation.

    With an off-center mean,
    the truncated normal distribution degenerates
    near the largest standard deviation;
    a fit parametrized by the standard deviation of the distribution
    to be truncated used to fail there,
    with a ValueError or even an OverflowError,
    whereas the fit in the natural parameters
    remains regular up to the boundary.
    """
    sigma = compute_truncated_normal_maximum_standard_deviation(
        minimum, maximum, mu, settings=numerical_settings
    ) * (1.0 - shrinkage)
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        minimum, maximum, mu, sigma, settings=numerical_settings
    )
    mean, std = _compute_truncated_normal_moments(
        normal_mu, normal_sigma, minimum, maximum
    )
    assert mean == pytest.approx(mu, rel=1e-7)
    assert std == pytest.approx(sigma, rel=1e-7)


@pytest.mark.parametrize(
    ("length", "reduced_mu"),
    [(1e-300, 0.6), (1e30, 0.6), (1e6, 0.500001)],
    ids=["extremely-short", "extremely-long", "long-near-middle"],
)
def test_tilt_solver_extreme_support_lengths(length, reduced_mu) -> None:
    """Check the tilt solver on extremely short and long supports.

    The bracket of the solver is dimensionless like its tolerance;
    a fixed bracket would need more doublings than the allowed iterations
    on an extremely short support
    and more root-finding iterations than the solver allows on a long one.
    The solver tolerance applies to the dimensionless tilt;
    an absolute tolerance on the tilt itself
    would lose the mean to an error growing with the square of the support length.

    Args:
        length: The length of the support.
        reduced_mu: The mean divided by the length.
    """
    mu = reduced_mu * length
    tilt = compute_tilted_uniform_tilt(0.0, length, mu, settings=numerical_settings)
    mean = compute_tilted_uniform_mean(tilt, 0.0, length, **tilt_settings)
    assert mean / length == pytest.approx(reduced_mu, rel=1e-12)


@pytest.mark.parametrize(
    ("minimum", "maximum", "mu", "sigma"),
    [
        (0.0, 1.0, 0.5, 1e-5),
        (0.0, 1.0, 0.001, 1e-5),
        (0.0, 1.0, 0.5, 1e-8),
        (0.0, None, 1.0, 1e-4),
    ],
    ids=[
        "bounded-middle",
        "bounded-near-minimum",
        "bounded-tiny-sigma",
        "minimum-only",
    ],
)
def test_weakly_truncated_normal(minimum, maximum, mu, sigma) -> None:
    """Check the parameters when the truncation is negligible.

    When every bound is far enough from the mean in standard deviations,
    by a margin derived from the solver tolerance,
    the truncation moves the moments by less than this tolerance
    and the parameters are those of the normal distribution itself;
    these cases used to underflow the quadrature,
    whose initial grid is much coarser than such a narrow density.
    """
    assert compute_truncated_normal_parameters(
        minimum, maximum, mu, sigma, settings=numerical_settings
    ) == (mu, sigma)


def test_weak_truncation_boundary() -> None:
    """Check the continuity of the parameters across the shortcut boundary.

    Just below the margin the parameters are fitted;
    they must match the normal parameters returned by the shortcut
    to within the solver tolerance,
    which validates the value of the margin.
    """
    mu = 0.5
    # The margin is about 8.43 at the default tolerance;
    # at 7.85 standard deviations the truncation still moves the moments
    # by about 1e-13, measurable in double precision yet below the tolerance.
    sigma = mu / 7.85
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        0.0, 1.0, mu, sigma, settings=numerical_settings
    )
    assert (normal_mu, normal_sigma) != (mu, sigma)
    assert normal_mu == pytest.approx(mu, rel=1e-12)
    assert normal_sigma == pytest.approx(sigma, rel=1e-12)


def test_truncated_normal_parameters_scale_invariance() -> None:
    """Check the fit on a semi-infinite support far from a unit scale.

    The reduction uses the distance from the mean to the bound as its unit;
    with a fixed unit,
    the quadrature missed the mass of the density
    and the solver silently returned its initial guess.
    """
    scale = 1e6
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        0.0, None, scale, 0.9 * scale, settings=numerical_settings
    )
    unit_mu, unit_sigma = compute_truncated_normal_parameters(
        0.0, None, 1.0, 0.9, settings=numerical_settings
    )
    assert normal_mu / scale == pytest.approx(unit_mu, rel=1e-7)
    assert normal_sigma / scale == pytest.approx(unit_sigma, rel=1e-7)


@pytest.mark.parametrize(
    "mu",
    [0.8, 0.999, 1.0 - 1e-7],
    ids=["near-maximum", "very-near-maximum", "extremely-near-maximum"],
)
def test_truncated_normal_parameters_reflection(mu) -> None:
    """Check the reflection invariance of the truncated normal fit.

    The support is reduced so that the density peaks near its origin;
    a mean near the maximum must thus be fitted as accurately
    as its mirror image near the minimum,
    which a reduction always anchored at the minimum could not do,
    as the exponent of the reduced density cancels out at the other end.
    """
    sigma = 0.2 * (1.0 - mu)
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        0.0, 1.0, mu, sigma, settings=numerical_settings
    )
    mirror_mu, mirror_sigma = compute_truncated_normal_parameters(
        0.0, 1.0, 1.0 - mu, sigma, settings=numerical_settings
    )
    assert normal_mu == pytest.approx(1.0 - mirror_mu, rel=1e-7)
    assert normal_sigma == pytest.approx(mirror_sigma, rel=1e-7)


def test_truncated_normal_parameters_far_bound() -> None:
    """Check that a bound beyond the weak-truncation margin is dropped.

    The reduction maps a bounded support onto the unit interval
    whatever its length,
    so a far bound squeezes the density into a sliver
    that the quadrature cannot resolve,
    although this bound moves the moments by less than the solver tolerance;
    the parameters must be those of the fit without the far bound.
    """
    two_sided = compute_truncated_normal_parameters(
        -1e9, 2.0, 0.0, 1.0, settings=numerical_settings
    )
    one_sided = compute_truncated_normal_parameters(
        None, 2.0, 0.0, 1.0, settings=numerical_settings
    )
    assert_allclose(two_sided, one_sided, rtol=1e-12)


def test_truncated_normal_parameters_far_bound_with_truncation() -> None:
    """Check that a far bound is kept when the other bound truncates the density.

    A bound beyond the Gaussian margin cuts off a negligible mass
    only when no bound truncates the density.
    Otherwise the fitted normal distribution is centered outside the support
    and the density retained on it decays like exp(-z) instead of exp(-z^2/2),
    so the mass cut off by such a bound is not negligible at all;
    dropping it used to move the standard deviation by about 2e-4 relative,
    silently and far beyond the solver tolerance.
    """
    minimum = 0.0
    maximum = 1.0
    mu = 0.9
    # The minimum is about 9.5 standard deviations away from the mean,
    # hence beyond the Gaussian margin of about 8.4,
    # whereas the maximum truncates the density
    # about one standard deviation away from the mean.
    sigma = 0.095
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        minimum, maximum, mu, sigma, settings=numerical_settings
    )
    distribution = truncnorm(
        (minimum - normal_mu) / normal_sigma,
        (maximum - normal_mu) / normal_sigma,
        loc=normal_mu,
        scale=normal_sigma,
    )
    assert distribution.mean() == pytest.approx(mu, rel=1e-12)
    assert distribution.std() == pytest.approx(sigma, rel=1e-12)


def test_truncated_normal_parameters_underflowing_scale() -> None:
    """Check a support and a standard deviation near the underflow limit.

    The reduction of such a support used to squeeze the density into a sliver,
    whose second reduced moment underflowed to zero and crashed the fit;
    the weak-truncation margins now drop the maximum beforehand.
    """
    normal_mu, normal_sigma = compute_truncated_normal_parameters(
        0.0, 1.0, 8e-155, 1e-155, settings=numerical_settings
    )
    # The maximum is dropped as it lies beyond the weak-truncation margin
    # and the minimum is eight standard deviations away from the mean,
    # so the normal parameters are the requested moments
    # up to a relative correction of about z*phi(z) at z=8, i.e. 4e-14.
    assert normal_mu / 8e-155 == pytest.approx(1.0, rel=1e-7)
    assert normal_sigma / 1e-155 == pytest.approx(1.0, rel=1e-7)


@pytest.mark.parametrize(
    ("tilt", "curvature", "bounded"),
    [
        (1e5, -1.0, True),
        (0.0, -1e308, True),
        (0.0, -1e308, False),
        (0.0, -1e220, True),
    ],
    ids=["shift", "normalization", "normalization_unbounded", "underflow"],
)
def test_unresolvable_reduced_moments(tilt, curvature, bounded) -> None:
    """Check the parameters for which the quadrature cannot resolve the density.

    Each row pins a numerical failure mode
    that silently corrupted the moments in double precision:
    a shifted exponent so large that the density is lost in the roundoff,
    a normalization that is no longer a positive finite number,
    on a bounded support and on a semi-infinite one,
    and a second central moment that underflows to zero.
    The fit fed by these moments cannot reach such parameters,
    as the weak-truncation margins drop the bounds
    that would squeeze the density beforehand,
    hence this direct call.
    """
    assert _compute_reduced_moments(tilt, curvature, bounded) is None


def test_flat_reduced_moments() -> None:
    """Check the moments of a reduced density that needs no breakpoint.

    A zero tilt and a zero curvature make the reduced density
    that of the uniform distribution over the unit interval,
    whose scale is the support itself,
    so the quadrature resolves it without graded breakpoints;
    the moments are those of this distribution,
    namely 1/12, 0 and 1/80 for the central ones.
    """
    assert_allclose(
        _compute_reduced_moments(0.0, 0.0, True),
        (0.5, 1.0 / 12.0, 0.0, 1.0 / 80.0),
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize(
    ("family", "parameters", "moments"),
    [
        (
            MaximumEntropyFamily.UNIFORM,
            {"minimum": support_minimum, "maximum": support_maximum},
            (
                (support_minimum + support_maximum) / 2,
                (support_maximum - support_minimum) / 12**0.5,
            ),
        ),
        (
            MaximumEntropyFamily.NORMAL,
            {"mu": default_mu, "sigma": default_sigma},
            (default_mu, default_sigma),
        ),
        (
            MaximumEntropyFamily.EXPONENTIAL,
            {"rate": 2.0, "minimum": support_minimum},
            (support_minimum + 0.5, 0.5),
        ),
        (
            MaximumEntropyFamily.REFLECTED_EXPONENTIAL,
            {"rate": 2.0, "maximum": support_maximum},
            (support_maximum - 0.5, 0.5),
        ),
        (
            MaximumEntropyFamily.TILTED_UNIFORM,
            {"tilt": 1.5, "minimum": support_minimum, "maximum": support_maximum},
            # The tilted expressions are validated against the mpmath reference
            # by test_tilted_uniform_moments; this row checks the delegation.
            (
                compute_tilted_uniform_mean(
                    1.5, support_minimum, support_maximum, **tilt_settings
                ),
                compute_tilted_uniform_standard_deviation(
                    1.5, support_minimum, support_maximum, **tilt_settings
                ),
            ),
        ),
    ],
    ids=[
        "uniform",
        "normal",
        "exponential",
        "reflected-exponential",
        "tilted-uniform",
    ],
)
def test_compute_maximum_entropy_moments(family, parameters, moments) -> None:
    """Check the closed-form moments of the maximum-entropy families."""
    assert_allclose(
        compute_maximum_entropy_moments(
            family, parameters, settings=numerical_settings
        ),
        moments,
        rtol=1e-12,
    )


def test_compute_maximum_entropy_moments_error(snapshot) -> None:
    """Check the error raised for the family without closed-form moments."""
    with assert_exception(ValueError, snapshot):
        compute_maximum_entropy_moments(
            MaximumEntropyFamily.TRUNCATED_NORMAL, {}, settings=numerical_settings
        )
