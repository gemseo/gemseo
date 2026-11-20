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
"""Tests for the maximum-entropy distributions of both libraries."""

from __future__ import annotations

import logging
from typing import Final

import pytest
from numpy import inf
from numpy.testing import assert_allclose
from numpy.testing import assert_equal
from pydantic import ValidationError

from gemseo.space import RandomSpace
from gemseo.uncertainty import create_distribution
from gemseo.uncertainty import get_available_distributions
from gemseo.uncertainty.distribution import MaximumEntropyNumericalSettings
from gemseo.uncertainty.distribution import OTMaximumEntropyDistribution
from gemseo.uncertainty.distribution import OTMaximumEntropyDistribution_Settings
from gemseo.uncertainty.distribution import SPMaximumEntropyDistribution
from gemseo.uncertainty.distribution import SPMaximumEntropyDistribution_Settings
from gemseo.uncertainty.distribution import distribution_factory
from gemseo.uncertainty.distribution._maximum_entropy.family import MaximumEntropyFamily
from gemseo.uncertainty.distribution._maximum_entropy.selection import (
    compute_maximum_entropy_parameters,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_tilt,
)
from gemseo.uncertainty.distribution._maximum_entropy.truncated_normal import (
    compute_truncated_normal_maximum_standard_deviation,
)
from gemseo.util.testing.helper import assert_exception

logger_name: Final[str] = "gemseo.uncertainty.distribution.core.base_maximum_entropy"
"""The name of the logger of the maximum-entropy distributions."""

substitution_warning: Final[str] = (
    "The truncated normal distribution maximizing the entropy "
    "for this information cannot be delivered accurately by the library; "
    "the {} distribution maximizing the entropy "
    "for the support and the mean only is used instead, "
    "with the standard deviation {} instead of {}."
)
"""The template of the warning of the substitution of the truncated normal."""

support_minimum: Final[float] = 2.0
"""The minimum of the support."""

support_maximum: Final[float] = 5.0
"""The maximum of the support."""

support_middle: Final[float] = (support_minimum + support_maximum) / 2
"""The middle of the support."""

numerical_settings: Final[MaximumEntropyNumericalSettings] = (
    MaximumEntropyNumericalSettings()
)
"""The default numerical settings, as the util functions have no default values."""


def _compute_shrunk_maximum_sigma(
    minimum: float | None,
    maximum: float | None,
    mu: float,
    shrinkage: float,
) -> float:
    """Compute a standard deviation just below the largest deliverable one.

    Args:
        minimum: The minimum of the random variable, if any.
        maximum: The maximum of the random variable, if any.
        mu: The mean of the random variable.
        shrinkage: The relative distance below the largest standard deviation.

    Returns:
        The shrunk standard deviation.
    """
    return compute_truncated_normal_maximum_standard_deviation(
        minimum, maximum, mu, settings=numerical_settings
    ) * (1.0 - shrinkage)


common_information: Final[
    dict[str, tuple[dict[str, float], MaximumEntropyFamily, float, float]]
] = {
    "uniform": (
        {"minimum": support_minimum, "maximum": support_maximum},
        MaximumEntropyFamily.UNIFORM,
        support_middle,
        (support_maximum - support_minimum) / 12**0.5,
    ),
    "normal": ({"mu": 1.0, "sigma": 2.0}, MaximumEntropyFamily.NORMAL, 1.0, 2.0),
    "exponential": (
        {"minimum": support_minimum, "mu": 3.0},
        MaximumEntropyFamily.EXPONENTIAL,
        3.0,
        3.0 - support_minimum,
    ),
    "decreasing_tilted_uniform": (
        {"minimum": support_minimum, "maximum": support_maximum, "mu": 3.0},
        MaximumEntropyFamily.TILTED_UNIFORM,
        3.0,
        0.7772281368,
    ),
    "truncated_normal_from_below": (
        {"minimum": support_minimum, "mu": 3.0, "sigma": 0.5},
        MaximumEntropyFamily.TRUNCATED_NORMAL,
        3.0,
        0.5,
    ),
    "truncated_normal_from_above": (
        {"maximum": support_maximum, "mu": 4.0, "sigma": 0.5},
        MaximumEntropyFamily.TRUNCATED_NORMAL,
        4.0,
        0.5,
    ),
    "truncated_normal": (
        {
            "minimum": support_minimum,
            "maximum": support_maximum,
            "mu": 4.0,
            "sigma": 0.5,
        },
        MaximumEntropyFamily.TRUNCATED_NORMAL,
        4.0,
        0.5,
    ),
}
"""The information that both libraries support,
with the family of the resulting distribution,
its mean and its standard deviation.

The standard deviation of the uniform distribution is (maximum - minimum) / 12**0.5,
the one of the exponential distribution is its mean minus its minimum
and the one of the tilted uniform distribution comes from the exponential density
exp(-t * (x - minimum)) on the support, with t solving the mean (t = 0.71637...),
integrated independently by adaptive quadrature.
"""

openturns_information: Final[
    dict[str, tuple[dict[str, float], MaximumEntropyFamily]]
] = {
    "reflected_exponential": (
        {"maximum": support_maximum, "mu": 4.0},
        MaximumEntropyFamily.REFLECTED_EXPONENTIAL,
    ),
    "increasing_tilted_uniform": (
        {"minimum": support_minimum, "maximum": support_maximum, "mu": 4.2},
        MaximumEntropyFamily.TILTED_UNIFORM,
    ),
}
"""The information that only OpenTURNS supports."""

ot_classes: Final[tuple[type, type]] = (
    OTMaximumEntropyDistribution,
    OTMaximumEntropyDistribution_Settings,
)
"""The distribution and settings classes of OpenTURNS."""

sp_classes: Final[tuple[type, type]] = (
    SPMaximumEntropyDistribution,
    SPMaximumEntropyDistribution_Settings,
)
"""The distribution and settings classes of SciPy."""


@pytest.fixture(params=[ot_classes, sp_classes], ids=["openturns", "scipy"])
def classes(request):
    """The distribution and settings classes of a library."""
    return request.param


@pytest.mark.parametrize(
    "information", common_information.values(), ids=common_information
)
def test_moments_and_support(classes, information) -> None:
    """Check that the distribution realizes the information given by the user."""
    distribution_class, settings_class = classes
    kwargs, _, mean, standard_deviation = information
    distribution = distribution_class(settings_class(**kwargs))
    assert distribution.mean == pytest.approx(mean, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(
        standard_deviation, rel=1e-7
    )
    assert_equal(
        distribution.support,
        [kwargs.get("minimum", -inf), kwargs.get("maximum", inf)],
    )


@pytest.mark.parametrize(
    ("minimum", "maximum", "mu"),
    [(support_minimum, support_maximum, 3.0), (support_minimum, None, 3.0)],
    ids=["bounded", "bounded-below"],
)
@pytest.mark.parametrize(
    "shrinkage",
    [0.0, 1e-5, 1e-3, 1e-2],
    ids=["largest", "shrinkage-1e-5", "shrinkage-1e-3", "shrinkage-1e-2"],
)
def test_nearly_largest_standard_deviation_delivery(
    classes, minimum, maximum, mu, shrinkage
) -> None:
    """Check the delivered moments at and just below the largest standard deviation.

    In this regime,
    the library either represents the exact truncated normal distribution
    or substitutes the distribution maximizing the entropy
    for the support and the mean only,
    whose standard deviation differs from the requested one
    by the relative shrinkage;
    the delivered standard deviation must be at least that close.
    A zero shrinkage requests the largest standard deviation itself,
    where the truncated normal distribution degenerates
    into the substituted family.
    These cases used to raise errors or deliver wrong moments.
    """
    distribution_class, settings_class = classes
    sigma = _compute_shrunk_maximum_sigma(minimum, maximum, mu, shrinkage)
    distribution = distribution_class(
        settings_class(minimum=minimum, maximum=maximum, mu=mu, sigma=sigma)
    )
    assert distribution.mean == pytest.approx(mu, rel=1e-6)
    assert distribution.standard_deviation == pytest.approx(
        sigma, rel=max(2.0 * shrinkage, 1e-12)
    )


def test_substituted_family_is_visible(classes, caplog) -> None:
    """Check that a substituted boundary family is visible to the user.

    Near the largest standard deviation,
    both libraries substitute the tilted uniform distribution
    for the truncated normal one they cannot deliver accurately;
    the string representation must show the delivered family
    and a warning must state the substitution
    and the delivered standard deviation.
    """
    distribution_class, settings_class = classes
    mu = 3.0
    sigma = _compute_shrunk_maximum_sigma(support_minimum, support_maximum, mu, 1e-5)
    distribution = distribution_class(
        settings_class(
            minimum=support_minimum, maximum=support_maximum, mu=mu, sigma=sigma
        )
    )
    assert str(MaximumEntropyFamily.TILTED_UNIFORM) in repr(distribution)
    assert (
        logger_name,
        logging.WARNING,
        substitution_warning.format(
            MaximumEntropyFamily.TILTED_UNIFORM, distribution.standard_deviation, sigma
        ),
    ) in caplog.record_tuples


def test_substituted_family_with_tilt_tolerance(caplog) -> None:
    """Check the substituted family when the tilt is within its tolerance.

    Near the largest standard deviation,
    a tilt tolerance greater than the relative distance from the mean
    to the middle of the support
    makes the substituted distribution uniform instead of tilted uniform,
    consistently with the resolution of the family from the settings.
    Only OpenTURNS substitutes here:
    it cannot build this truncated normal distribution at all,
    while SciPy delivers it more accurately
    than a uniform distribution missing the mean by the absorbed tilt.
    """
    mu = 3.0
    sigma = _compute_shrunk_maximum_sigma(support_minimum, support_maximum, mu, 1e-5)
    settings = OTMaximumEntropyDistribution_Settings(
        minimum=support_minimum,
        maximum=support_maximum,
        mu=mu,
        sigma=sigma,
        numerical_settings=MaximumEntropyNumericalSettings(tilt_tolerance=0.2),
    )
    distribution = OTMaximumEntropyDistribution(settings)
    assert str(MaximumEntropyFamily.UNIFORM) in repr(distribution)
    assert (
        logger_name,
        logging.WARNING,
        substitution_warning.format(
            MaximumEntropyFamily.UNIFORM, distribution.standard_deviation, sigma
        ),
    ) in caplog.record_tuples
    assert distribution.mean == pytest.approx(support_middle, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(
        (support_maximum - support_minimum) / 12**0.5, rel=1e-7
    )


def _create_near_largest_standard_deviation(classes, shrinkage):
    """Create the distribution of a standard deviation just below the largest one.

    Args:
        classes: The distribution and settings classes of a library.
        shrinkage: The relative distance below the largest standard deviation.

    Returns:
        The distribution, its settings, the parameters of the truncated normal
        distribution, the requested standard deviation and the largest one.
    """
    distribution_class, settings_class = classes
    mu = 3.0
    maximum_sigma = compute_truncated_normal_maximum_standard_deviation(
        support_minimum, support_maximum, mu, settings=numerical_settings
    )
    sigma = maximum_sigma * (1.0 - shrinkage)
    settings = settings_class(
        minimum=support_minimum, maximum=support_maximum, mu=mu, sigma=sigma
    )
    family, parameters = compute_maximum_entropy_parameters(
        support_minimum, support_maximum, mu, sigma, settings=numerical_settings
    )
    assert family == MaximumEntropyFamily.TRUNCATED_NORMAL
    return distribution_class(settings), settings, parameters, sigma, maximum_sigma


@pytest.mark.parametrize(
    ("classes", "shrinkage"),
    [(ot_classes, 1e-1), (sp_classes, 1e-4), (sp_classes, 1e-1)],
    ids=["openturns-1e-1", "scipy-1e-4", "scipy-1e-1"],
)
def test_kept_truncated_normal_is_more_accurate(classes, shrinkage) -> None:
    """Check that the truncated normal distribution is kept when it is accurate.

    The gating keeps the truncated normal distribution
    when the moment error of the library delivering it
    is at most the substitution error,
    i.e. the distance from the requested standard deviation to the largest one;
    the delivered moments are then the requested ones.
    """
    distribution, settings, parameters, sigma, maximum_sigma = (
        _create_near_largest_standard_deviation(classes, shrinkage)
    )
    assert str(MaximumEntropyFamily.TRUNCATED_NORMAL) in repr(distribution)
    assert distribution.mean == pytest.approx(settings.mu, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(sigma, rel=1e-7)
    measured = distribution._compute_truncated_normal_representation_error(
        parameters, settings
    )
    assert measured <= maximum_sigma - sigma


def test_substituted_truncated_normal_is_more_accurate() -> None:
    """Check that the tilted uniform distribution is substituted when it is accurate.

    OpenTURNS delivers the truncated normal distribution
    with a moment error greater than the substitution error
    when the standard deviation is just below the largest one,
    so the tilted uniform distribution is substituted;
    its delivered standard deviation is as close as the shrinkage to the requested one.
    """
    shrinkage = 1e-4
    distribution, settings, _, sigma, _ = _create_near_largest_standard_deviation(
        ot_classes, shrinkage
    )
    assert str(MaximumEntropyFamily.TILTED_UNIFORM) in repr(distribution)
    assert distribution.mean == pytest.approx(settings.mu, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(sigma, rel=2.0 * shrinkage)


def test_nearly_largest_standard_deviation_symmetric(classes) -> None:
    """Check the selected family for a mean at the middle of the support.

    Within the boundary slack of the largest standard deviation,
    the requested standard deviation is treated as the largest one;
    the tilt is zero when the mean is at the middle of the support,
    so the selected family is the uniform one.
    """
    distribution_class, settings_class = classes
    shrinkage = 1e-11
    sigma = _compute_shrunk_maximum_sigma(
        support_minimum, support_maximum, support_middle, shrinkage
    )
    distribution = distribution_class(
        settings_class(
            minimum=support_minimum,
            maximum=support_maximum,
            mu=support_middle,
            sigma=sigma,
        )
    )
    assert str(MaximumEntropyFamily.UNIFORM) in repr(distribution)
    assert distribution.mean == pytest.approx(support_middle, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(sigma, rel=2.0 * shrinkage)


@pytest.mark.parametrize(
    ("minimum", "mu", "shrinkage", "family"),
    [
        (None, 4.0, 1e-5, MaximumEntropyFamily.REFLECTED_EXPONENTIAL),
        (support_minimum, 4.2, 3e-6, MaximumEntropyFamily.TILTED_UNIFORM),
    ],
    ids=["reflected", "increasing"],
)
def test_nearly_largest_standard_deviation_openturns(
    minimum, mu, shrinkage, family
) -> None:
    """Check the substituted family just below the largest standard deviation.

    The substituted distribution is
    the reflected exponential one for a support bounded from above only,
    and one with an increasing probability density function
    for a support bounded on both sides.
    OpenTURNS delivers them.

    Args:
        minimum: The lower bound of the support, if any.
        mu: The mean.
        shrinkage: The relative distance to the largest standard deviation.
        family: The expected family.
    """
    sigma = _compute_shrunk_maximum_sigma(minimum, support_maximum, mu, shrinkage)
    distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(
            minimum=minimum, maximum=support_maximum, mu=mu, sigma=sigma
        )
    )
    assert str(family) in repr(distribution)
    assert distribution.standard_deviation == pytest.approx(sigma, rel=2.0 * shrinkage)


@pytest.mark.parametrize(
    ("minimum", "mu", "shrinkage"),
    [(None, 4.0, 1e-5), (support_minimum, 4.2, 3e-6)],
    ids=["reflected", "increasing"],
)
def test_nearly_largest_standard_deviation_scipy(minimum, mu, shrinkage) -> None:
    """Check the kept family just below the largest standard deviation.

    SciPy implements neither the reflected exponential distribution
    nor the distribution with an increasing probability density function
    and keeps the truncated normal distribution instead,
    so that settings accepted by the validation always construct.

    Args:
        minimum: The lower bound of the support, if any.
        mu: The mean.
        shrinkage: The relative distance to the largest standard deviation.
    """
    sigma = _compute_shrunk_maximum_sigma(minimum, support_maximum, mu, shrinkage)
    distribution = SPMaximumEntropyDistribution(
        SPMaximumEntropyDistribution_Settings(
            minimum=minimum, maximum=support_maximum, mu=mu, sigma=sigma
        )
    )
    assert str(MaximumEntropyFamily.TRUNCATED_NORMAL) in repr(distribution)
    assert distribution.mean == pytest.approx(mu, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(sigma, rel=1e-7)


@pytest.mark.parametrize(
    "information", openturns_information.values(), ids=openturns_information
)
def test_moments_and_support_openturns_only(information) -> None:
    """Check the distributions that only OpenTURNS can create."""
    kwargs, _ = information
    distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(**kwargs)
    )
    assert distribution.mean == pytest.approx(kwargs["mu"], rel=1e-7)
    assert_equal(
        distribution.support,
        [kwargs.get("minimum", -inf), kwargs.get("maximum", inf)],
    )


@pytest.mark.parametrize(
    "information",
    list(common_information.values()) + list(openturns_information.values()),
    ids=list(common_information) + list(openturns_information),
)
def test_resolved_family_is_visible(information) -> None:
    """Check that the resolved distribution family is visible to the user."""
    kwargs, family, *_ = information
    distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(**kwargs)
    )
    assert str(family) in repr(distribution)
    for name, value in kwargs.items():
        assert f"{name}={value}" in repr(distribution)


@pytest.mark.parametrize(
    "information", common_information.values(), ids=common_information
)
def test_libraries_agree(information) -> None:
    """Check that both libraries realize the same distribution."""
    kwargs, *_ = information
    ot_distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(**kwargs)
    )
    sp_distribution = SPMaximumEntropyDistribution(
        SPMaximumEntropyDistribution_Settings(**kwargs)
    )
    assert ot_distribution.mean == pytest.approx(sp_distribution.mean, rel=1e-7)
    assert ot_distribution.standard_deviation == pytest.approx(
        sp_distribution.standard_deviation, rel=1e-7
    )
    assert ot_distribution.compute_cdf(support_middle) == pytest.approx(
        sp_distribution.compute_cdf(support_middle), rel=1e-7
    )


@pytest.mark.parametrize(
    "information", openturns_information.values(), ids=openturns_information
)
def test_scipy_error(information, snapshot) -> None:
    """Check the error raised when SciPy cannot create the distribution."""
    kwargs, _ = information
    with assert_exception(ValidationError, snapshot):
        SPMaximumEntropyDistribution_Settings(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"minimum": support_minimum},
        {"maximum": support_maximum},
        {"mu": 3.0},
        {"sigma": 0.5},
        {"minimum": support_minimum, "sigma": 0.5},
        {"maximum": support_maximum, "sigma": 0.5},
        {"minimum": support_minimum, "maximum": support_maximum, "sigma": 0.5},
    ],
    ids=[
        "nothing",
        "minimum",
        "maximum",
        "mu",
        "sigma",
        "minimum-sigma",
        "maximum-sigma",
        "minimum-maximum-sigma",
    ],
)
def test_information_error(classes, kwargs, snapshot) -> None:
    """Check the error raised when no distribution maximizes the entropy."""
    _, settings_class = classes
    with assert_exception(ValidationError, snapshot):
        settings_class(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": support_maximum, "maximum": support_minimum},
        {"minimum": support_minimum, "maximum": support_maximum, "mu": support_minimum},
        {"minimum": support_minimum, "maximum": support_maximum, "mu": support_maximum},
        {"minimum": support_minimum, "mu": support_minimum},
        {
            "minimum": support_minimum,
            "maximum": support_maximum,
            "mu": 3.0,
            "sigma": 10.0,
        },
    ],
    ids=[
        "minimum-above-maximum",
        "mu-at-minimum-with-maximum",
        "mu-at-maximum",
        "mu-at-minimum-without-maximum",
        "sigma-above-largest",
    ],
)
def test_inconsistent_information_error(kwargs, snapshot) -> None:
    """Check the error raised when the information is inconsistent."""
    with assert_exception(ValidationError, snapshot):
        OTMaximumEntropyDistribution_Settings(**kwargs)


def test_mean_at_the_middle_of_the_support() -> None:
    """Check that a mean at the middle of the support gives a uniform distribution."""
    distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(
            minimum=support_minimum, maximum=support_maximum, mu=support_middle
        )
    )
    assert str(MaximumEntropyFamily.UNIFORM) in repr(distribution)
    assert distribution.mean == pytest.approx(support_middle, rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(
        (support_maximum - support_minimum) / 12**0.5, rel=1e-7
    )


@pytest.mark.parametrize(
    ("tilt_tolerance", "family"),
    [
        (
            MaximumEntropyNumericalSettings().tilt_tolerance,
            MaximumEntropyFamily.TILTED_UNIFORM,
        ),
        (0.1, MaximumEntropyFamily.UNIFORM),
    ],
    ids=["default-tolerance", "tolerance-at-the-distance-to-the-middle"],
)
def test_tilt_tolerance_sets_the_family(tilt_tolerance, family) -> None:
    """Check that the tilt tolerance of the numerical settings sets the family.

    A tilt tolerance greater than the distance from the mean
    to the middle of the support makes the distribution uniform.
    """
    settings = OTMaximumEntropyDistribution_Settings(
        minimum=support_minimum,
        maximum=support_maximum,
        mu=support_middle + 0.1,
        numerical_settings=MaximumEntropyNumericalSettings(
            tilt_tolerance=tilt_tolerance
        ),
    )
    assert str(family) in repr(OTMaximumEntropyDistribution(settings))


def test_mode_is_not_supported(snapshot) -> None:
    """Check that a most likely value cannot be given, as it adds information."""
    with assert_exception(ValidationError, snapshot):
        OTMaximumEntropyDistribution_Settings(
            minimum=support_minimum, maximum=support_maximum, mode=3.0
        )


factory_cases: Final[pytest.MarkDecorator] = pytest.mark.parametrize(
    ("class_name", "settings_class"),
    [
        ("OTMaximumEntropyDistribution", OTMaximumEntropyDistribution_Settings),
        ("SPMaximumEntropyDistribution", SPMaximumEntropyDistribution_Settings),
    ],
    ids=["openturns", "scipy"],
)
"""The parametrization over the distributions created by name."""


@factory_cases
def test_distribution_is_listed(class_name, settings_class) -> None:
    """Check that the distribution is listed by the facade."""
    assert class_name in get_available_distributions()


@factory_cases
def test_create_distribution(class_name, settings_class) -> None:
    """Check that the distribution can be created by name with the facade."""
    settings = settings_class(minimum=support_minimum, maximum=support_maximum)
    distribution = create_distribution(class_name, settings=settings)
    assert distribution.mean == pytest.approx(support_middle)


@factory_cases
def test_settings_target_class_name(class_name, settings_class) -> None:
    """Check that the settings are bound to the distribution class by their name.

    This is how RandomSpace creates the distribution of a random variable.
    """
    settings = settings_class(minimum=support_minimum, maximum=support_maximum)
    assert settings.target_class_name == class_name


@factory_cases
def test_create_from_settings(class_name, settings_class) -> None:
    """Check that the distribution factory can create it from its settings."""
    settings = settings_class(minimum=support_minimum, maximum=support_maximum)
    assert distribution_factory.create_from_settings(settings).mean == pytest.approx(
        support_middle
    )


def test_base_class_is_not_available() -> None:
    """Check that the base class is not proposed as a distribution to create."""
    assert "BaseMaximumEntropyDistribution" not in get_available_distributions()


def test_random_space() -> None:
    """Check that the settings can define the random variables of a random space."""
    space = RandomSpace()
    space.add_variable(
        "x",
        OTMaximumEntropyDistribution_Settings(
            minimum=support_minimum, maximum=support_maximum
        ),
    )
    space.add_variable(
        "y",
        OTMaximumEntropyDistribution_Settings(mu=1.0, sigma=2.0),
        OTMaximumEntropyDistribution_Settings(minimum=support_minimum, mu=3.0),
    )
    assert list(space.variables) == ["x", "y"]
    assert_allclose(
        space.variables["x"].distribution.mean, [support_middle], rtol=1e-12
    )
    assert_allclose(space.variables["y"].distribution.mean, [1.0, 3.0], rtol=1e-12)
    assert_allclose(
        space.variables["x"].distribution.support,
        [[support_minimum, support_maximum]],
        rtol=1e-12,
    )
    assert space.compute_samples(3).shape == (3, 3)


def _assert_tilted_uniform_moments(
    distribution: OTMaximumEntropyDistribution | SPMaximumEntropyDistribution,
    minimum: float,
    maximum: float,
    mu: float,
) -> None:
    """Check the moments of a tilted uniform distribution.

    The mean is the required one
    and the standard deviation is the closed-form one
    of the tilted uniform distribution.

    Args:
        distribution: The distribution to check.
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable.
        mu: The required mean of the random variable.
    """
    length = maximum - minimum
    # The moments are compared in reduced coordinates,
    # so that the accuracy requirement does not depend on the support.
    assert (distribution.mean - minimum) / length == pytest.approx(
        (mu - minimum) / length, rel=1e-10
    )
    tilt = compute_tilted_uniform_tilt(
        minimum, maximum, mu, settings=numerical_settings
    )
    assert distribution.standard_deviation / length == pytest.approx(
        compute_tilted_uniform_standard_deviation(
            tilt,
            minimum,
            maximum,
            small_tilt=numerical_settings.small_tilt,
            large_tilt=numerical_settings.large_tilt,
        )
        / length,
        rel=1e-12,
    )


@pytest.mark.parametrize(
    ("minimum", "maximum", "mu"),
    [
        (
            support_minimum,
            support_maximum,
            support_middle - 1e-7 * (support_maximum - support_minimum),
        ),
        (
            support_minimum,
            support_maximum,
            support_middle - 1e-4 * (support_maximum - support_minimum),
        ),
        (1e9, 1e9 + 1.0, 1e9 + 0.25),
    ],
    ids=["nearly-zero-tilt", "small-tilt", "far-from-origin"],
)
def test_tilted_uniform_delivered_moments(classes, minimum, maximum, mu) -> None:
    """Check the delivered moments of tilted uniform distributions.

    SciPy computes the moments of a truncated exponential distribution
    with cancellation-prone formulas that collapse for a small dimensionless tilt
    and OpenTURNS integrates them in absolute coordinates,
    which loses all accuracy on a support far from the origin;
    the delivered moments must come from the closed-form expressions instead.
    """
    distribution_class, settings_class = classes
    distribution = distribution_class(
        settings_class(minimum=minimum, maximum=maximum, mu=mu)
    )
    _assert_tilted_uniform_moments(distribution, minimum, maximum, mu)


def test_tilted_uniform_increasing_density_far_from_origin() -> None:
    """Check an increasing density on a support far from the origin.

    OpenTURNS builds this distribution
    as an exponential distribution transformed by `-x`
    and truncated to the support,
    and estimates a range for the transformed distribution
    that is much narrower than the support and lies far from the origin;
    the lower end of that range is not the lower end of the support
    and the lower truncation bound must not be compared with it.
    This family is not available with SciPy,
    hence a test restricted to OpenTURNS.
    """
    minimum = 1e9
    maximum = 1e9 + 100.0
    mu = 1e9 + 99.99
    distribution = OTMaximumEntropyDistribution(
        OTMaximumEntropyDistribution_Settings(minimum=minimum, maximum=maximum, mu=mu)
    )
    assert_allclose(distribution.support, [minimum, maximum], rtol=1e-12)
    _assert_tilted_uniform_moments(distribution, minimum, maximum, mu)


@pytest.mark.parametrize(
    ("kwargs", "cdf_at_mean"),
    [
        ({"minimum": -1e9, "maximum": 2.0, "mu": 0.0, "sigma": 1.0}, 0.483),
        ({"minimum": 0.0, "maximum": 1e6, "mu": 999999.9, "sigma": 0.02}, 0.5),
    ],
    ids=["very-distant-minimum", "mean-near-the-maximum"],
)
def test_truncated_normal_extreme_supports(classes, kwargs, cdf_at_mean) -> None:
    """Check the fits that a cancellation used to corrupt silently.

    A bound farther than the weak-truncation margin is dropped by the fit
    and the reduction is reflected when the mean is nearer the maximum,
    so that the density peaks near the origin of the reduced support;
    without these,
    the delivered moments used to be wrong by tens of percents,
    without any warning.
    The value of the cumulative distribution function at the mean
    checks that the delivered sampler realizes the exact fit too.
    """
    distribution_class, settings_class = classes
    distribution = distribution_class(settings_class(**kwargs))
    assert str(MaximumEntropyFamily.TRUNCATED_NORMAL) in repr(distribution)
    assert distribution.mean == pytest.approx(kwargs["mu"], rel=1e-7)
    assert distribution.standard_deviation == pytest.approx(kwargs["sigma"], rel=1e-7)
    assert distribution.compute_cdf(kwargs["mu"]) == pytest.approx(
        cdf_at_mean, rel=1e-3
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver_tolerance": 1.0},
        {"small_tilt": 2.0},
        {"large_tilt": 5.0},
        {"tilt_tolerance": 0.5},
    ],
    ids=[
        "solver_tolerance-too-large",
        "small_tilt-too-large",
        "large_tilt-too-small",
        "tilt_tolerance-at-half",
    ],
)
def test_numerical_settings_error(kwargs, snapshot) -> None:
    """Check the errors raised by invalid numerical settings.

    The derived tolerances are meaningless for a large solver tolerance,
    the series expansions are only accurate below a small dimensionless tilt
    and the asymptotic ones only above a large one,
    and a tilt tolerance of half of the length of the support
    would move every mean to its middle.
    The bounds of the two tilts also keep the small one below the large one.
    """
    with assert_exception(ValidationError, snapshot):
        MaximumEntropyNumericalSettings(**kwargs)


def test_kept_truncated_normal_warning(caplog) -> None:
    """Check the warning when the kept truncated normal is inaccurate.

    Near the largest standard deviation,
    the library can deliver the truncated normal distribution
    with a moment error that is smaller than the substitution error
    yet larger than the solver tolerance;
    the distribution is kept and a warning states the delivered accuracy.
    """
    sigma = _compute_shrunk_maximum_sigma(0.0, None, 1.0, 1e-4)
    settings = SPMaximumEntropyDistribution_Settings(minimum=0.0, mu=1.0, sigma=sigma)
    distribution = SPMaximumEntropyDistribution(settings)
    assert str(MaximumEntropyFamily.TRUNCATED_NORMAL) in repr(distribution)
    _, parameters = compute_maximum_entropy_parameters(
        0.0, None, 1.0, sigma, settings=numerical_settings
    )
    error = distribution._compute_truncated_normal_representation_error(
        parameters, settings
    )
    assert (
        logger_name,
        logging.WARNING,
        (
            "The truncated normal distribution maximizing the entropy "
            "for this information is delivered by the library "
            f"with a moment error of about {error}."
        ),
    ) in caplog.record_tuples
