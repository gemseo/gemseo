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
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest
from numpy.random import RandomState
from numpy.testing import assert_allclose
from openturns import Normal

from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_fitter import (
    OTDistributionFitter,
)
from gemseo.uncertainty.distributions.openturns.fitting import (
    OTDistributionFitter as OldOTDistributionFitter,
)
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def data() -> RealArray:
    """100 samples normally distributed."""
    return RandomState(1).normal(size=100)


@pytest.fixture(scope="module")
def fitter(data) -> OTDistributionFitter:
    """The distribution fitter based on OpenTURNS."""
    return OTDistributionFitter("x", data)


@pytest.fixture(scope="module")
def normal(fitter) -> OTDistribution:
    """The normal distribution fitted from the samples."""
    return fitter.fit("Normal")


def test_alias():
    """Check that the deprecated import."""
    assert OldOTDistributionFitter is OTDistributionFitter


def test_data(fitter, data):
    """Check that the data attribute stores the samples."""
    assert id(fitter.data) == id(data)


def test_distribution_dimension_error(fitter) -> None:
    """Check the error raised when using a distribution of dimension higher than 1."""
    ot_joint_distribution = OTJointDistribution([OTNormalDistribution()] * 2)
    with pytest.raises(
        KeyError,
        match=re.escape(
            "OTJointDistribution(Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0); "
            "IndependentCopula)"
        ),
    ):
        fitter.compute_measure(ot_joint_distribution, "BIC")


def test_fit(normal) -> None:
    """Check the type of the distribution returned by the fit() method."""
    assert isinstance(normal, OTDistribution)
    assert isinstance(normal.distribution, Normal)
    assert not isinstance(normal, OTNormalDistribution)


@pytest.mark.parametrize("from_name", [False, True])
def test_bic(fitter, normal, from_name) -> None:
    """Check the BIC value."""
    distribution = "Normal" if from_name else normal
    quality_measure = fitter.compute_measure(distribution, "BIC")
    assert_allclose(quality_measure, 2.59394512877)


@pytest.mark.parametrize("from_name", [False, True])
def test_kolmogorov(fitter, normal, from_name) -> None:
    """Check the Kolmogorov test results."""
    distribution = "Normal" if from_name else normal
    acceptable, details = fitter.compute_measure(distribution, "Kolmogorov")
    assert acceptable
    assert details["level"] == 0.05
    assert_allclose(details["statistics"], 0.04330972976650932)
    assert_allclose(details["p-value"], 0.9879299613543082)


def test_warning(fitter, caplog) -> None:
    """Check the warning message logged when all criteria are lower than alpha."""
    measures = [
        fitter.compute_measure("Uniform", "Kolmogorov"),
        fitter.compute_measure("Triangular", "Kolmogorov"),
    ]
    fitter.select_from_measures(measures, "Kolmogorov")
    assert not caplog.record_tuples

    fitter.select_from_measures(measures, "Kolmogorov", level=0.99)
    assert caplog.record_tuples[-1] == (
        "gemseo.uncertainty.distributions.base_distribution_fitter",
        30,
        "All criteria values are lower than the significance level 0.99.",
    )

    measures = [
        fitter.compute_measure("Uniform", "Kolmogorov"),
        fitter.compute_measure("Uniform", "Kolmogorov"),
    ]
    fitter.select_from_measures(measures, "Kolmogorov")
    assert caplog.record_tuples[-1] == (
        "gemseo.uncertainty.distributions.base_distribution_fitter",
        30,
        "All criteria values are lower than the significance level 0.05.",
    )


@pytest.mark.parametrize(
    ("fitting_criterion", "selection_criterion", "level"),
    [
        ("BIC", "best", 0.05),
        ("BIC", "first", 0.05),
        ("Kolmogorov", "best", 0.05),
        ("Kolmogorov", "first", 0.05),
        ("Kolmogorov", "first", 0.99),
    ],
)
def test_select_from_measures_index(
    fitter, fitting_criterion, selection_criterion, level
) -> None:
    """Check the index returned by select_from_measures."""
    measures = [
        fitter.compute_measure("Uniform", fitting_criterion),
        fitter.compute_measure("Triangular", fitting_criterion),
    ]
    index = fitter.select_from_measures(
        measures,
        fitting_criterion,
        selection_criterion=selection_criterion,
        level=level,
    )
    assert index == 1


def test_available_criteria(fitter) -> None:
    """Verify that the available_criteria property works correctly."""
    assert fitter.available_criteria == ["BIC", "ChiSquared", "Kolmogorov"]


def test_available_significance_tests(fitter) -> None:
    """Verify that the available_significance_tests property works correctly."""
    assert fitter.available_significance_tests == ["ChiSquared", "Kolmogorov"]


def test_available_distributions(fitter) -> None:
    """Verify that the available_distributions property works correctly."""
    assert "Normal" in fitter.available_distributions


@pytest.mark.parametrize("first_distribution", ["Normal", ""])
@pytest.mark.parametrize("second_distribution", ["Exponential", ""])
def test_select(fitter, first_distribution, second_distribution) -> None:
    """Check the type of the distribution returned by the select() method."""
    first_distribution = first_distribution or fitter.fit("Normal")
    second_distribution = second_distribution or fitter.fit("Exponential")
    distribution = fitter.select([first_distribution, second_distribution], "BIC")
    assert isinstance(distribution, OTDistribution)
    assert isinstance(distribution.distribution, Normal)
    assert not isinstance(distribution, OTNormalDistribution)
