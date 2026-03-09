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
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy.random import RandomState
from scipy.stats import goodness_of_fit
from scipy.stats import norm

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_fitter import (
    SPDistributionFitter,
)
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.fixture(scope="module")
def data() -> RealArray:
    """100 samples normally distributed."""
    return RandomState(1).normal(size=100)


@pytest.fixture(scope="module")
def fitter(data) -> SPDistributionFitter:
    """The distribution fitter based on SciPy."""
    return SPDistributionFitter(data)


@pytest.fixture(scope="module")
def normal(fitter) -> SPDistribution:
    """The normal distribution fitted from the samples."""
    return fitter.fit("norm")


def test_fit(normal, data) -> None:
    """Check the distribution returned by the fit() method."""
    assert isinstance(normal, SPDistribution)
    assert normal.distribution.dist.name == "norm"
    assert not isinstance(normal, SPNormalDistribution)
    parameters = norm.fit(data)
    assert normal.mean == norm.mean(*parameters)


@pytest.mark.parametrize("from_name", [False, True])
@pytest.mark.parametrize(
    ("criterion", "scipy_name"),
    [
        (SPDistributionFitter.FittingCriterion.ANDERSON_DARLING, "ad"),
        (SPDistributionFitter.FittingCriterion.CRAMER_VON_MISES, "cvm"),
        (SPDistributionFitter.FittingCriterion.FILLIBEN, "filliben"),
        (SPDistributionFitter.FittingCriterion.KOLMOGOROV_SMIRNOV, "ks"),
    ],
)
def test_compute_measure(
    fitter, data, normal, from_name, criterion, scipy_name
) -> None:
    """Check the compute_measure."""
    distribution = "norm" if from_name else normal
    acceptable, details = fitter.compute_measure(distribution, criterion)
    result = goodness_of_fit(norm, data, statistic=scipy_name, random_state=0)
    level = 0.05
    assert acceptable is (result.pvalue >= level)
    assert details["level"] == level
    assert details["p-value"] == result.pvalue
    assert details["statistics"] == result.statistic


def test_available_distributions(fitter) -> None:
    """Verify that the available_distributions property works correctly."""
    assert "norm" in fitter.available_distributions


def test_available_criteria(fitter) -> None:
    """Verify that the available_criteria property works correctly."""
    assert fitter.available_criteria == [
        "AndersonDarling",
        "CramerVonMises",
        "Filliben",
        "KolmogorovSmirnov",
    ]


def test_available_significance_tests(fitter) -> None:
    """Verify that the available_significance_tests property works correctly."""
    assert fitter.available_significance_tests == [
        "AndersonDarling",
        "CramerVonMises",
        "Filliben",
        "KolmogorovSmirnov",
    ]
