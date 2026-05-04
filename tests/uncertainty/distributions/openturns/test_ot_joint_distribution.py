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

import pytest
from numpy import allclose
from numpy import array
from numpy import inf
from numpy import int_
from openturns import NormalCopula

from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def joint_distribution() -> OTJointDistribution:
    """A joint probability distribution."""
    return OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=[
                OTNormalDistribution_Settings(),
                OTNormalDistribution_Settings(),
            ]
        )
    )


@pytest.mark.parametrize(
    ("n_marginals", "expected"),
    [
        (1, "Normal(mu=0.0, sigma=1.0)"),
        (
            2,
            (
                "OTJointDistribution("
                "Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0); "
                "IndependentCopula(dimension = 2))"
            ),
        ),
    ],
)
def test_repr(joint_distribution, n_marginals, expected) -> None:
    """Check the string representation of a joint probability distribution."""
    assert (
        repr(
            OTJointDistribution(
                OTJointDistribution_Settings(
                    marginal_settings=[OTNormalDistribution_Settings()] * n_marginals
                )
            )
        )
        == expected
    )


def test_str(joint_distribution) -> None:
    """Check the string representation of the joint probability distribution."""
    assert (
        repr(joint_distribution)
        == str(joint_distribution)
        == (
            "OTJointDistribution("
            "Normal(mu=0.0, sigma=1.0), "
            "Normal(mu=0.0, sigma=1.0); "
            "IndependentCopula(dimension = 2)"
            ")"
        )
    )


@pytest.mark.parametrize("n_samples", [3, int_(3)])
def test_get_sample(joint_distribution, n_samples) -> None:
    sample = joint_distribution.compute_samples(n_samples)
    assert sample.shape == (3, 2)


def test_get_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_cdf(array([0.0, 0.0]))
    assert allclose(result, array([0.5, 0.5]))


def test_get_inverse_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_inverse_cdf(array([0.5, 0.5]))
    assert allclose(result, array([0.0, 0.0]))


def test_mean(joint_distribution) -> None:
    assert allclose(joint_distribution.mean, array([0.0, 0.0]))


def test_std(joint_distribution) -> None:
    assert allclose(joint_distribution.standard_deviation, array([1.0, 1.0]))


def test_support(joint_distribution) -> None:
    expectation = array([-inf, inf])
    for element in joint_distribution.support:
        assert allclose(element, expectation)


def test_range(joint_distribution) -> None:
    expectation = array([-7.650628, 7.650628])
    for element in joint_distribution.range:
        assert allclose(element, expectation, 1e-3)


def test_copula() -> None:
    """Check the use of an OpenTURNS copula."""
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()] * 2,
            copula=NormalCopula(2),
        )
    )
    assert repr(distribution) == (
        "OTJointDistribution("
        "Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0); "
        "NormalCopula(R = [[ 1 0 ]\n [ 0 1 ]]))"
    )
    assert distribution.distribution.getCopula().getName() == "NormalCopula"


def test_copula_dimension_error(snapshot):
    """Check that an error is raised
    when the copula dimension does not match the number of marginals."""
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()],
            copula=NormalCopula(2),
        )


def test_copula_dimension_error_if_sub_copulas(snapshot):
    """Check that an error is raised
    when the copula dimension does not match the number of marginals
    and the copula uses block-independent copulas."""
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()],
            copula=(((0, 1), NormalCopula(2)),),
        )


def test_sub_copula_dimension_error(snapshot):
    """Check that an error is raised
    when the dimension of a block-independent copula does not match
    the number of marginals."""
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()],
            copula=(((0,), NormalCopula(2)),),
        )


def test_copula_indices_error(snapshot):
    """Check that an error is raised
    when a component associated with a block-independent copula is out of range."""
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()] * 2,
            copula=(((0, 2), NormalCopula(2)),),
        )


def test_duplication_error(snapshot):
    """Check the error raised when adding two copulas to the same variable."""
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()] * 2,
            copula=(((0,), NormalCopula(1)), ((0,), NormalCopula(1))),
        )
