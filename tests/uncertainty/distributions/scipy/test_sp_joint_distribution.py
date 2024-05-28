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
from numpy import allclose
from numpy import array
from numpy import inf

from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution


@pytest.fixture(scope="module")
def distributions() -> list[SPNormalDistribution]:
    """Two normal distributions."""
    return [SPNormalDistribution(), SPNormalDistribution()]


@pytest.fixture(scope="module")
def joint_distribution(
    distributions: Sequence[SPDistribution],
) -> SPJointDistribution:
    """The joint probability distribution."""
    return SPJointDistribution(distributions)


def test_constructor(joint_distribution) -> None:
    assert joint_distribution.transformation == "x"


def test_str(joint_distribution) -> None:
    """Check the string representation of the joint probability distribution."""
    assert (
        repr(joint_distribution)
        == str(joint_distribution)
        == (
            "SPJointDistribution("
            "norm(mu=0.0, sigma=1.0), "
            "norm(mu=0.0, sigma=1.0); "
            "IndependentCopula"
            ")"
        )
    )


def test_copula(distributions) -> None:
    """Check that using a copula which is not None raises an error."""
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            "There is not copula distribution yet for SciPy-based distributions."
        ),
    ):
        SPJointDistribution(distributions, copula="foo")


def test_compute_samples(joint_distribution) -> None:
    assert joint_distribution.compute_samples(3).shape == (3, 2)


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
    expectation = array([-7.034484, 7.034487])
    for element in joint_distribution.range:
        assert allclose(element, expectation, 1e-3)
