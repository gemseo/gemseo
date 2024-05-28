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

import pytest
from numpy import array
from numpy import inf
from numpy.random import RandomState
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.exponential import SPExponentialDistribution
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
from gemseo.uncertainty.distributions.scipy.triangular import SPTriangularDistribution
from gemseo.uncertainty.distributions.scipy.uniform import SPUniformDistribution


def test_joint_distribution() -> None:
    """Check the joint probability distribution associated with a SPDistribution."""
    assert SPJointDistribution == SPDistribution.JOINT_DISTRIBUTION_CLASS


def test_constructor() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 1})
    assert distribution.transformation == "x"


def test_bad_distribution() -> None:
    with pytest.raises(
        ImportError, match=re.escape("Dummy cannot be imported from scipy.stats.")
    ):
        SPDistribution("Dummy", {"loc": 0.0, "scale": 1})


def test_bad_distribution_parameters() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The arguments of norm(loc=0.0, max=1) are wrong; "
            "more details on https://docs.scipy.org/doc/scipy/reference/stats.html."
        ),
    ):
        SPDistribution("norm", {"loc": 0.0, "max": 1})


def test_str() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2.0})
    assert str(distribution) == "norm(loc=0.0, scale=2.0)"
    distribution = SPDistribution(
        "norm",
        {"loc": 0.0, "scale": 2.0},
        standard_parameters={"mean": 0, "var": 4},
    )
    assert str(distribution) == "norm(mean=0, var=4)"


def test_compute_samples() -> None:
    random_state = RandomState(0)
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    sample = distribution.compute_samples(3, random_state)
    assert sample.ndim == 1
    assert_almost_equal(sample, array([3.528105, 0.800314, 1.957476]), decimal=3)


def test_get_cdf() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution.compute_cdf(0.0) == 0.5


def test_get_inverse_cdf() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution.compute_inverse_cdf(0.5) == 0.0


def test_cdf() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution._cdf(0.0) == 0.5


def test_pdf() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution._pdf(0.0) == pytest.approx(0.19947114020071632, abs=1e-3)


def test_mean() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution.mean == 0.0


def test_std() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert distribution.standard_deviation == 2.0


def test_support() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert_equal(distribution.support, array([-inf, inf]))


def test_range() -> None:
    distribution = SPDistribution("norm", {"loc": 0.0, "scale": 2})
    assert_almost_equal(distribution.range, array([-14.068968, 14.068974]), decimal=3)


def test_normal() -> None:
    distribution = SPNormalDistribution()
    assert str(distribution) == "norm(mu=0.0, sigma=1.0)"


def test_uniform() -> None:
    distribution = SPUniformDistribution()
    assert str(distribution) == "uniform(lower=0.0, upper=1.0)"


def test_exponential() -> None:
    distribution = SPExponentialDistribution()
    assert str(distribution) == "expon(loc=0.0, scale=1.0)"


def test_triangular() -> None:
    distribution = SPTriangularDistribution()
    assert str(distribution) == "triang(lower=0.0, mode=0.5, upper=1.0)"
