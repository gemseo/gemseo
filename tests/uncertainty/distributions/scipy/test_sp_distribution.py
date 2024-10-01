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
import scipy.stats as scipy_stats
from numpy import array
from numpy import exp
from numpy import inf
from numpy.random import RandomState
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.scipy.beta import SPBetaDistribution
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.exponential import SPExponentialDistribution
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
from gemseo.uncertainty.distributions.scipy.log_normal import SPLogNormalDistribution
from gemseo.uncertainty.distributions.scipy.normal import SPNormalDistribution
from gemseo.uncertainty.distributions.scipy.triangular import SPTriangularDistribution
from gemseo.uncertainty.distributions.scipy.uniform import SPUniformDistribution
from gemseo.uncertainty.distributions.scipy.weibull import SPWeibullDistribution


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


MU_L_0, SIGMA_L_0 = compute_mu_l_and_sigma_l(1.0, 1.0, 0.0)
MU_L_1, SIGMA_L_1 = compute_mu_l_and_sigma_l(1.2, 1.3, 0.4)


@pytest.mark.parametrize(
    ("sp_dist_name", "cls", "sp_kwargs", "kwargs", "str_"),
    [
        (
            "beta",
            SPBetaDistribution,
            {"a": 2.0, "b": 2.0, "loc": 0.0, "scale": 1.0},
            {},
            "beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)",
        ),
        (
            "beta",
            SPBetaDistribution,
            {"a": 0.1, "b": 0.2, "loc": 0.3, "scale": 0.1},
            {"alpha": 0.1, "beta": 0.2, "minimum": 0.3, "maximum": 0.4},
            "beta(lower=0.3, upper=0.4, alpha=0.1, beta=0.2)",
        ),
        (
            "expon",
            SPExponentialDistribution,
            {"scale": 1.0, "loc": 0.0},
            {},
            "expon(rate=1.0, loc=0.0)",
        ),
        (
            "expon",
            SPExponentialDistribution,
            {"scale": 10.0, "loc": 0.2},
            {"rate": 0.1, "loc": 0.2},
            "expon(rate=0.1, loc=0.2)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution,
            {"s": SIGMA_L_0, "scale": exp(MU_L_0), "loc": 0.0},
            {},
            "lognorm(mu=1.0, sigma=1.0, loc=0.0)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution,
            {"s": 2.5, "scale": exp(1.5), "loc": 0.0},
            {"mu": 1.5, "sigma": 2.5, "set_log": True},
            "lognorm(mu=1.5, sigma=2.5, loc=0.0)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution,
            {"s": SIGMA_L_1, "scale": exp(MU_L_1), "loc": 0.4},
            {"mu": 1.2, "sigma": 1.3, "location": 0.4},
            "lognorm(mu=1.2, sigma=1.3, loc=0.4)",
        ),
        (
            "norm",
            SPNormalDistribution,
            {"loc": 0.0, "scale": 1.0},
            {},
            "norm(mu=0.0, sigma=1.0)",
        ),
        (
            "norm",
            SPNormalDistribution,
            {"loc": 0.1, "scale": 0.2},
            {"mu": 0.1, "sigma": 0.2},
            "norm(mu=0.1, sigma=0.2)",
        ),
        (
            "triang",
            SPTriangularDistribution,
            {"loc": 0.0, "scale": 1.0, "c": 0.5},
            {},
            "triang(lower=0.0, mode=0.5, upper=1.0)",
        ),
        (
            "triang",
            SPTriangularDistribution,
            {"loc": 0.1, "scale": 0.2, "c": 0.5},
            {"minimum": 0.1, "mode": 0.2, "maximum": 0.3},
            "triang(lower=0.1, mode=0.2, upper=0.3)",
        ),
        (
            "uniform",
            SPUniformDistribution,
            {"loc": 0.0, "scale": 1.0},
            {},
            "uniform(lower=0.0, upper=1.0)",
        ),
        (
            "uniform",
            SPUniformDistribution,
            {"loc": 0.1, "scale": 0.1},
            {"minimum": 0.1, "maximum": 0.2},
            "uniform(lower=0.1, upper=0.2)",
        ),
        (
            "weibull_min",
            SPWeibullDistribution,
            {"loc": 0.0, "scale": 1.0, "c": 1.0},
            {},
            "weibull_min(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            "weibull_min",
            SPWeibullDistribution,
            {"loc": 0.1, "scale": 0.2, "c": 0.3},
            {"location": 0.1, "scale": 0.2, "shape": 0.3},
            "weibull_min(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            "weibull_max",
            SPWeibullDistribution,
            {"loc": 0.0, "scale": 1.0, "c": 1.0},
            {"use_weibull_min": False},
            "weibull_max(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            "weibull_max",
            SPWeibullDistribution,
            {"loc": 0.1, "scale": 0.2, "c": 0.3},
            {"use_weibull_min": False, "location": 0.1, "scale": 0.2, "shape": 0.3},
            "weibull_max(location=0.1, scale=0.2, shape=0.3)",
        ),
    ],
)
def test_specific_sp_distributions(sp_dist_name, cls, sp_kwargs, kwargs, str_) -> None:
    """Check the specific SciPy-based distributions.

    Args:
        sp_dist_name: The name of the SciPy class.
        cls: The class deriving from SPDistribution.
        sp_kwargs: The keyword arguments to instantiate the SciPy class.
        kwargs: The keyword arguments to instantiate ``cls``.
        str_: The expected string representation of the ``cls`` instance.
    """
    ot_distribution = cls(**kwargs)
    distribution = ot_distribution.distribution
    expected_distribution = getattr(scipy_stats, sp_dist_name)(**sp_kwargs)
    assert distribution.dist.name == expected_distribution.dist.name
    assert distribution.mean() == pytest.approx(expected_distribution.mean())
    assert str(ot_distribution) == str_


def test_lognormal_distribution():
    """Check that (mu,sigma)=(1,1) for the default log-normal distribution.

    We do this test because we transform the (mu, sigma) into (mu_l, sigma_l) and
    instantiate the wrapped distribution with it. This is a way to test this
    transformation.
    """
    distribution = SPLogNormalDistribution()
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.standard_deviation == pytest.approx(1.0)
