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
from pathlib import Path
from unittest.mock import patch

import pytest
from numpy import array
from numpy import inf
from numpy import int32
from numpy import ndarray
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from openturns import Beta
from openturns import Dirac
from openturns import Exponential
from openturns import LogNormal
from openturns import Normal
from openturns import RandomGenerator
from openturns import Triangular
from openturns import Uniform
from openturns import WeibullMax
from openturns import WeibullMin

from gemseo.uncertainty.distributions import scalar_distribution_mixin
from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.openturns.beta import OTBetaDistribution
from gemseo.uncertainty.distributions.openturns.dirac import OTDiracDistribution
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.exponential import (
    OTExponentialDistribution,
)
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.log_normal import (
    OTLogNormalDistribution,
)
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distributions.openturns.triangular import (
    OTTriangularDistribution,
)
from gemseo.uncertainty.distributions.openturns.uniform import OTUniformDistribution
from gemseo.uncertainty.distributions.openturns.weibull import OTWeibullDistribution


def test_joint_distribution() -> None:
    """Check the joint distribution associated with a OTDistribution."""
    assert OTJointDistribution == OTDistribution.JOINT_DISTRIBUTION_CLASS


def test_constructor() -> None:
    distribution = OTDistribution("Normal", (0, 1))
    assert distribution.transformation == "x"


def test_bad_distribution() -> None:
    with pytest.raises(
        ImportError, match=re.escape("Dummy cannot be imported from openturns.")
    ):
        OTDistribution("Dummy", (0, 1))


def test_bad_distribution_parameters() -> None:
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The arguments of Normal(0, 1, 2) are wrong; "
            "more details on "
            "http://openturns.github.io/openturns/latest/user_manual/probabilistic_modelling.html."  # noqa: E501
        ),
    ):
        OTDistribution("Normal", (0, 1, 2))


def test_str() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert str(distribution) == "Normal(0, 2)"
    distribution = OTDistribution(
        "Normal", (0, 2), standard_parameters={"mean": 0, "var": 4}
    )
    assert str(distribution) == "Normal(mean=0, var=4)"


def test_compute_samples() -> None:
    RandomGenerator.SetSeed(0)
    distribution = OTDistribution("Normal", (0, 2))
    sample = distribution.compute_samples(3)
    assert isinstance(sample, ndarray)
    assert sample.ndim == 1
    assert_almost_equal(sample, array([1.216403, -2.532346, -0.876531]), decimal=3)


def test_get_cdf() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert distribution.compute_cdf(0.0) == 0.5


def test_get_inverse_cdf() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert distribution.compute_inverse_cdf(0.5) == 0.0


def test_cdf() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert distribution._cdf(0.0) == 0.5


def test_pdf() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert distribution._pdf(0.0) == pytest.approx(0.19947114020071632, abs=1e-3)


def test_mean() -> None:
    assert OTDistribution("Normal", (0, 2)).mean == 0.0


def test_std() -> None:
    assert OTDistribution("Normal", (0, 2)).standard_deviation == 2.0


def test_support() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert_equal(distribution.support, array([-inf, inf]))


def test_range() -> None:
    distribution = OTDistribution("Normal", (0, 2))
    assert_almost_equal(distribution.range, array([-15.301256, 15.301256]), decimal=3)
    distribution = OTDistribution("Uniform", (0, 1))
    assert_equal(distribution.range, array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"lower_bound": 0.5, "upper_bound": 0.6}, array([0.5, 0.6])),
        ({"lower_bound": 0.5}, array([0.5, 2.0])),
        ({"upper_bound": 0.5}, array([0.0, 0.5])),
    ],
)
def test_truncation(kwargs, expected) -> None:
    """Check the support after truncation."""
    distribution = OTDistribution("Uniform", (0.0, 2.0), **kwargs)
    assert_equal(distribution.support, expected)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"upper_bound": 1.5}, "u_b is greater than the current upper bound."),
        ({"lower_bound": -0.5}, "l_b is less than the current lower bound."),
        (
            {"lower_bound": -0.5, "upper_bound": 1.0},
            "l_b is less than the current lower bound.",
        ),
        (
            {"lower_bound": 0.0, "upper_bound": 1.5},
            "u_b is greater than the current upper bound.",
        ),
    ],
)
def test_truncate_exception(kwargs, expected) -> None:
    """Check that exceptions are raised when mistruncating a distribution."""
    with pytest.raises(ValueError, match=re.escape(expected)):
        OTDistribution("Uniform", (0, 1), **kwargs)


def test_transformation() -> None:
    distribution = OTDistribution("Uniform", (0, 2), transformation="2*x")
    assert_almost_equal(distribution.support, array([0.0, 4.0]), decimal=5)


@pytest.mark.parametrize(
    (
        "dimension",
        "file_path",
        "directory_path",
        "file_name",
        "file_extension",
        "expected",
    ),
    [
        (1, None, None, None, None, "distribution.png"),
        (2, None, None, None, None, "distribution.png"),
        (2, "foo/bar.png", None, None, None, Path("foo/bar.png")),
        (2, None, "foo", None, None, Path("foo/distribution.png")),
        (2, None, "foo", "bar", "svg", Path("foo/bar.svg")),
    ],
)
def test_plot_save(
    dimension,
    file_path,
    directory_path,
    file_name,
    file_extension,
    expected,
    tmp_wd,
) -> None:
    """Check the file path computed by plot()."""
    triangular = OTTriangularDistribution()
    with patch.object(scalar_distribution_mixin, "save_show_figure") as mock_method:
        triangular.plot(
            show=False,
            save=True,
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_extension,
        )

        args = mock_method.call_args.args
        if isinstance(expected, Path):
            assert args[2] == expected
        else:
            assert args[2] == Path(tmp_wd / expected)


def test_compute_cdf_int32():
    """Check that openturns-based compute_cdf handles numpy.int32."""
    expected = pytest.approx(0.8, abs=0.1)
    x = int32(1)
    assert OTNormalDistribution().compute_cdf(x) == expected
    assert OTJointDistribution([OTNormalDistribution()]).compute_cdf([x]) == expected


@pytest.mark.parametrize(
    ("ot_cls", "cls", "ot_args", "kwargs", "str_"),
    [
        (
            Beta,
            OTBetaDistribution,
            (2.0, 2.0, 0.0, 1.0),
            {},
            "Beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)",
        ),
        (
            Beta,
            OTBetaDistribution,
            (0.1, 0.2, 0.3, 0.4),
            {"alpha": 0.1, "beta": 0.2, "minimum": 0.3, "maximum": 0.4},
            "Beta(lower=0.3, upper=0.4, alpha=0.1, beta=0.2)",
        ),
        (
            Dirac,
            OTDiracDistribution,
            (),
            {},
            "Dirac(loc=0.0)",
        ),
        (Dirac, OTDiracDistribution, (1.0,), {"variable_value": 1.0}, "Dirac(loc=1.0)"),
        (
            Exponential,
            OTExponentialDistribution,
            (),
            {},
            "Exponential(rate=1.0, loc=0.0)",
        ),
        (
            Exponential,
            OTExponentialDistribution,
            (0.1, 0.2),
            {"rate": 0.1, "loc": 0.2},
            "Exponential(rate=0.1, loc=0.2)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution,
            (*compute_mu_l_and_sigma_l(1.0, 1.0, 0.0), 0.0),
            {},
            "LogNormal(mu=1.0, sigma=1.0, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution,
            (1.5, 2.5, 0.0),
            {"mu": 1.5, "sigma": 2.5, "set_log": True},
            "LogNormal(mu=1.5, sigma=2.5, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution,
            (*compute_mu_l_and_sigma_l(1.2, 1.3, 0.4), 0.4),
            {"mu": 1.2, "sigma": 1.3, "location": 0.4},
            "LogNormal(mu=1.2, sigma=1.3, loc=0.4)",
        ),
        (Normal, OTNormalDistribution, (), {}, "Normal(mu=0.0, sigma=1.0)"),
        (
            Normal,
            OTNormalDistribution,
            (0.1, 0.2),
            {"mu": 0.1, "sigma": 0.2},
            "Normal(mu=0.1, sigma=0.2)",
        ),
        (
            Triangular,
            OTTriangularDistribution,
            (0.0, 0.5, 1.0),
            {},
            "Triangular(lower=0.0, mode=0.5, upper=1.0)",
        ),
        (
            Triangular,
            OTTriangularDistribution,
            (0.1, 0.2, 0.3),
            {"minimum": 0.1, "mode": 0.2, "maximum": 0.3},
            "Triangular(lower=0.1, mode=0.2, upper=0.3)",
        ),
        (
            Uniform,
            OTUniformDistribution,
            (0.0, 1.0),
            {},
            "Uniform(lower=0.0, upper=1.0)",
        ),
        (
            Uniform,
            OTUniformDistribution,
            (0.1, 0.2),
            {"minimum": 0.1, "maximum": 0.2},
            "Uniform(lower=0.1, upper=0.2)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution,
            (1.0, 1.0, 0.0),
            {},
            "WeibullMin(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution,
            (0.2, 0.3, 0.1),
            {"location": 0.1, "scale": 0.2, "shape": 0.3},
            "WeibullMin(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution,
            (1.0, 1.0, 0.0),
            {"use_weibull_min": False},
            "WeibullMax(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution,
            (0.2, 0.3, 0.1),
            {"use_weibull_min": False, "location": 0.1, "scale": 0.2, "shape": 0.3},
            "WeibullMax(location=0.1, scale=0.2, shape=0.3)",
        ),
    ],
)
def test_specific_ot_distributions(ot_cls, cls, ot_args, kwargs, str_) -> None:
    """Check the specific OpenTURNS-based distributions.

    Args:
        ot_cls: The OpenTURNS class.
        cls: The class deriving from OTDistribution.
        ot_args: The positional arguments to instantiate the OpenTURNS class.
        kwargs: The keyword arguments to instantiate ``cls``.
        str_: The expected string representation of the ``cls`` instance.
    """
    ot_distribution = cls(**kwargs)
    distribution = ot_distribution.distribution
    expected_distribution = ot_cls(*ot_args)
    assert distribution.getName() == expected_distribution.getName()
    assert distribution.getParameter() == expected_distribution.getParameter()
    assert str(ot_distribution) == str_


def test_lognormal_distribution():
    """Check that (mu,sigma)=(1,1) for the default log-normal distribution.

    We do this test because we transform the (mu, sigma) into (mu_l, sigma_l) and
    instantiate the wrapped distribution with it. This is a way to test this
    transformation.
    """
    distribution = OTLogNormalDistribution()
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.standard_deviation == pytest.approx(1.0)
