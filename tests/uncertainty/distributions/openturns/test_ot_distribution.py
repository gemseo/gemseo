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

from pathlib import Path
from unittest.mock import patch

import pytest
from numpy import array
from numpy import inf
from numpy import int32
from numpy import int_
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

from gemseo.uncertainty.distributions import base_univariate
from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.factory import DISTRIBUTION_FACTORY
from gemseo.uncertainty.distributions.openturns.beta_settings import (
    OTBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.dirac_settings import (
    OTDiracDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.exponential_settings import (
    OTExponentialDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.log_normal import (
    OTLogNormalDistribution,
)
from gemseo.uncertainty.distributions.openturns.log_normal_settings import (
    OTLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distributions.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.triangular import (
    OTTriangularDistribution,
)
from gemseo.uncertainty.distributions.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distributions.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def distribution() -> OTDistribution:
    """Normal distribution with mean 0 and std 2."""
    return OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Normal", parameters=(0, 2))
    )


def test_joint_distribution() -> None:
    """Check the joint distribution associated with a OTDistribution."""
    assert OTJointDistribution == OTDistribution.JOINT_DISTRIBUTION_CLASS


def test_constructor(distribution) -> None:
    assert distribution.transformation == "x"


def test_bad_distribution(snapshot) -> None:
    with assert_exception(ImportError, snapshot):
        OTDistribution(
            OTDistribution_Settings(interfaced_distribution="Dummy", parameters=(0, 1))
        )


def test_bad_distribution_parameters(snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        OTDistribution(
            OTDistribution_Settings(
                interfaced_distribution="Normal", parameters=(0, 1, 2)
            )
        )


def test_str() -> None:
    distribution = OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Normal", parameters=(0, 2))
    )
    assert str(distribution) == "Normal(0.0, 2.0)"
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Normal",
            parameters=(0, 2),
            standard_parameters={"mean": 0, "var": 4},
        )
    )
    assert str(distribution) == "Normal(mean=0, var=4)"


@pytest.mark.parametrize("n_samples", [3, int_(3)])
def test_compute_samples(distribution, n_samples) -> None:
    RandomGenerator.SetSeed(0)
    sample = distribution.compute_samples(n_samples)
    assert isinstance(sample, ndarray)
    assert sample.ndim == 1
    assert_almost_equal(sample, array([1.216403, -2.532346, -0.876531]), decimal=3)


def test_get_cdf(distribution) -> None:
    assert distribution.compute_cdf(0.0) == 0.5


def test_get_inverse_cdf(distribution) -> None:
    assert distribution.compute_inverse_cdf(0.5) == 0.0


def test_cdf(distribution) -> None:
    assert distribution._cdf(0.0) == 0.5


def test_pdf(distribution) -> None:
    assert distribution._pdf(0.0) == pytest.approx(0.19947114020071632, abs=1e-3)


def test_mean(distribution) -> None:
    assert distribution.mean == 0.0


def test_std(distribution) -> None:
    assert distribution.standard_deviation == 2.0


def test_support(distribution) -> None:
    assert_equal(distribution.support, array([-inf, inf]))


def test_range(distribution) -> None:
    assert_almost_equal(distribution.range, array([-15.301256, 15.301256]), decimal=3)
    distribution = OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Uniform", parameters=(0, 1))
    )
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
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Uniform", parameters=(0.0, 2.0), **kwargs
        )
    )
    assert_equal(distribution.support, expected)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"upper_bound": 1.5}, "upper_bound is greater than the current upper bound."),
        ({"lower_bound": -0.5}, "lower_bound is less than the current lower bound."),
        (
            {"lower_bound": -0.5, "upper_bound": 1.0},
            "lower_bound is less than the current lower bound.",
        ),
        (
            {"lower_bound": 0.0, "upper_bound": 1.5},
            "upper_bound is greater than the current upper bound.",
        ),
    ],
)
def test_truncate_exception(kwargs, expected, snapshot) -> None:
    """Check that exceptions are raised when mistruncating a distribution."""
    with assert_exception(ValueError, snapshot):
        OTDistribution(
            OTDistribution_Settings(
                interfaced_distribution="Uniform", parameters=(0, 1), **kwargs
            )
        )


def test_transformation() -> None:
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Uniform", parameters=(0, 2), transformation="2*x"
        )
    )
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
        (1, "", "", "", "", "pdf_cdf.png"),
        (2, "", "", "", "", "pdf_cdf.png"),
        (2, "foo/bar.png", "", "", "", Path("foo/bar.png")),
        (2, "", "foo", "", "", Path("foo/pdf_cdf.png")),
        (2, "", "foo", "bar", "svg", Path("foo/bar.svg")),
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
    with patch.object(base_univariate, "save_show_figure") as mock_method:
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
    assert (
        OTJointDistribution(
            OTJointDistribution_Settings(
                marginal_settings=[OTNormalDistribution_Settings()]
            )
        ).compute_cdf([x])
        == expected
    )


@pytest.mark.parametrize(
    ("ot_cls", "settings", "ot_args", "str_"),
    [
        (
            Beta,
            OTBetaDistribution_Settings(),
            (2.0, 2.0, 0.0, 1.0),
            "Beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)",
        ),
        (
            Beta,
            OTBetaDistribution_Settings(alpha=0.1, beta=0.2, minimum=0.3, maximum=0.4),
            (0.1, 0.2, 0.3, 0.4),
            "Beta(lower=0.3, upper=0.4, alpha=0.1, beta=0.2)",
        ),
        (
            Dirac,
            OTDiracDistribution_Settings(),
            (),
            "Dirac(loc=0.0)",
        ),
        (
            Dirac,
            OTDiracDistribution_Settings(variable_value=1.0),
            (1.0,),
            "Dirac(loc=1.0)",
        ),
        (
            Exponential,
            OTExponentialDistribution_Settings(),
            (),
            "Exponential(rate=1.0, loc=0.0)",
        ),
        (
            Exponential,
            OTExponentialDistribution_Settings(rate=0.1, loc=0.2),
            (0.1, 0.2),
            "Exponential(rate=0.1, loc=0.2)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(),
            (*compute_mu_l_and_sigma_l(1.0, 1.0, 0.0), 0.0),
            "LogNormal(mu=1.0, sigma=1.0, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(mu=1.5, sigma=2.5, set_log=True),
            (1.5, 2.5, 0.0),
            "LogNormal(mu=1.5, sigma=2.5, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(mu=1.2, sigma=1.3, location=0.4),
            (*compute_mu_l_and_sigma_l(1.2, 1.3, 0.4), 0.4),
            "LogNormal(mu=1.2, sigma=1.3, loc=0.4)",
        ),
        (Normal, OTNormalDistribution_Settings(), (), "Normal(mu=0.0, sigma=1.0)"),
        (
            Normal,
            OTNormalDistribution_Settings(mu=0.1, sigma=0.2),
            (0.1, 0.2),
            "Normal(mu=0.1, sigma=0.2)",
        ),
        (
            Triangular,
            OTTriangularDistribution_Settings(),
            (0.0, 0.5, 1.0),
            "Triangular(lower=0.0, mode=0.5, upper=1.0)",
        ),
        (
            Triangular,
            OTTriangularDistribution_Settings(minimum=0.1, mode=0.2, maximum=0.3),
            (0.1, 0.2, 0.3),
            "Triangular(lower=0.1, mode=0.2, upper=0.3)",
        ),
        (
            Uniform,
            OTUniformDistribution_Settings(),
            (0.0, 1.0),
            "Uniform(lower=0.0, upper=1.0)",
        ),
        (
            Uniform,
            OTUniformDistribution_Settings(minimum=0.1, maximum=0.2),
            (0.1, 0.2),
            "Uniform(lower=0.1, upper=0.2)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution_Settings(),
            (1.0, 1.0, 0.0),
            "WeibullMin(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution_Settings(location=0.1, scale=0.2, shape=0.3),
            (0.2, 0.3, 0.1),
            "WeibullMin(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution_Settings(use_weibull_min=False),
            (1.0, 1.0, 0.0),
            "WeibullMax(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution_Settings(
                use_weibull_min=False, location=0.1, scale=0.2, shape=0.3
            ),
            (0.2, 0.3, 0.1),
            "WeibullMax(location=0.1, scale=0.2, shape=0.3)",
        ),
    ],
)
def test_specific_ot_distributions(ot_cls, settings, ot_args, str_) -> None:
    """Check the specific OpenTURNS-based distributions.

    Args:
        ot_cls: The OpenTURNS class.
        settings: The settings of the OT distribution.
        ot_args: The positional arguments to instantiate the OpenTURNS class.
        str_: The expected string representation of the distribution.
    """
    ot_distribution = DISTRIBUTION_FACTORY.create_from_settings(settings)
    distribution = ot_distribution.distribution
    expected_distribution = ot_cls(*ot_args)
    assert distribution.getName() == expected_distribution.getName()
    assert distribution.getParameter() == expected_distribution.getParameter()
    assert str(ot_distribution) == str_


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("OTBetaDistribution", "Beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)"),
        ("OTDiracDistribution", "Dirac(loc=0.0)"),
        ("OTDistribution", "Uniform()"),
        ("OTExponentialDistribution", "Exponential(rate=1.0, loc=0.0)"),
        ("OTLogNormalDistribution", "LogNormal(mu=1.0, sigma=1.0, loc=0.0)"),
        ("OTNormalDistribution", "Normal(mu=0.0, sigma=1.0)"),
        ("OTTriangularDistribution", "Triangular(lower=0.0, mode=0.5, upper=1.0)"),
        ("OTUniformDistribution", "Uniform(lower=0.0, upper=1.0)"),
        ("OTWeibullDistribution", "WeibullMin(location=0.0, scale=1.0, shape=1.0)"),
    ],
)
def test_specific_ot_distributions_default(class_name, expected):
    """Check the OpenTURNS-based distributions using default settings."""
    distribution = DISTRIBUTION_FACTORY.create(class_name)
    assert str(distribution) == expected


def test_lognormal_distribution():
    """Check that (mu,sigma)=(1,1) for the default log-normal distribution.

    We do this test because we transform the (mu, sigma) into (mu_l, sigma_l) and
    instantiate the wrapped distribution with it. This is a way to test this
    transformation.
    """
    distribution = OTLogNormalDistribution()
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.standard_deviation == pytest.approx(1.0)


def test_settings_truncation_error(snapshot):
    """Test the message raised in the case of a truncation error."""
    with assert_exception(ValueError, snapshot):
        OTDistribution_Settings(
            interfaced_distribution="Normal", lower_bound=0.0, upper_bound=-1
        )
