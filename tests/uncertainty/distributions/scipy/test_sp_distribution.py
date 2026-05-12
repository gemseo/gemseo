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
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import scipy.stats as scipy_stats
from numpy import array
from numpy import exp
from numpy import inf
from numpy.random import RandomState
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo.uncertainty.distributions._log_normal_utils import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distributions.factory import DISTRIBUTION_FACTORY
from gemseo.uncertainty.distributions.scipy.beta_settings import (
    SPBetaDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution
from gemseo.uncertainty.distributions.scipy.distribution_settings import (
    SPDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.exponential_settings import (
    SPExponentialDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.joint import SPJointDistribution
from gemseo.uncertainty.distributions.scipy.log_normal import SPLogNormalDistribution
from gemseo.uncertainty.distributions.scipy.log_normal_settings import (
    SPLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.normal_settings import (
    SPNormalDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.triangular_settings import (
    SPTriangularDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.uniform import SPUniformDistribution
from gemseo.uncertainty.distributions.scipy.uniform_settings import (
    SPUniformDistribution_Settings,
)
from gemseo.uncertainty.distributions.scipy.weibull_settings import (
    SPWeibullDistribution_Settings,
)
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(scope="module")
def distribution() -> SPDistribution:
    """A normal distribution with mean 0 and std 2."""
    return SPDistribution(
        SPDistribution_Settings(
            interfaced_distribution="norm", parameters={"loc": 0.0, "scale": 2.0}
        )
    )


def test_joint_distribution() -> None:
    """Check the joint probability distribution associated with a SPDistribution."""
    assert SPJointDistribution == SPDistribution.JOINT_DISTRIBUTION_CLASS


def test_constructor() -> None:
    distribution = SPDistribution(
        SPDistribution_Settings(
            interfaced_distribution="norm", parameters={"loc": 0.0, "scale": 1}
        )
    )
    assert distribution.transformation == "x"


def test_bad_distribution(snapshot) -> None:
    with assert_exception(ImportError, snapshot):
        SPDistribution(
            SPDistribution_Settings(
                interfaced_distribution="Dummy", parameters={"loc": 0.0, "scale": 1}
            )
        )


def test_bad_distribution_parameters(snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        SPDistribution(
            SPDistribution_Settings(
                interfaced_distribution="norm", parameters={"loc": 0.0, "max": 1}
            )
        )


def test_str(distribution) -> None:
    assert str(distribution) == "norm(loc=0.0, scale=2.0)"
    distribution = SPDistribution(
        SPDistribution_Settings(
            interfaced_distribution="norm",
            parameters={"loc": 0.0, "scale": 2.0},
            standard_parameters={"mean": 0, "var": 4},
        )
    )
    assert str(distribution) == "norm(mean=0, var=4)"


def test_compute_samples(distribution) -> None:
    random_state = RandomState(0)
    sample = distribution.compute_samples(3, random_state)
    assert sample.ndim == 1
    assert_almost_equal(sample, array([3.528105, 0.800314, 1.957476]), decimal=3)


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
    assert_almost_equal(distribution.range, array([-14.068968, 14.068974]), decimal=3)


MU_L_0, SIGMA_L_0 = compute_mu_l_and_sigma_l(1.0, 1.0, 0.0)
MU_L_1, SIGMA_L_1 = compute_mu_l_and_sigma_l(1.2, 1.3, 0.4)


@pytest.mark.parametrize(
    ("sp_dist_name", "settings", "sp_kwargs", "str_"),
    [
        (
            "beta",
            SPBetaDistribution_Settings(),
            {"a": 2.0, "b": 2.0, "loc": 0.0, "scale": 1.0},
            "beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)",
        ),
        (
            "beta",
            SPBetaDistribution_Settings(alpha=0.1, beta=0.2, minimum=0.3, maximum=0.4),
            {"a": 0.1, "b": 0.2, "loc": 0.3, "scale": 0.1},
            "beta(lower=0.3, upper=0.4, alpha=0.1, beta=0.2)",
        ),
        (
            "expon",
            SPExponentialDistribution_Settings(),
            {"scale": 1.0, "loc": 0.0},
            "expon(rate=1.0, loc=0.0)",
        ),
        (
            "expon",
            SPExponentialDistribution_Settings(rate=0.1, loc=0.2),
            {"scale": 10.0, "loc": 0.2},
            "expon(rate=0.1, loc=0.2)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution_Settings(),
            {"s": SIGMA_L_0, "scale": exp(MU_L_0), "loc": 0.0},
            "lognorm(mu=1.0, sigma=1.0, loc=0.0)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution_Settings(mu=1.5, sigma=2.5, set_log=True),
            {"s": 2.5, "scale": exp(1.5), "loc": 0.0},
            "lognorm(mu=1.5, sigma=2.5, loc=0.0)",
        ),
        (
            "lognorm",
            SPLogNormalDistribution_Settings(mu=1.2, sigma=1.3, location=0.4),
            {"s": SIGMA_L_1, "scale": exp(MU_L_1), "loc": 0.4},
            "lognorm(mu=1.2, sigma=1.3, loc=0.4)",
        ),
        (
            "norm",
            SPNormalDistribution_Settings(),
            {"loc": 0.0, "scale": 1.0},
            "norm(mu=0.0, sigma=1.0)",
        ),
        (
            "norm",
            SPNormalDistribution_Settings(mu=0.1, sigma=0.2),
            {"loc": 0.1, "scale": 0.2},
            "norm(mu=0.1, sigma=0.2)",
        ),
        (
            "triang",
            SPTriangularDistribution_Settings(),
            {"loc": 0.0, "scale": 1.0, "c": 0.5},
            "triang(lower=0.0, mode=0.5, upper=1.0)",
        ),
        (
            "triang",
            SPTriangularDistribution_Settings(minimum=0.1, mode=0.2, maximum=0.3),
            {"loc": 0.1, "scale": 0.2, "c": 0.5},
            "triang(lower=0.1, mode=0.2, upper=0.3)",
        ),
        (
            "uniform",
            SPUniformDistribution_Settings(),
            {"loc": 0.0, "scale": 1.0},
            "uniform(lower=0.0, upper=1.0)",
        ),
        (
            "uniform",
            SPUniformDistribution_Settings(minimum=0.1, maximum=0.2),
            {"loc": 0.1, "scale": 0.1},
            "uniform(lower=0.1, upper=0.2)",
        ),
        (
            "weibull_min",
            SPWeibullDistribution_Settings(),
            {"loc": 0.0, "scale": 1.0, "c": 1.0},
            "weibull_min(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            "weibull_min",
            SPWeibullDistribution_Settings(location=0.1, scale=0.2, shape=0.3),
            {"loc": 0.1, "scale": 0.2, "c": 0.3},
            "weibull_min(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            "weibull_max",
            SPWeibullDistribution_Settings(use_weibull_min=False),
            {"loc": 0.0, "scale": 1.0, "c": 1.0},
            "weibull_max(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            "weibull_max",
            SPWeibullDistribution_Settings(
                use_weibull_min=False, location=0.1, scale=0.2, shape=0.3
            ),
            {"loc": 0.1, "scale": 0.2, "c": 0.3},
            "weibull_max(location=0.1, scale=0.2, shape=0.3)",
        ),
    ],
)
def test_specific_sp_distributions(sp_dist_name, settings, sp_kwargs, str_) -> None:
    """Check the specific SciPy-based distributions.

    Args:
        sp_dist_name: The name of the SciPy class.
        settings: The settings of the SP distribution.
        sp_kwargs: The keyword arguments to instantiate the SciPy class.
        str_: The expected string representation of the distribution.
    """
    sp_distribution = DISTRIBUTION_FACTORY.create_from_settings(settings)
    distribution = sp_distribution.distribution
    expected_distribution = getattr(scipy_stats, sp_dist_name)(**sp_kwargs)
    assert distribution.dist.name == expected_distribution.dist.name
    assert distribution.mean() == pytest.approx(expected_distribution.mean())
    assert str(sp_distribution) == str_


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("SPBetaDistribution", "beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)"),
        ("SPDistribution", "uniform()"),
        ("SPExponentialDistribution", "expon(rate=1.0, loc=0.0)"),
        ("SPLogNormalDistribution", "lognorm(mu=1.0, sigma=1.0, loc=0.0)"),
        ("SPNormalDistribution", "norm(mu=0.0, sigma=1.0)"),
        ("SPTriangularDistribution", "triang(lower=0.0, mode=0.5, upper=1.0)"),
        ("SPUniformDistribution", "uniform(lower=0.0, upper=1.0)"),
        ("SPWeibullDistribution", "weibull_min(location=0.0, scale=1.0, shape=1.0)"),
    ],
)
def test_specific_sp_distributions_default(class_name, expected):
    """Check the SciPy-based distributions using default settings."""
    distribution = DISTRIBUTION_FACTORY.create(class_name)
    assert str(distribution) == expected


def test_lognormal_distribution():
    """Check that (mu,sigma)=(1,1) for the default log-normal distribution.

    We do this test because we transform the (mu, sigma) into (mu_l, sigma_l) and
    instantiate the wrapped distribution with it. This is a way to test this
    transformation.
    """
    distribution = SPLogNormalDistribution()
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.standard_deviation == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("save", "show", "directory_path"),
    [
        (True, False, ""),
        (True, False, "my_dir"),
        (False, True, ""),
        (False, False, ""),
    ],
)
def test_plot_html(save, show, directory_path) -> None:
    """Check the save and show behavior of plot() in the HTML branch."""
    distribution = SPUniformDistribution()
    mock_fig = MagicMock()
    with patch(
        "gemseo.uncertainty.distributions.base_univariate.make_subplots",
        return_value=mock_fig,
    ):
        distribution.plot(
            save=save, show=show, file_extension="html", directory_path=directory_path
        )
    expected_dir = Path().cwd() if directory_path == "" else Path(directory_path)
    if save:
        mock_fig.write_html.assert_called_once_with(expected_dir / "pdf_cdf.html")
    else:
        mock_fig.write_html.assert_not_called()
    if show:
        mock_fig.show.assert_called_once()
    else:
        mock_fig.show.assert_not_called()
