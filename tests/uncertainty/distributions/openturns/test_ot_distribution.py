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
from numpy import allclose
from numpy import array
from numpy import inf
from numpy import int32
from numpy import ndarray
from numpy.random import RandomState
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal
from openturns import RandomGenerator

from gemseo.uncertainty.distributions import scalar_distribution_mixin
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.uncertainty.distributions.openturns.exponential import (
    OTExponentialDistribution,
)
from gemseo.uncertainty.distributions.openturns.fitting import OTDistributionFitter
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distributions.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distributions.openturns.triangular import (
    OTTriangularDistribution,
)
from gemseo.uncertainty.distributions.openturns.uniform import OTUniformDistribution


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


def test_normal() -> None:
    distribution = OTNormalDistribution()
    assert str(distribution) == "Normal(mu=0.0, sigma=1.0)"


def test_uniform() -> None:
    distribution = OTUniformDistribution()
    assert str(distribution) == "Uniform(lower=0.0, upper=1.0)"


def test_exponential() -> None:
    distribution = OTExponentialDistribution()
    assert str(distribution) == "Exponential(rate=1.0, loc=0.0)"


def test_triangular() -> None:
    distribution = OTTriangularDistribution()
    assert str(distribution) == "Triangular(lower=0.0, mode=0.5, upper=1.0)"


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


@pytest.fixture
def norm_data() -> ndarray:
    """Normal samples."""
    return RandomState(1).normal(size=100)


def test_otdistfitter_distribution(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    dist = OTJointDistribution([OTNormalDistribution()] * 2)
    with pytest.raises(TypeError):
        factory.compute_measure(dist, "BIC")


def test_otdistfitter_fit(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    assert isinstance(dist, OTDistribution)


def tst_otdistfitter_bic(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    quality_measure = factory.compute_measure(dist, "BIC")
    assert allclose(quality_measure, 2.59394512877)
    factory = OTDistributionFitter("x", norm_data)
    quality_measure = factory.compute_measure("Normal", "BIC")
    assert allclose(quality_measure, 2.59394512877)


def test_otdistfitter_kolmogorov(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.fit("Normal")
    acceptable, details = factory.compute_measure(dist, "Kolmogorov")
    assert acceptable
    assert "statistics" in details
    assert "p-value" in details
    assert "level" in details
    assert details["level"] == 0.05
    assert allclose(details["statistics"], 0.04330972976650932)
    assert allclose(details["p-value"], 0.9879299613543082)


def test_otdistfitter_available(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    assert "BIC" in factory.available_criteria
    assert "BIC" not in factory.available_significance_tests
    assert "Normal" in factory.available_distributions


def test_otdistfitter_select(norm_data) -> None:
    factory = OTDistributionFitter("x", norm_data)
    dist = factory.select(["Normal", "Exponential"], "BIC")
    assert isinstance(dist, OTDistribution)


def test_compute_cdf_int32():
    """Check that openturns-based compute_cdf handles numpy.int32."""
    expected = pytest.approx(0.8, abs=0.1)
    x = int32(1)
    assert OTNormalDistribution().compute_cdf(x) == expected
    assert OTJointDistribution([OTNormalDistribution()]).compute_cdf([x]) == expected
