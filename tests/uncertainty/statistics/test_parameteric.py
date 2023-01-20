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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import numbers
import re

import openturns as ot
import pytest
from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.parametric import ParametricStatistics
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    ToleranceIntervalSide,
)
from gemseo.utils.testing import image_comparison
from numpy import array
from numpy import inf
from numpy import vstack
from numpy.random import exponential
from numpy.random import lognormal
from numpy.random import normal
from numpy.random import rand
from numpy.random import seed
from numpy.random import weibull
from numpy.testing import assert_allclose
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """This fixture is a random sample of four random variables distributed according to
    the uniform, normal, weibull and exponential probability distributions."""
    seed(0)
    n_samples = 100
    uniform_rand = rand(n_samples)
    normal_rand = normal(size=n_samples)
    weibull_rand = weibull(1.5, size=n_samples)
    exponential_rand = exponential(size=n_samples)
    data = Dataset()
    data.set_from_array(
        vstack((uniform_rand, normal_rand, weibull_rand, exponential_rand)).T,
        ["x_1", "x_2", "x_3"],
        {"x_1": 1, "x_2": 1, "x_3": 2},
    )
    return data


@pytest.fixture(scope="module")
def tested_distributions() -> list[str]:
    """The tested distributions."""
    return ["Exponential", "Normal", "Uniform"]


@pytest.fixture(scope="module")
def statistics(dataset, tested_distributions) -> ParametricStatistics:
    """The statistics associated with the dataset and tested distributions."""
    return ParametricStatistics(dataset, tested_distributions)


def test_wrong_fitting_criterion(dataset, tested_distributions):
    """Check that an ValueError is raised when using a wrong fitting criterion."""
    with pytest.raises(
        ValueError,
        match=(
            r"dummy is not a name of fitting test; "
            r"available ones are: BIC, Kolmogorov, ChiSquared."
        ),
    ):
        ParametricStatistics(dataset, tested_distributions, fitting_criterion="dummy")


def test_str(statistics):
    """Check __str__."""
    assert str(statistics) == (
        "ParametricStatistics_Dataset\n"
        "   n_samples: 100\n"
        "   n_variables: 3\n"
        "   variables: x_1, x_2, x_3"
    )


def test_n_samples(dataset, statistics):
    """Check n_samples."""
    assert statistics.n_samples == dataset.n_samples


def test_n_variables(dataset, statistics):
    """Check n_variables."""
    assert statistics.n_variables == dataset.n_variables


@pytest.mark.parametrize(
    "kwargs,expected_is_pvalue",
    [({}, False), ({"fitting_criterion": "Kolmogorov"}, True)],
)
def test_get_criteria(
    dataset, statistics, tested_distributions, expected_is_pvalue, kwargs
):
    """Check get_criteria()."""
    statistics = ParametricStatistics(dataset, tested_distributions, **kwargs)
    criteria, is_pvalue = statistics.get_criteria("x_1")
    assert is_pvalue == expected_is_pvalue
    for distribution, criterion in criteria.items():
        assert distribution in tested_distributions
        assert isinstance(criterion, numbers.Number)


def test_get_criteria_wrong_fitting_criterion(dataset, tested_distributions):
    """Check get_criteria() with with wrong fitting criterion."""
    with pytest.raises(
        ValueError,
        match=(
            r"foo is not a name of distribution available for fitting\; "
            r"available ones are: Arcsine, Beta,*"
        ),
    ):
        ParametricStatistics(dataset, ["foo"])


@pytest.mark.parametrize(
    [
        "statistic_estimation",
        "statistic_estimator",
        "statistic_estimator_args",
        "statistic_estimator_kwargs",
    ],
    [
        ([-0.004948, -1.58328, 0.050429, 0.011991], "compute_minimum", (), {}),
        ([0.998018, 2.421654, inf, inf], "compute_maximum", (), {}),
        ([1.002966, 4.004934, inf, inf], "compute_range", (), {}),
        ([0.496535, 0.419187, 0.873166, 0.98483], "compute_mean", (), {}),
        (
            [0.289531, 1.156125, 0.822737, 0.972839],
            "compute_standard_deviation",
            (),
            {},
        ),
        ([0.083828, 1.336625, 0.676896, 0.946416], "compute_variance", (), {}),
        (
            [0.496545, 0.479822, 0.579011, 0.60554],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([0.5]),
                    "x_2": 0.5,
                    "x_3": array([0.5, 0.5]),
                },
            ),
            {},
        ),
        (
            [0.503455, 0.520178, 0.420989, 0.39446],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([0.5]),
                    "x_2": 0.5,
                    "x_3": array([0.5, 0.5]),
                },
            ),
            {"greater": False},
        ),
        (
            [0.496545, 0.479822, 0.579011, 0.60554],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([0.5]),
                    "x_2": 0.5,
                    "x_3": array([0.5, 0.5]),
                },
            ),
            {"greater": True},
        ),
        ([0.496535, 0.419187, 0.620707, 0.686311], "compute_quantile", (0.5,), {}),
        ([0.496535, 0.419187, 0.620707, 0.686311], "compute_quartile", (2,), {}),
        ([0.496535, 0.419187, 0.620707, 0.686311], "compute_percentile", (50,), {}),
        ([0.496535, 0.419187, 0.620707, 0.686311], "compute_median", (), {}),
        ([0.496535, 0.419187, 0.873166, 0.98483], "compute_moment", (1,), {}),
    ],
)
def test_statistics(
    statistics,
    statistic_estimation,
    statistic_estimator,
    statistic_estimator_args,
    statistic_estimator_kwargs,
):
    """Check the computation of the different statistics."""
    result = getattr(statistics, statistic_estimator)(
        *statistic_estimator_args, **statistic_estimator_kwargs
    )
    result_1 = result["x_1"]
    result_2 = result["x_2"]
    result_3 = result["x_3"]
    assert_allclose(result_1, statistic_estimation[0:1], atol=1e-6)
    assert_allclose(result_2, statistic_estimation[1:2], atol=1e-6)
    assert_allclose(result_3, statistic_estimation[2:4], atol=1e-6)
    assert result_1.shape == (1,)
    assert result_2.shape == (1,)
    assert result_3.shape == (2,)


def test_compute_margin(statistics):
    """Check compute_margin()."""
    margin = statistics.compute_margin(3.0)
    mean_std = statistics.compute_mean_std(3.0)
    for name, value in margin.items():
        assert_equal(value, mean_std[name])


def test_plot_criteria(tmp_wd, statistics, dataset, tested_distributions):
    """Check plot_criteria()."""
    statistics.plot_criteria("x_2", save=True, show=False)
    statistics.plot_criteria("x_2", title="title", save=True, show=False)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The variable 'dummy' is missing from the dataset; "
            "available ones are: x_1, x_2, x_3."
        ),
    ):
        statistics.plot_criteria("dummy", save=True, show=False)

    stats = ParametricStatistics(
        dataset, tested_distributions, fitting_criterion="Kolmogorov"
    )
    stats.plot_criteria("x_2", save=True, show=False)


@pytest.mark.parametrize(
    "baseline_images,fitting_criterion,title",
    [
        (["fitting_BIC.png"], "BIC", None),
        (["fitting_Kolmogorov.png"], "Kolmogorov", None),
        (["fitting_title.png"], "BIC", "My title"),
    ],
)
@image_comparison(None)
def test_plot_criteria_images(
    baseline_images, dataset, pyplot_close_all, fitting_criterion, title
):
    statistics = ParametricStatistics(
        dataset,
        ["Exponential", "Normal", "Uniform"],
        fitting_criterion=fitting_criterion,
    )
    statistics.plot_criteria("x_1", show=False, title=title)


@pytest.mark.parametrize("coverage", [-0.5, 1.5])
def test_tolerance_interval_wrong_coverage(statistics, dataset, coverage):
    """Check tolerance_interval() with a wrong coverage."""
    with pytest.raises(
        ValueError,
        match=re.escape("The argument 'coverage' must be a number in [0,1]."),
    ):
        statistics.compute_tolerance_interval(coverage)


@pytest.mark.parametrize("confidence", [-0.5, 1.5])
def test_tolerance_interval_wrong_confidence(statistics, dataset, confidence):
    """Check tolerance_interval() with a wrong confidence."""
    with pytest.raises(
        ValueError,
        match=re.escape("The argument 'confidence' must be a number in [0,1]."),
    ):
        statistics.compute_tolerance_interval(0.1, confidence=confidence)


@pytest.mark.parametrize(
    "distribution,generate_samples",
    [
        ("Exponential", lambda n: exponential(size=n)),
        ("WeibullMin", lambda n: array(ot.WeibullMin().getSample(n))),
        ("LogNormal", lambda n: lognormal(size=n)),
        ("Uniform", lambda n: rand(n)),
        ("Normal", lambda n: normal(size=n)),
    ],
)
def test_tolerance_interval(generate_samples, distribution):
    """Check compute_tolerance_intervals() with different distributions."""
    seed(0)
    dataset = Dataset()
    dataset.set_from_array(generate_samples(100).reshape((-1, 1)))
    statistics = ParametricStatistics(dataset, [distribution])
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=ToleranceIntervalSide.BOTH
    )
    assert tolerance_interval["x_0"][0].lower.shape == (1,)
    assert tolerance_interval["x_0"][0].upper.shape == (1,)
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=ToleranceIntervalSide.UPPER
    )
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=ToleranceIntervalSide.LOWER
    )
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    assert tolerance_interval["x_0"][0].upper == inf

    b_value = statistics.compute_tolerance_interval(
        0.9, side=ToleranceIntervalSide.LOWER
    )
    a_value = statistics.compute_tolerance_interval(
        0.95, side=ToleranceIntervalSide.LOWER
    )
    assert b_value["x_0"][0].lower >= a_value["x_0"][0].lower


def test_abvalue_normal():
    """Check that A-value is lower than B-value."""
    seed(0)
    dataset = Dataset()
    dataset.set_from_array(normal(size=100).reshape((-1, 1)))
    stats = ParametricStatistics(dataset, ["Normal"])
    assert stats.compute_a_value()["x_0"][0] <= stats.compute_b_value()["x_0"][0]


def test_available(statistics):
    assert "Normal" in ParametricStatistics.AVAILABLE_DISTRIBUTIONS
    assert "BIC" in ParametricStatistics.AVAILABLE_CRITERIA
    assert "Kolmogorov" in ParametricStatistics.AVAILABLE_SIGNIFICANCE_TESTS
    assert "Normal" in statistics.get_fitting_matrix()


@pytest.mark.parametrize(
    "name,options,expression",
    [
        (
            "tolerance_interval",
            {"coverage": 0.9, "tolerance": 0.99, "side": ToleranceIntervalSide.LOWER},
            "TI[X; 0.9, LOWER, 0.99]",
        ),
        (
            "tolerance_interval",
            {
                "show_name": True,
                "coverage": 0.9,
                "tolerance": 0.99,
                "side": ToleranceIntervalSide.LOWER,
            },
            "TI[X; coverage=0.9, side=LOWER, tolerance=0.99]",
        ),
        ("a_value", {}, "Aval[X]"),
        ("b_value", {}, "Bval[X]"),
        ("maximum", {}, "Max[X]"),
        ("mean", {}, "E[X]"),
        ("mean_std", {}, "E_StD[X]"),
        ("mean_std", {"factor": 3.0}, "E_StD[X; 3.0]"),
        ("margin", {}, "Margin[X]"),
        ("margin", {"factor": 3.0}, "Margin[X; 3.0]"),
        ("minimum", {}, "Min[X]"),
        ("percentile", {"order": 10}, "p[X; 10]"),
        ("probability", {"value": 1.0}, "P[X >= 1.0]"),
        ("probability", {"value": 1.0, "greater": False}, "P[X <= 1.0]"),
        ("quantile", {}, "Q[X]"),
        ("quartile", {"order": 1}, "q[X; 1]"),
        ("range", {}, "R[X]"),
        ("variance", {}, "V[X]"),
        ("moment", {}, "M[X]"),
        ("foo", {}, "foo[X]"),
        ("foo", {"bar": 2}, "foo[X; 2]"),
        ("foo", {"bar": 2, "show_name": True}, "foo[X; bar=2]"),
    ],
)
def test_expression(name, options, expression):
    """Check the string expression of a statistic applied to a variable."""
    assert ParametricStatistics.compute_expression("X", name, **options) == expression
