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

import contextlib
import numbers
import re
from unittest import mock

import openturns as ot
import pytest
from matplotlib.figure import Figure
from numpy import array
from numpy import inf
from numpy import vstack
from numpy.random import RandomState
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.datasets.dataset import Dataset
from gemseo.uncertainty.distributions.scalar_distribution_mixin import (
    ScalarDistributionMixin,
)
from gemseo.uncertainty.distributions.scipy import distribution_fitter
from gemseo.uncertainty.distributions.scipy.distribution_fitter import (
    SPDistributionFitter,
)
from gemseo.uncertainty.statistics.ot_parametric_statistics import (
    OTParametricStatistics,
)
from gemseo.uncertainty.statistics.parametric_statistics import ParametricStatistics
from gemseo.uncertainty.statistics.sp_parametric_statistics import (
    SPParametricStatistics,
)
from gemseo.uncertainty.statistics.tolerance_interval.distribution import (
    BaseToleranceInterval,
)
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.repr_html import REPR_HTML_WRAPPER
from gemseo.utils.testing.helpers import image_comparison

RNG = RandomState(0)


@pytest.fixture(scope="module")
def dataset() -> Dataset:
    """This fixture is a random sample of four random variables distributed according to
    the uniform, normal, weibull and exponential probability distributions."""
    n_samples = 100
    uniform_rand = RNG.random(n_samples)
    normal_rand = RNG.normal(size=n_samples)
    weibull_rand = RNG.weibull(1.5, size=n_samples)
    exponential_rand = RNG.exponential(size=n_samples)
    return Dataset.from_array(
        vstack((uniform_rand, normal_rand, weibull_rand, exponential_rand)).T,
        ["x_1", "x_2", "x_3"],
        {"x_1": 1, "x_2": 1, "x_3": 2},
    )


@pytest.fixture(scope="module")
def sp_dataset() -> Dataset:
    """This fixture is a random sample of four random variables distributed according to
    the uniform, normal, weibull and exponential probability distributions."""
    # This dataset is the same as the previous one,
    # but on the windows CI,
    # using the same fixture caused problems
    # with test_sp_statistics and test_sp_statistics_custom.
    n_samples = 100
    rng = RandomState(0)
    uniform_rand = rng.random(n_samples)
    normal_rand = rng.normal(size=n_samples)
    weibull_rand = rng.weibull(1.5, size=n_samples)
    exponential_rand = rng.exponential(size=n_samples)
    return Dataset.from_array(
        vstack((uniform_rand, normal_rand, weibull_rand, exponential_rand)).T,
        ["x_1", "x_2", "x_3"],
        {"x_1": 1, "x_2": 1, "x_3": 2},
    )


@pytest.fixture(scope="module")
def statistics(dataset, tested_distributions) -> OTParametricStatistics:
    """OpenTURNS statistics associated with the dataset and tested distributions."""
    return OTParametricStatistics(dataset, tested_distributions)


@pytest.fixture(scope="module")
def tested_distributions() -> list[str]:
    """The tested distributions."""
    return ["Exponential", "Normal", "Uniform"]


def test_parametric_statistics():
    """Check that ParametricStatistics is OTParametricStatistics."""
    assert ParametricStatistics is OTParametricStatistics


@image_comparison(["pdf_cdf_x_1", "pdf_cdf_x_2", "pdf_cdf_x_3_0", "pdf_cdf_x_3_1"])
def test_plot(statistics) -> None:
    """Check the visualizations generated by the method plot()."""
    graphs = statistics.plot(show=False)
    assert len(graphs) == 4
    for graph in graphs.values():
        assert isinstance(graph, Figure)


def test_repr(statistics) -> None:
    """Check __repr__."""
    assert (
        repr(statistics)
        == str(statistics)
        == (
            "OTParametricStatistics(Dataset)\n"
            "   n_samples: 100\n"
            "   n_variables: 3\n"
            "   variables: x_1, x_2, x_3"
        )
    )


def test_repr_html_(statistics) -> None:
    """Check _repr_html_."""
    assert statistics._repr_html_() == REPR_HTML_WRAPPER.format(
        "OTParametricStatistics(Dataset)<br/>"
        "<ul>"
        "<li>n_samples: 100</li>"
        "<li>n_variables: 3</li>"
        "<li>variables: x_1, x_2, x_3</li>"
        "</ul>"
    )


def test_n_samples(dataset, statistics) -> None:
    """Verify that n_samples corresponds to the number of samples in the dataset."""
    assert statistics.n_samples == len(dataset)


def test_n_variables(dataset, statistics) -> None:
    """Verify that n_variables corresponds to the variable names in the dataset."""
    assert statistics.n_variables == len(dataset.variable_names)


@pytest.mark.parametrize(
    ("kwargs", "expected_is_pvalue"),
    [({}, False), ({"fitting_criterion": "Kolmogorov"}, True)],
)
def test_get_criteria(
    dataset, statistics, tested_distributions, expected_is_pvalue, kwargs
) -> None:
    """Verify the results returned by get_criteria."""
    statistics = OTParametricStatistics(dataset, tested_distributions, **kwargs)
    criteria, is_pvalue = statistics.get_criteria("x_1")
    assert is_pvalue == expected_is_pvalue
    for distribution, criterion in criteria.items():
        assert distribution in tested_distributions
        assert isinstance(criterion, numbers.Number)


@pytest.mark.parametrize(
    (
        "statistic_estimation",
        "statistic_estimator",
        "statistic_estimator_args",
        "statistic_estimator_kwargs",
    ),
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
) -> None:
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


def test_compute_margin(statistics) -> None:
    """Verify that compute_margin and compute_mean_std compute the same quantity."""
    margin = statistics.compute_margin(3.0)
    mean_std = statistics.compute_mean_std(3.0)
    for name, value in margin.items():
        assert_equal(value, mean_std[name])


def test_plot_criteria(tmp_wd, statistics, dataset, tested_distributions) -> None:
    """Check plot_criteria()."""
    fig = statistics.plot_criteria("x_2", save=True, show=False)
    assert isinstance(fig, Figure)
    statistics.plot_criteria("x_2", title="title", save=True, show=False)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The variable 'dummy' is missing from the dataset; "
            "available ones are: x_1, x_2, x_3."
        ),
    ):
        statistics.plot_criteria("dummy", save=True, show=False)

    stats = OTParametricStatistics(
        dataset, tested_distributions, fitting_criterion="Kolmogorov"
    )
    stats.plot_criteria("x_2", save=True, show=False)


@pytest.mark.parametrize(
    ("baseline_images", "fitting_criterion", "title"),
    [
        (["fitting_BIC.png"], "BIC", None),
        (["fitting_Kolmogorov.png"], "Kolmogorov", None),
        (["fitting_title.png"], "BIC", "My title"),
    ],
)
@image_comparison(None)
def test_plot_criteria_images(
    baseline_images, dataset, fitting_criterion, title
) -> None:
    """Verify that the images generated by plot_criteria are correct."""
    statistics = OTParametricStatistics(
        dataset,
        ["Exponential", "Normal", "Uniform"],
        fitting_criterion=fitting_criterion,
    )
    statistics.plot_criteria("x_1", show=False, title=title)


@pytest.mark.parametrize("coverage", [-0.5, 1.5])
def test_tolerance_interval_wrong_coverage(statistics, coverage) -> None:
    """Verify that compute_tolerance_interval raises an error if wrong coverage."""
    with pytest.raises(
        ValueError,
        match=re.escape("The argument 'coverage' must be a number in [0,1]."),
    ):
        statistics.compute_tolerance_interval(coverage)


@pytest.mark.parametrize("confidence", [-0.5, 1.5])
def test_tolerance_interval_wrong_confidence(statistics, confidence) -> None:
    """Verify that compute_tolerance_interval raises an error if wrong confidence."""
    with pytest.raises(
        ValueError,
        match=re.escape("The argument 'confidence' must be a number in [0,1]."),
    ):
        statistics.compute_tolerance_interval(0.1, confidence=confidence)


@pytest.mark.parametrize(
    ("distribution", "generate_samples"),
    [
        ("Exponential", lambda n: RNG.exponential(size=n)),
        ("WeibullMin", lambda n: array(ot.WeibullMin().getSample(n))),
        ("LogNormal", lambda n: RNG.lognormal(size=n)),
        ("Uniform", RNG.random),
        ("Normal", lambda n: RNG.normal(size=n)),
    ],
)
def test_tolerance_interval(generate_samples, distribution) -> None:
    """Check compute_tolerance_intervals() with different distributions."""
    dataset = Dataset.from_array(generate_samples(100).reshape((-1, 1)))
    statistics = OTParametricStatistics(dataset, [distribution])
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=BaseToleranceInterval.ToleranceIntervalSide.BOTH
    )
    assert tolerance_interval["x_0"][0].lower.shape == (1,)
    assert tolerance_interval["x_0"][0].upper.shape == (1,)
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=BaseToleranceInterval.ToleranceIntervalSide.UPPER
    )
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    tolerance_interval = statistics.compute_tolerance_interval(
        0.1, side=BaseToleranceInterval.ToleranceIntervalSide.LOWER
    )
    assert tolerance_interval["x_0"][0].lower <= tolerance_interval["x_0"][0].upper
    assert tolerance_interval["x_0"][0].upper == inf

    b_value = statistics.compute_tolerance_interval(
        0.9, side=BaseToleranceInterval.ToleranceIntervalSide.LOWER
    )
    a_value = statistics.compute_tolerance_interval(
        0.95, side=BaseToleranceInterval.ToleranceIntervalSide.LOWER
    )
    assert b_value["x_0"][0].lower >= a_value["x_0"][0].lower


def test_abvalue_normal() -> None:
    """Check that A-value is lower than B-value."""
    dataset = Dataset.from_array(RNG.normal(size=(100, 1)))
    stats = OTParametricStatistics(dataset, ["Normal"])
    assert stats.compute_a_value()["x_0"][0] <= stats.compute_b_value()["x_0"][0]


def test_available(statistics) -> None:
    """Verify that Normal is a DistributionName and a key of the fitting matrix."""
    assert "Normal" in OTParametricStatistics.DistributionName.__members__
    assert "Normal" in statistics.get_fitting_matrix()


@pytest.mark.parametrize(
    ("name", "options", "expression"),
    [
        (
            "tolerance_interval",
            {
                "coverage": 0.9,
                "tolerance": 0.99,
                "side": BaseToleranceInterval.ToleranceIntervalSide.LOWER,
            },
            "TI[X; 0.9, 0.99, lower]",
        ),
        (
            "tolerance_interval",
            {
                "show_name": True,
                "coverage": 0.9,
                "tolerance": 0.99,
                "side": BaseToleranceInterval.ToleranceIntervalSide.LOWER,
            },
            "TI[X; coverage=0.9, side=lower, tolerance=0.99]",
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
def test_expression(name, options, expression) -> None:
    """Check the string expression of a statistic applied to a variable."""
    assert OTParametricStatistics.compute_expression("X", name, **options) == expression


def test_plot_args(statistics) -> None:
    """Check the arguments passed to ScalarDistributionMixin.plot by the method
    plot()."""
    with mock.patch.object(ScalarDistributionMixin, "plot") as plot:
        statistics.plot(save=1, show=2, directory_path=3, file_format=4)

    assert len(plot.call_args.args) == 0
    assert plot.call_args.kwargs == {
        "save": 1,
        "show": 2,
        "directory_path": 3,
        "file_extension": 4,
    }


def test_sp_statistics(sp_dataset):
    """Check SPParametricStatistics with default settings."""
    statistics = SPParametricStatistics(sp_dataset, ("expon", "norm", "uniform"))
    assert compare_dict_of_arrays(
        statistics.compute_mean(),
        {
            "x_1": array([0.47279384]),
            "x_2": array([0.19233434]),
            "x_3": array([0.87316603, 0.98482981]),
        },
        tolerance=0.1,
    )


def test_sp_statistics_custom(sp_dataset):
    """Check SPParametricStatistics with custom settings."""
    statistics = SPParametricStatistics(
        sp_dataset,
        ("expon", "norm", "uniform"),
        fitting_criterion=SPParametricStatistics.FittingCriterion.FILLIBEN,
        level=0.1,
        selection_criterion=SPParametricStatistics.SelectionCriterion.FIRST,
    )
    assert compare_dict_of_arrays(
        statistics.compute_mean(),
        {
            "x_1": array([0.49653466]),
            "x_2": array([0.41918684]),
            "x_3": array([0.87316603, 0.98482981]),
        },
        tolerance=0.1,
    )


@pytest.mark.parametrize("fitting_criterion", SPParametricStatistics.FittingCriterion)
def test_sp_fitting_criteria(sp_dataset, fitting_criterion):
    """Verify that fitting_criterion is passed to scipy.stats.goodness_of_fit."""
    with mock.patch.object(distribution_fitter, "goodness_of_fit") as goodness_of_fit:  # noqa: SIM117
        with contextlib.suppress(TypeError):
            SPParametricStatistics(
                sp_dataset, ("expon", "norm"), fitting_criterion=fitting_criterion
            )

    assert (
        goodness_of_fit.call_args.kwargs["statistic"]
        == SPDistributionFitter._CRITERIA_TO_WRAPPED_OBJECTS[fitting_criterion]
    )
