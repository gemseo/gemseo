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

import re
from unittest import mock

import pytest
from numpy import array
from numpy import concatenate
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_allclose

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.boxplot import Boxplot
from gemseo.post.dataset.lines import Lines
from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.utils.testing.helpers import concretize_classes
from gemseo.utils.testing.helpers import image_comparison


@pytest.fixture(scope="module")
def dataset():
    """A set of data."""
    data = Dataset()
    column = linspace(1.0, 10.0, 10)[:, newaxis]
    data.add_variable("x_1", column)
    data.add_variable("x_2", concatenate((-column, column), 1))
    return data


@pytest.fixture(scope="module")
def statistics(dataset):
    """The statistics for both x_1 and x_2."""
    return EmpiricalStatistics(dataset)


@pytest.fixture(scope="module")
def x_1_statistics(dataset):
    """The statistics for x_1 only."""
    return EmpiricalStatistics(dataset, ["x_1"])


def test_n_samples(dataset, statistics):
    """Check that the number of samples corresponds to the length of the dataset."""
    assert statistics.n_samples == len(dataset)


def test_n_variables(dataset, statistics):
    """Check that the number of variables is equal to the one in the dataset."""
    assert statistics.n_variables == len(dataset.variable_names)


def test_n_variables_x_1(x_1_statistics):
    """Check that x_1 statistics use only 1 variable."""
    assert x_1_statistics.n_variables == 1


def test_variables(dataset, statistics):
    """Check that the variables are the variables of the dataset."""
    assert statistics.names == dataset.variable_names


def test_variables_x_1(x_1_statistics):
    """Check that x_1 statistics use only x_1."""
    assert x_1_statistics.names == ["x_1"]


@pytest.mark.parametrize(
    (
        "statistic_estimation",
        "statistic_estimator",
        "statistic_estimator_args",
        "statistic_estimator_kwargs",
    ),
    [
        ([1.0, -10.0, 1.0], "compute_minimum", (), {}),
        ([10.0, -1.0, 10.0], "compute_maximum", (), {}),
        ([9.0, 9.0, 9.0], "compute_range", (), {}),
        ([5.5, -5.5, 5.5], "compute_mean", (), {}),
        ([2.872281, 2.872281, 2.872281], "compute_standard_deviation", (), {}),
        ([8.25, 8.25, 8.25], "compute_variance", (), {}),
        (
            [0.8, 0.3, 0.8],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([-3.0, 3.0]),
                },
            ),
            {},
        ),
        (
            [0.3, 0.8, 0.3],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([-3.0, 3.0]),
                },
            ),
            {"greater": False},
        ),
        (
            [0.8, 0.3, 0.8],
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([-3.0, 3.0]),
                },
            ),
            {"greater": True},
        ),
        ([5.5, -5.5, 5.5], "compute_quantile", (0.5,), {}),
        ([5.5, -5.5, 5.5], "compute_quartile", (2,), {}),
        ([5.5, -5.5, 5.5], "compute_percentile", (50,), {}),
        ([5.5, -5.5, 5.5], "compute_median", (), {}),
        ([0.0, 0.0, 0.0], "compute_moment", (1,), {}),
    ],
)
def test_statistics(
    statistics,
    x_1_statistics,
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
    assert_allclose(result_1, array(statistic_estimation[0:1]), atol=1e-6)
    assert_allclose(result_2, array(statistic_estimation[1:3]), atol=1e-6)
    assert result_1.shape == (1,)
    assert result_2.shape == (2,)

    result = getattr(x_1_statistics, statistic_estimator)(
        *statistic_estimator_args, **statistic_estimator_kwargs
    )
    assert_allclose(result["x_1"], array(statistic_estimation[0:1]), atol=1e-6)
    assert "x_2" not in result


def test_quartile_error(statistics):
    """Check that compute_quartile() raises an error when using a wrong order."""
    with pytest.raises(
        ValueError, match=re.escape("Quartile order must be in {1, 2, 3}.")
    ):
        statistics.compute_quartile(0.25)


@pytest.mark.parametrize("order", [0.25, -1])
def test_percentile_error(statistics, order):
    """Check that compute_percentile() raises an error when using a wrong order."""
    with pytest.raises(
        TypeError, match=re.escape("Percentile order must be in {0, 1, 2, ..., 100}.")
    ):
        statistics.compute_percentile(order)


@pytest.mark.parametrize(
    ("value", "kwargs"),
    [
        ([0.8, 0.1], {}),
        ([0.3, 0.1], {"greater": False}),
        ([0.8, 0.1], {"greater": True}),
    ],
)
def test_compute_joint_probability(statistics, value, kwargs):
    """Check compute_joint_probability()."""
    result = statistics.compute_joint_probability(
        {
            "x_1": array([3.0]),
            "x_2": array([-3.0, 3.0]),
        },
        **kwargs,
    )
    result_1 = result["x_1"]
    result_2 = result["x_2"]
    assert_allclose(result_1, value[0], atol=1e-6)
    assert_allclose(result_2, value[1], atol=1e-6)
    assert isinstance(result_1, float)
    assert isinstance(result_2, float)


def test_variation_coefficient():
    """Check compute_variation_coefficient()."""

    class NewStatistics(Statistics):
        compute_mean = lambda self: {"x": 2}  # noqa: E731
        compute_standard_deviation = lambda self: {"x": 6}  # noqa: E731

    with concretize_classes(NewStatistics):
        assert NewStatistics(Dataset()).compute_variation_coefficient() == {"x": 3}


@image_comparison(["boxplot_x_1", "boxplot_x_2"])
def test_plot_boxplot(statistics, pyplot_close_all):
    """Check the visualizations generated by the method plot_boxplot()."""
    graphs = statistics.plot_boxplot(show=False)
    assert len(graphs) == 2
    for graph in graphs.values():
        assert isinstance(graph, Boxplot)


def test_plot_boxplot_args(statistics):
    """Check the arguments passed to Boxplot by the method plot_boxplot()."""
    with mock.patch.object(Boxplot, "__init__", return_value=None) as __init__:  # noqa: SIM117
        with mock.patch.object(Boxplot, "execute") as execute:
            statistics.plot_boxplot(
                save=1, show=2, directory_path=3, file_format=4, kwarg=5
            )

    assert __init__.call_args.args == (statistics.dataset,)
    assert __init__.call_args.kwargs == {"kwarg": 5, "variables": ["x_2"]}
    assert execute.call_args.kwargs == {
        "save": 1,
        "show": 2,
        "directory_path": 3,
        "file_format": 4,
    }


@image_comparison(["cdf_x_1", "cdf_x_2_0", "cdf_x_2_1"])
def test_plot_cdf(statistics, pyplot_close_all):
    """Check the visualizations generated by the method plot_cdf()."""
    graphs = statistics.plot_cdf(show=False)
    assert len(graphs) == 3
    for graph in graphs.values():
        assert isinstance(graph, Lines)


def test_plot_cdf_args(statistics):
    """Check the arguments passed to Lines by the method plot_cdf()."""
    with mock.patch.object(Lines, "execute") as execute:
        statistics.plot_cdf(
            save=1, show=2, directory_path=3, file_format=4, plot_abscissa_variable=True
        )

    assert execute.call_args.kwargs == {
        "save": 1,
        "show": 2,
        "directory_path": 3,
        "file_format": 4,
    }


@image_comparison(["pdf_x_1", "pdf_x_2_0", "pdf_x_2_1"])
def test_plot_pdf(statistics, pyplot_close_all):
    """Check the visualizations generated by the method plot_pdf()."""
    graphs = statistics.plot_pdf(show=False)
    assert len(graphs) == 3
    for graph in graphs.values():
        assert isinstance(graph, Lines)


def test_plot_pdf_args(statistics):
    """Check the arguments passed to Lines by the method plot_pdf()."""
    with mock.patch.object(Lines, "execute") as execute:
        statistics.plot_pdf(
            save=1, show=2, directory_path=3, file_format=4, plot_abscissa_variable=True
        )
    assert execute.call_args.kwargs == {
        "save": 1,
        "show": 2,
        "directory_path": 3,
        "file_format": 4,
    }
