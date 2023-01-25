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

import pytest
from gemseo.core.dataset import Dataset
from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics
from gemseo.uncertainty.statistics.statistics import Statistics
from gemseo.utils.pytest_conftest import concretize_classes
from numpy import array
from numpy import concatenate
from numpy import linspace
from numpy import newaxis
from numpy.testing import assert_allclose


@pytest.fixture(scope="module")
def dataset():
    """A set of data."""
    data = Dataset()
    column = linspace(1.0, 10.0, 10)[:, newaxis]
    data.add_variable("x_1", column)
    data.add_variable("x_2", concatenate((column, column), 1))
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
    assert statistics.n_samples == dataset.length


def test_n_variables(dataset, statistics):
    """Check that the number of variables is equal to the one in the dataset."""
    assert statistics.n_variables == dataset.n_variables


def test_n_variables_x_1(x_1_statistics):
    """Check that x_1 statistics use only 1 variable."""
    assert x_1_statistics.n_variables == 1


def test_variables(dataset, statistics):
    """Check that the variables are the variables of the dataset."""
    assert statistics.names == dataset.variables


def test_variables_x_1(x_1_statistics):
    """Check that x_1 statistics use only x_1."""
    assert x_1_statistics.names == ["x_1"]


@pytest.mark.parametrize(
    [
        "statistic_estimation",
        "statistic_estimator",
        "statistic_estimator_args",
        "statistic_estimator_kwargs",
    ],
    [
        (1.0, "compute_minimum", (), {}),
        (10.0, "compute_maximum", (), {}),
        (9.0, "compute_range", (), {}),
        (5.5, "compute_mean", (), {}),
        (2.872281, "compute_standard_deviation", (), {}),
        (8.25, "compute_variance", (), {}),
        (
            0.8,
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([3.0, 3.0]),
                },
            ),
            {},
        ),
        (
            0.3,
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([3.0, 3.0]),
                },
            ),
            {"greater": False},
        ),
        (
            0.8,
            "compute_probability",
            (
                # thresh
                {
                    "x_1": array([3.0]),
                    "x_2": array([3.0, 3.0]),
                },
            ),
            {"greater": True},
        ),
        (5.5, "compute_quantile", (0.5,), {}),
        (5.5, "compute_quartile", (2,), {}),
        (5.5, "compute_percentile", (50,), {}),
        (5.5, "compute_median", (), {}),
        (0.0, "compute_moment", (1,), {}),
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
    assert_allclose(result_1, array([statistic_estimation]), atol=1e-6)
    assert_allclose(result_2, array([statistic_estimation] * 2), atol=1e-6)
    assert result_1.shape == (1,)
    assert result_2.shape == (2,)

    result = getattr(x_1_statistics, statistic_estimator)(
        *statistic_estimator_args, **statistic_estimator_kwargs
    )
    assert_allclose(result["x_1"], array([statistic_estimation]), atol=1e-6)
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
    "value,kwargs", [(0.8, {}), (0.3, {"greater": False}), (0.8, {"greater": True})]
)
def test_compute_joint_probability(statistics, value, kwargs):
    """Check compute_joint_probability()."""
    result = statistics.compute_joint_probability(
        {
            "x_1": array([3.0]),
            "x_2": array([3.0, 3.0]),
        },
        **kwargs,
    )
    result_1 = result["x_1"]
    result_2 = result["x_2"]
    assert_allclose(result_1, value, atol=1e-6)
    assert_allclose(result_2, value, atol=1e-6)
    assert isinstance(result_1, float)
    assert isinstance(result_2, float)


def test_variation_coefficient():
    """Check compute_variation_coefficient()."""

    class NewStatistics(Statistics):
        compute_mean = lambda self: {"x": 2}  # noqa: E731
        compute_standard_deviation = lambda self: {"x": 6}  # noqa: E731

    with concretize_classes(NewStatistics):
        assert NewStatistics(Dataset()).compute_variation_coefficient() == {"x": 3}
