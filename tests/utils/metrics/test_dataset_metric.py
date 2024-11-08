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
from __future__ import annotations

import re

import pytest
from numpy import array
from numpy import zeros
from numpy.testing import assert_almost_equal

from gemseo.datasets.dataset import Dataset
from gemseo.utils.metrics.dataset_metric import DatasetMetric
from gemseo.utils.metrics.mean_metric import MeanMetric
from gemseo.utils.metrics.squared_error_metric import SquaredErrorMetric

dataset_duplicate_variable_name = Dataset.from_array(
    array([[1, 5, 10], [2, 6, 11]]),
    variable_names=["a", "d", "b"],
    variable_names_to_group_names={"a": "outputs", "d": "inputs", "b": "outputs"},
)
dataset_duplicate_variable_name.add_variable("d", data=[10, 11], group_name="outputs")


def test_duplicate_variable_name():
    """Check that a dataset having a variable belonging to more than one group raises an
    error."""
    dm = DatasetMetric(SquaredErrorMetric())
    a = dataset_duplicate_variable_name
    b = a.copy()
    with pytest.raises(
        ValueError,
        match=re.escape("A variable cannot belong to more than one group."),
    ):
        assert MeanMetric(dm).compute(a, b) == pytest.approx(4)


@pytest.mark.parametrize(
    ("a", "group_name"),
    [
        (Dataset.from_array(array([[1, 5], [2, 6]]), variable_names=["a", "b"]), ()),
        (
            Dataset.from_array(
                array([[1, 5, 10, 15], [2, 6, 11, 16]]),
                variable_names=["a", "b", "c"],
                variable_names_to_n_components={"a": 1, "b": 1, "c": 2},
            ),
            (),
        ),
    ],
)
def test_mse_dataset_metric(a, group_name):
    """Check the MSE metric on datasets.

    The case of both scalar and vector elements is considered.
    """

    b = a.copy()

    def f(x):
        return x + 2

    b.transform_data(f)

    dm = DatasetMetric(
        SquaredErrorMetric(), group_names=group_name, variable_names=a.variable_names
    )
    assert MeanMetric(dm).compute(a, b) == pytest.approx(4)

    for name in ["a", "b"]:
        assert_almost_equal(
            dm.compute(a, b).get_view(variable_names=name).to_numpy().ravel(),
            array([4, 4]),
        )

    if "c" in a.variable_names:
        assert_almost_equal(
            dm.compute(a, b).get_view(variable_names="c").to_numpy().ravel(),
            array([4, 4, 4, 4]),
        )


@pytest.mark.parametrize(
    ("group_names", "variable_names", "indices", "components", "expected_shape"),
    [
        ((), (), (), (), (2, 4)),
        (["inputs", "outputs"], (), (), (), (2, 4)),
        ("inputs", (), (), (), (2, 2)),
        (["inputs"], (), (), (), (2, 2)),
        ((), "a", (), (), (2, 1)),
        ((), ["a"], (), (), (2, 1)),
        ((), "c", (), 0, (2, 1)),
        ((), "c", (), [0], (2, 1)),
        ((), (), 0, (), (1, 4)),
        ((), (), [0], (), (1, 4)),
    ],
)
def test_sub_selection(
    group_names, variable_names, indices, components, expected_shape
):
    """Check that variables and rows to which metric is computed can be selected.

    Variable selection can be done by group names, variable names, components. Rows can
    be selected by indices.
    """
    dataset = Dataset.from_array(
        array([[1, 5, 10, 15], [2, 6, 11, 16]]),
        variable_names=["a", "b", "c"],
        variable_names_to_group_names={"a": "inputs", "b": "inputs", "c": "outputs"},
        variable_names_to_n_components={"a": 1, "b": 1, "c": 2},
    )
    dm = DatasetMetric(
        SquaredErrorMetric(),
        group_names=group_names,
        variable_names=variable_names,
        components=components,
        indices=indices,
    )
    assert_almost_equal(dm.compute(dataset, dataset), zeros(expected_shape))
