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
"""Test the class OptimizationDataset."""

from __future__ import annotations

import pytest
from numpy import arange
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

from gemseo.datasets.dataset import Dataset
from gemseo.datasets.optimization_dataset import OptimizationDataset


@pytest.fixture(scope="module")
def dataset() -> OptimizationDataset:
    """An optimization dataset."""
    dataset = OptimizationDataset()
    dataset.add_design_group(1, "x")
    dataset.add_objective_group(2, "f")
    dataset.add_constraint_group(3, "c")
    dataset.add_observable_group(4, "o")
    return dataset


def test_design_variable_names(dataset):
    """Test the property design_variable_names."""
    assert dataset.design_variable_names == ["x"]


def test_constraint_names(dataset):
    """Test the property constraint_names."""
    assert dataset.constraint_names == ["c"]


def test_objective_names(dataset):
    """Test the property objective_names."""
    assert dataset.objective_names == ["f"]


def test_observable_names(dataset):
    """Test the property observable_names."""
    assert dataset.observable_names == ["o"]


def test_n_iterations(dataset):
    """Test the property n_iterations."""
    assert dataset.n_iterations == 1


def tset_iterations(dataset):
    """Test the property iterations."""
    assert_equal(dataset.iterations, arange(1))


def test_add_design_variable():
    """Test the method add_design_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_design_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.DESIGN_GROUP
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_observable_variable():
    """Test the method add_observable_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_observable_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.OBSERVABLE_GROUP
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_objective_variable():
    """Test the method add_objective_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_objective_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.OBJECTIVE_GROUP
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_constraint_variable():
    """Test the method add_constraint_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_constraint_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.CONSTRAINT_GROUP
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_design_group():
    """Test the method add_design_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_design_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.DESIGN_GROUP, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_objective_group():
    """Test the method add_objective_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_objective_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.OBJECTIVE_GROUP, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_observable_group():
    """Test the method add_observable_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_observable_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.OBSERVABLE_GROUP, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_constraint_group():
    """Test the method add_constraint_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_constraint_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.CONSTRAINT_GROUP, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_design_dataset(dataset):
    """Test the property design_dataset."""
    design_dataset = dataset.get_view(group_names=dataset.DESIGN_GROUP)
    assert_frame_equal(dataset.design_dataset, design_dataset)


def test_objective_dataset(dataset):
    """Test the property objective_dataset."""
    objective_dataset = dataset.get_view(group_names=dataset.OBJECTIVE_GROUP)
    assert_frame_equal(dataset.objective_dataset, objective_dataset)


def test_observable_dataset(dataset):
    """Test the property observable_dataset."""
    observable_dataset = dataset.get_view(group_names=dataset.OBSERVABLE_GROUP)
    assert_frame_equal(dataset.observable_dataset, observable_dataset)


def test_constraint_dataset(dataset):
    """Test the property constraint_dataset."""
    constraint_dataset = dataset.get_view(group_names=dataset.CONSTRAINT_GROUP)
    assert_frame_equal(dataset.constraint_dataset, constraint_dataset)


def test_iterations(dataset):
    """Test the property iterations."""
    assert_equal(dataset.iterations, [1])
