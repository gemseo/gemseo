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
from numpy import array
from numpy import nan
from numpy.testing import assert_equal
from pandas.testing import assert_frame_equal

from gemseo.dataset.dataset import Dataset
from gemseo.dataset.optimization_dataset import OptimizationDataset


@pytest.fixture(scope="module")
def dataset() -> OptimizationDataset:
    """An optimization dataset."""
    dataset = OptimizationDataset()
    dataset.add_design_group(1, "x")
    dataset.add_objective_group(2, "f")
    dataset.add_observable_group(4, "o")
    dataset.add_equality_constraint_group(5, "eq")
    dataset.add_inequality_constraint_group(6, "ineq")
    return dataset


def test_design_variable_names(dataset) -> None:
    """Test the property design_variable_names."""
    assert dataset.design_variable_names == ["x"]


def test_eq_constraint_names(dataset) -> None:
    """Test the property equality_constraint_names."""
    assert dataset.equality_constraint_names == ["eq"]


def test_ineq_constraint_names(dataset) -> None:
    """Test the property inequality_constraint_names."""
    assert dataset.inequality_constraint_names == ["ineq"]


def test_objective_names(dataset) -> None:
    """Test the property objective_names."""
    assert dataset.objective_names == ["f"]


def test_observable_names(dataset) -> None:
    """Test the property observable_names."""
    assert dataset.observable_names == ["o"]


def test_n_iterations(dataset) -> None:
    """Test the property n_iterations."""
    assert dataset.n_iterations == 1


def tset_iterations(dataset) -> None:
    """Test the property iterations."""
    assert_equal(dataset.iterations, arange(1))


def test_add_design_variable() -> None:
    """Test the method add_design_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_design_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.design_group
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_observable_variable() -> None:
    """Test the method add_observable_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_observable_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.observable_group
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_objective_variable() -> None:
    """Test the method add_objective_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_objective_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.objective_group
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_equality_constraint_variable() -> None:
    """Test the method add_eq_constraint_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_equality_constraint_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.equality_constraint_group
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_inequality_constraint_variable() -> None:
    """Test the method add_ineq_constraint_variable."""
    o_dataset = OptimizationDataset()
    o_dataset.add_inequality_constraint_variable("x", [[1.0], [2.0]])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_variable(
        "x", [[1.0], [2.0]], group_name=OptimizationDataset.inequality_constraint_group
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_design_group() -> None:
    """Test the method add_design_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_design_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.design_group, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_objective_group() -> None:
    """Test the method add_objective_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_objective_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.objective_group, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_observable_group() -> None:
    """Test the method add_observable_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_observable_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(OptimizationDataset.observable_group, [[1.0], [2.0]], ["x"])
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_equality_constraint_group() -> None:
    """Test the method add_eq_constraint_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_equality_constraint_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(
        OptimizationDataset.equality_constraint_group, [[1.0], [2.0]], ["x"]
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_add_inequality_constraint_group() -> None:
    """Test the method add_ineq_constraint_group."""
    o_dataset = OptimizationDataset()
    o_dataset.add_inequality_constraint_group([[1.0], [2.0]], ["x"])

    dataset = Dataset()
    dataset.name = o_dataset.__class__.__name__
    dataset.add_group(
        OptimizationDataset.inequality_constraint_group, [[1.0], [2.0]], ["x"]
    )
    dataset.index = arange(1, len(dataset) + 1)

    assert_frame_equal(o_dataset, dataset)


def test_design_dataset(dataset) -> None:
    """Test the property design_dataset."""
    design_dataset = dataset.get_view(group_names=dataset.design_group)
    assert_frame_equal(dataset.design_dataset, design_dataset)


def test_objective_dataset(dataset) -> None:
    """Test the property objective_dataset."""
    objective_dataset = dataset.get_view(group_names=dataset.objective_group)
    assert_frame_equal(dataset.objective_dataset, objective_dataset)


def test_observable_dataset(dataset) -> None:
    """Test the property observable_dataset."""
    observable_dataset = dataset.get_view(group_names=dataset.observable_group)
    assert_frame_equal(dataset.observable_dataset, observable_dataset)


def test_equality_constraint_dataset(dataset) -> None:
    """Test the property equality_constraint_dataset."""
    equality_constraint_dataset = dataset.get_view(
        group_names=dataset.equality_constraint_group
    )
    assert_frame_equal(dataset.equality_constraint_dataset, equality_constraint_dataset)


def test_inequality_constraint_dataset(dataset) -> None:
    """Test the property inequality_constraint_dataset."""
    inequality_constraint_dataset = dataset.get_view(
        group_names=dataset.inequality_constraint_group
    )
    assert_frame_equal(
        dataset.inequality_constraint_dataset, inequality_constraint_dataset
    )


def test_iterations(dataset) -> None:
    """Test the property iterations."""
    assert_equal(dataset.iterations, [1])


def test_get_best_iter_history(optim_data) -> None:
    """Test the method get_best_iter_history."""
    dv, obj, eq, ineq = optim_data

    dataset = OptimizationDataset()
    best_iteration_history = dataset.get_best_iteration_history()
    assert_equal(best_iteration_history, [])

    dataset.add_design_group(dv, "x")

    best_iteration_history = dataset.get_best_iteration_history()
    assert_equal(best_iteration_history, [])

    dataset.add_objective_group(obj, "obj")
    best_iteration_history = dataset.get_best_iteration_history()
    assert_equal(best_iteration_history, [1, 2, 3, 3, 3, 6, 6])

    dataset.add_equality_constraint_group(
        eq, ("ec1", "ec2"), variable_name_to_n_components={"ec1": 1, "ec2": 2}
    )
    best_iteration_history = dataset.get_best_iteration_history()
    assert_equal(best_iteration_history, [1, 2, 2, 4, 5, 6, 6])

    dataset.add_inequality_constraint_group(
        ineq, ("ic1", "ic2"), variable_name_to_n_components={"ic1": 1, "ic2": 2}
    )
    best_iteration_history = dataset.get_best_iteration_history()
    assert_equal(best_iteration_history, [1, 2, 2, 2, 2, 2, 7])

    dataset_2 = dataset.get_view(
        group_names=(
            dataset.design_group,
            dataset.objective_group,
            dataset.inequality_constraint_group,
        )
    )

    best_iteration_history = dataset_2.get_best_iteration_history()
    assert_equal(best_iteration_history, [1, 2, 3, 3, 3, 3, 3])


@pytest.mark.parametrize(
    ("add_constraint_group", "feasible_value"),
    [
        ("add_inequality_constraint_group", -1.0),
        ("add_equality_constraint_group", 0.0),
    ],
)
@pytest.mark.parametrize("component", [0, 1, None])
@pytest.mark.parametrize(
    ("iteration", "expected"),
    [(1, [1, 2, 3]), (2, [1, 1, 3]), (3, [1, 1, 1])],
)
def test_get_best_iter_history_nan_constraint(
    add_constraint_group, feasible_value, iteration, component, expected
) -> None:
    """Check that an iteration whose constraints contain NaN is infeasible.

    Its violation is infinite,
    so it is never preferred as least infeasible iteration,
    and the first iteration is recorded
    even when all its constraints are NaN.
    """
    constraint_values = array([
        [feasible_value, feasible_value],
        [0.5, feasible_value],
        [feasible_value, feasible_value],
    ])
    # component is None for the "full NaN" case,
    # i.e. all the constraint components of the iteration are NaN.
    nan_components = slice(None) if component is None else component
    constraint_values[iteration - 1, nan_components] = nan
    dataset = OptimizationDataset()
    dataset.add_design_group(array([[0.0], [1.0], [3.0]]), "x")
    dataset.add_objective_group(array([[5.0], [1.0], [3.0]]), "f")
    getattr(dataset, add_constraint_group)(constraint_values, ("g1", "g2"))
    assert_equal(dataset.get_best_iteration_history(), expected)


@pytest.mark.parametrize(
    ("objective_values", "constraint_values", "expected"),
    [
        ([nan, 3.0, 1.0], [-1.0, -1.0, -1.0], [1, 2, 3]),
        ([5.0, nan, 1.0], [1.0, -1.0, -1.0], [1, 2, 3]),
        ([nan, 3.0, 1.0], None, [1, 2, 3]),
    ],
    ids=[
        "feasible_first",
        "after_infeasible",
        "unconstrained_first",
    ],
)
def test_get_best_iter_history_nan_objective(
    objective_values, constraint_values, expected
) -> None:
    """Check that a NaN objective is considered as the worst objective value."""
    dataset = OptimizationDataset()
    dataset.add_design_group(array([[0.0], [1.0], [2.0]]), "x")
    dataset.add_objective_group(array(objective_values).reshape(-1, 1), "f")
    if constraint_values is not None:
        dataset.add_inequality_constraint_group(
            array(constraint_values).reshape(-1, 1), "g"
        )
    assert_equal(dataset.get_best_iteration_history(), expected)
