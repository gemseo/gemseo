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
"""Test linear combination discipline."""

from __future__ import annotations

import pytest
from numpy import array
from numpy import eye
from numpy import ndarray
from numpy import zeros
from numpy.testing import assert_equal

from gemseo.core.function.discipline_adapter_generator import DisciplineAdapterGenerator
from gemseo.discipline.linear_combination import LinearCombination
from gemseo.doe.custom_doe.custom_doe import CustomDOE
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.optimization.problem import OptimizationProblem
from gemseo.space.design import DesignSpace
from gemseo.util.comparison import compare_dict_of_arrays
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker


def test_linear_combination_execution(linear_combination) -> None:
    """Test  linear combination discipline execution."""
    output_data = linear_combination.execute({
        "alpha": array([1.0]),
        "beta": array([1.0]),
    })
    assert all(output_data["delta"] == array([-3.0]))


def test_linear_combination_execution2points(linear_combination) -> None:
    """Test  linear combination discipline execution."""
    output_data = linear_combination.execute({
        "alpha": array([1.0, 0.0]),
        "beta": array([1.0, -1.0]),
    })
    assert all(output_data["delta"] == array([-3.0, 0.0]))


def test_check_gradient(linear_combination) -> None:
    """Test jacobian computation by finite differences."""
    linear_combination.io.input_grammar.defaults = {
        "alpha": array([1.0]),
        "beta": array([1.0]),
    }
    checker = DisciplineJacobianChecker(linear_combination)
    assert checker.check(atol=1e-3, rtol=1e-3, step=1e-4)


def test_check_gradient2points(linear_combination) -> None:
    """Test jacobian computation by finite differences."""
    linear_combination.io.input_grammar.defaults = {
        "alpha": array([1.0, 0.0]),
        "beta": array([1.0, -1.0]),
    }
    checker = DisciplineJacobianChecker(linear_combination)
    assert checker.check(atol=1e-3, rtol=1e-3, step=1e-4)


def test_parallel_doe_execution(linear_combination) -> None:
    """Test parallel execution."""
    custom_doe = CustomDOE()
    design_space = DesignSpace()
    design_space.add_real_variable(
        "alpha", lower_bound=-1.0, upper_bound=1.0, value=0.0
    )
    design_space.add_real_variable("beta", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    opt_problem = OptimizationProblem(design_space)
    opt_problem.objective = DisciplineAdapterGenerator(linear_combination).get_function(
        input_names=["alpha", "beta"],
        output_names=["delta"],
        default_input_data={
            "alpha": array([1.0]),
            "beta": array([1.0]),
        },
    )
    custom_doe.execute(
        opt_problem,
        settings=CustomDOE_Settings(
            samples=array([[1.0, 0.0], [1.0, -1.0]]).T, eval_jac=True, n_processes=2
        ),
    )
    v = opt_problem.database.get_function_value("delta", array([1.0, 1.0]))
    assert v == array([-3.0])


def test_default_values(linear_combination) -> None:
    """Check that all the input variables have zero as default value."""
    assert compare_dict_of_arrays(
        linear_combination.io.input_grammar.defaults,
        {"alpha": zeros(1), "beta": zeros(1)},
    )


def test_linearization_after_execution_cache_none(linear_combination):
    linear_combination.set_cache(linear_combination.CacheType.NONE)
    input_data = {"alpha": array([1.0]), "beta": array([1.0])}
    linear_combination.execute(input_data)
    assert not linear_combination.linearize(input_data)


@pytest.mark.parametrize(
    ("average", "input_coefficients", "expected"),
    [
        (False, {}, {"alpha": 1.0, "beta": 1.0}),
        (True, {}, {"alpha": 0.5, "beta": 0.5}),
        (True, {"alpha": 0.3}, {"alpha": 0.3, "beta": 1.0}),
        (False, {"alpha": 0.3, "gamma": 1.0}, {"alpha": 0.3, "beta": 1.0}),
    ],
    ids=["default", "average", "average_with_coefficient", "unknown_coefficient"],
)
def test_average(average, input_coefficients, expected):
    """Check average."""
    linear_combination = LinearCombination(
        input_names=["alpha", "beta"],
        output_name="delta",
        input_coefficients=input_coefficients,
        average=average,
    )
    assert linear_combination._LinearCombination__coefficients == expected


@pytest.fixture
def namespaced_discipline() -> LinearCombination:
    """A LinearCombination with a namespaced input and a namespaced output."""
    discipline = LinearCombination(
        ["a", "b"], "y", input_coefficients={"a": 2.0, "b": 3.0}
    )
    discipline.add_namespace_to_input("a", "ns")
    discipline.add_namespace_to_output("y", "ns")
    return discipline


@pytest.fixture
def namespaced_input_data() -> dict[str, ndarray]:
    """The input data of the namespaced discipline."""
    return {"ns:a": array([1.0, 2.0]), "b": array([10.0, 20.0])}


def test_namespaces(namespaced_discipline, namespaced_input_data):
    """Check the output with a namespaced input and a namespaced output."""
    output_data = namespaced_discipline.execute(namespaced_input_data)
    assert_equal(output_data["ns:y"], array([32.0, 64.0]))


def test_namespaces_jacobian(namespaced_discipline, namespaced_input_data):
    """Check the Jacobian with a namespaced input and a namespaced output."""
    jac = namespaced_discipline.linearize(
        namespaced_input_data, compute_all_jacobians=True
    )
    assert set(jac["ns:y"]) == {"ns:a", "b"}
    assert_equal(jac["ns:y"]["ns:a"].toarray(), 2.0 * eye(2))
    assert_equal(jac["ns:y"]["b"].toarray(), 3.0 * eye(2))
