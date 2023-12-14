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

from numpy import array
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.lib_custom import CustomDOE
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_discipline_adapter_generator import (
    MDODisciplineAdapterGenerator,
)
from gemseo.utils.comparisons import compare_dict_of_arrays


def test_linear_combination_execution(linear_combination):
    """Test  linear combination discipline execution."""
    output_data = linear_combination.execute({
        "alpha": array([1.0]),
        "beta": array([1.0]),
    })
    assert all(output_data["delta"] == array([-3.0]))


def test_linear_combination_execution2points(linear_combination):
    """Test  linear combination discipline execution."""
    output_data = linear_combination.execute({
        "alpha": array([1.0, 0.0]),
        "beta": array([1.0, -1.0]),
    })
    assert all(output_data["delta"] == array([-3.0, 0.0]))


def test_check_gradient(linear_combination):
    """Test jacobian computation by finite differences."""
    linear_combination.default_inputs = {
        "alpha": array([1.0]),
        "beta": array([1.0]),
    }
    assert linear_combination.check_jacobian(threshold=1e-3, step=1e-4)


def test_check_gradient2points(linear_combination):
    """Test jacobian computation by finite differences."""
    linear_combination.default_inputs = {
        "alpha": array([1.0, 0.0]),
        "beta": array([1.0, -1.0]),
    }
    assert linear_combination.check_jacobian(threshold=1e-3, step=1e-4)


def test_parallel_doe_execution(linear_combination):
    """Test parallel execution."""
    custom_doe = CustomDOE()
    design_space = DesignSpace()
    design_space.add_variable("alpha", l_b=-1.0, u_b=1.0, value=0.0)
    design_space.add_variable("beta", l_b=-1.0, u_b=1.0, value=0.0)
    opt_problem = OptimizationProblem(design_space)
    opt_problem.objective = MDODisciplineAdapterGenerator(
        linear_combination
    ).get_function(
        input_names=["alpha", "beta"],
        output_names=["delta"],
        default_inputs={
            "alpha": array([1.0]),
            "beta": array([1.0]),
        },
    )
    custom_doe.execute(
        problem=opt_problem,
        samples=array([[1.0, 0.0], [1.0, -1.0]]).T,
        eval_jac=True,
        n_processes=2,
    )
    v = opt_problem.database.get_function_value("delta", array([1.0, 1.0]))
    assert v == array([-3.0])


def test_default_values(linear_combination):
    """Check that all the input variables have zero as default value."""
    assert compare_dict_of_arrays(
        linear_combination.default_inputs,
        {"alpha": zeros(1), "beta": zeros(1)},
    )
