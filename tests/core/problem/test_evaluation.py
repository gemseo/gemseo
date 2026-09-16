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
from math import prod

from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.doe.custom_doe.custom_doe import CustomDOE
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception


def test_default(caplog):
    """Check that a DOELibrary can handle an EvaluationProblem."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2)

    evaluation_problem = EvaluationProblem(design_space)
    evaluation_problem.add_observable(ArrayFunction(sum, name="sum"))
    evaluation_problem.add_observable(ArrayFunction(prod, name="prod"))

    custom_doe = CustomDOE()
    custom_doe.execute(
        evaluation_problem,
        settings=CustomDOE_Settings(samples=array([[2.0, 3.0], [4.0, 5.0]])),
    )

    get_function_history = evaluation_problem.database.get_function_history
    assert_equal(get_function_history("sum"), array([5.0, 9.0]))
    assert_equal(get_function_history("prod"), array([6.0, 20.0]))
    result = "\n".join([line[2] for line in caplog.record_tuples])
    expected_result = r"""^Evaluation problem:
   Evaluate the functions: prod, sum
   over the design space:
      \+------\+-------------\+-------\+-------------\+-------\+
      \| Name \| Lower bound \| Value \| Upper bound \| Type  \|
      \+------\+-------------\+-------\+-------------\+-------\+
      \| x\[0\] \|     -inf    \|  None \|     inf     \| float \|
      \| x\[1\] \|     -inf    \|  None \|     inf     \| float \|
      \+------\+-------------\+-------\+-------------\+-------\+
Running the algorithm CustomDOE:
    50%\|█████     \| 1\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
   100%\|██████████\| 2\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]$"""
    assert re.match(expected_result, result)


def test_set_database():
    """Test the setter `database`."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=1)

    evaluation_problem = EvaluationProblem(design_space)
    evaluation_problem.add_observable(ArrayFunction(sum, name="sum"))
    evaluation_problem.add_observable(ArrayFunction(prod, name="prod"))

    database = Database(name="test_database", input_space=design_space)
    database.store(array([0.0]), array([1.0, 1.0]))

    assert evaluation_problem.database.n_iterations == 0
    evaluation_problem.database = database
    assert evaluation_problem.database.n_iterations == 1
    assert evaluation_problem.database.name == "test_database"


def test_evaluate_functions_without_input_value_on_random_space(snapshot) -> None:
    """Check the error when no input value is passed for a RandomSpace."""
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    problem = EvaluationProblem(random_space)
    problem.add_observable(ArrayFunction(sum, name="sum"))

    with assert_exception(ValueError, snapshot):
        problem.evaluate_functions(output_functions=problem.observables)


def test_evaluate_functions_with_input_value_on_random_space() -> None:
    """Check the evaluation of the functions of a problem based on a RandomSpace.

    Membership and normalization are notions of a
    [DesignSpace][gemseo.space.design.DesignSpace],
    so the input value passed explicitly is used as is.
    """
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    problem = EvaluationProblem(random_space)
    problem.add_observable(ArrayFunction(sum, name="sum"))

    output_values, _ = problem.evaluate_functions(
        array([0.5]),
        input_value_is_normalized=False,
        output_functions=problem.observables,
    )

    assert_allclose(output_values["sum"], 0.5)


def test_evaluate_functions_with_normalized_input_value_on_random_space(
    snapshot,
) -> None:
    """Check the error when a normalized input value is passed for a RandomSpace."""
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    problem = EvaluationProblem(random_space)
    problem.add_observable(ArrayFunction(sum, name="sum"))

    with assert_exception(ValueError, snapshot):
        problem.evaluate_functions(array([0.5]), output_functions=problem.observables)


def test_preprocess_function_expecting_normalized_inputs_on_random_space(
    snapshot,
) -> None:
    """Check the error when a function expects normalized inputs on a RandomSpace.

    Normalization is a notion of a
    [DesignSpace][gemseo.space.design.DesignSpace],
    so such a function cannot be fed with the inputs it expects
    when the input space is not one;
    evaluating it at a point of the space instead
    would return a value that is not the one asked for.
    """
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    function = ArrayFunction(lambda x: array([x[0] ** 2]), name="square")
    function.expects_normalized_inputs = True

    problem = EvaluationProblem(random_space)
    problem.add_observable(function)
    with assert_exception(ValueError, snapshot):
        problem.preprocess_functions(is_function_input_normalized=False)


def test_reset_on_random_space() -> None:
    """Check EvaluationProblem.reset with its default arguments on a RandomSpace.

    A [RandomSpace][gemseo.space.random.RandomSpace] defines no current value,
    so it has nothing to restore;
    resetting the input space of a problem based on such a space
    must be a no-op instead of raising.
    """
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    problem = EvaluationProblem(random_space)
    problem.add_observable(ArrayFunction(sum, name="sum"))
    problem.database.store(array([1.0]), {"sum": array([1.0])})
    problem.evaluation_counter.current = 1
    problem.evaluation_counter.enabled = True

    problem.reset()

    assert len(problem.database) == 0
    assert problem.evaluation_counter.current == 0
    assert problem.evaluation_counter.enabled is False
    assert problem.input_space is random_space
    assert list(problem.input_space) == ["x"]


def test_reset_on_partially_valued_design_space() -> None:
    """Check EvaluationProblem.reset on a design space with a partial current value.

    The variables without a value keep their `None` marker in the stored mapping,
    so the partial current value is restored as it was.
    """
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("y")
    design_space.set_current_variable("x", array([1.0]))

    problem = EvaluationProblem(design_space)
    design_space.set_current_variable("x", array([2.0]))
    design_space.set_current_variable("y", array([3.0]))

    problem.reset()

    assert not design_space.has_current_value
    assert_equal(design_space._current_value["x"], array([1.0]))
    assert design_space._current_value["y"] is None


def test_reset_restores_the_current_value_of_the_design_space() -> None:
    """Check that EvaluationProblem.reset restores the initial current value."""
    design_space = DesignSpace()
    design_space.add_variable("x", value=1.0)

    problem = EvaluationProblem(design_space)
    design_space.set_current_variable("x", array([2.0]))

    problem.reset()

    assert_equal(design_space.get_current_value(), array([1.0]))


def test_preprocess_functions_finite_differences_on_random_space() -> None:
    """Check that the Jacobian is approximated by finite differences over a
    RandomSpace.

    A [RandomSpace][gemseo.space.random.RandomSpace] has no bounds,
    so the gradient approximator must be created
    without a design space to bound the perturbations.
    """
    random_space = RandomSpace()
    random_space.add_variable("x", OTUniformDistribution_Settings())

    problem = EvaluationProblem(
        random_space, differentiation_method="finite_differences"
    )
    problem.add_observable(ArrayFunction(lambda x: x**2, name="f"))
    problem.preprocess_functions(is_function_input_normalized=False)

    function = problem.observables[0]
    assert_allclose(function.jac(array([2.0])), array([[4.0]]), atol=1e-4)
