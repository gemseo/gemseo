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
from functools import partial
from math import prod

import pytest
from numpy import array
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo import configuration
from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.problem.database import Database
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.doe.custom_doe.custom_doe import CustomDOE
from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.optimization.power_2 import Power2
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.space.transformation._working import create_working_transformation
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
      \+------\+-------------\+-------\+-------------\+------\+
      \| Name \| Lower bound \| Value \| Upper bound \| Type \|
      \+------\+-------------\+-------\+-------------\+------\+
      \| x\[0\] \|     -inf    \|  None \|     inf     \| real \|
      \| x\[1\] \|     -inf    \|  None \|     inf     \| real \|
      \+------\+-------------\+-------\+-------------\+------\+
Running the algorithm CustomDOE:
    50%\|█████     \| 1\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
   100%\|██████████\| 2\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]$"""
    assert re.match(expected_result, result)


@pytest.mark.usefixtures("restore_configuration_options")
def test_check_desvars_bounds(snapshot):
    """Check that check_desvars_bounds drives the membership check.

    Args:
        snapshot: The snapshot fixture.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)

    evaluation_problem = EvaluationProblem(design_space)
    evaluation_problem.add_observable(ArrayFunction(sum, name="sum"))
    output_functions, _ = evaluation_problem.get_functions(
        observable_names=(), jacobian_names=None
    )
    evaluate = partial(
        evaluation_problem.evaluate_functions,
        array([2.0]),
        input_value_is_normalized=False,
        output_functions=output_functions,
        jacobian_functions=None,
    )

    assert configuration.check_desvars_bounds
    with assert_exception(ValueError, snapshot):
        evaluate()

    configuration.check_desvars_bounds = False
    assert_equal(evaluate()[0]["sum"], array([2.0]))


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


@pytest.mark.parametrize("space_is_a_design_space", [False, True])
def test_recording_a_function_expecting_normalized_inputs(
    space_is_a_design_space,
    snapshot,
) -> None:
    """Check the error when a function expects normalized inputs.

    The evaluation half evaluates a function
    in the coordinates the input space declares,
    whatever that space is,
    and the half normalizing a point wraps it
    instead of handing it a normalized point back;
    evaluating such a function at a point of the space
    would silently return a value that is not the one asked for.
    """
    if space_is_a_design_space:
        space = DesignSpace()
        space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    else:
        space = RandomSpace()
        space.add_variable("x", OTUniformDistribution_Settings())

    function = ArrayFunction(lambda x: array([x[0] ** 2]), name="square")
    function.expects_normalized_inputs = True

    problem = EvaluationProblem(space)
    problem.add_observable(function)
    with assert_exception(ValueError, snapshot):
        problem.bind_functions()


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


def test_bind_functions_finite_differences_on_random_space() -> None:
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
    problem.bind_functions()

    function = problem.observables[0]
    assert_allclose(function.jac(array([2.0])), array([[4.0]]), atol=1e-4)


def test_wrapper_original_is_transitive() -> None:
    """Check that `original` reaches the raw function through stacked wrappers."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=1, lower_bound=0.0, upper_bound=1.0)

    problem = EvaluationProblem(design_space)
    raw_function = ArrayFunction(sum, name="sum")
    problem.add_observable(raw_function)

    chain = create_working_transformation(design_space, normalize=True)
    problem.bind_functions()
    assert problem.observables[0].original is raw_function

    # Stacking a second wrapper must not hide the raw function behind the first one.
    working_problem = problem.create_working_problem(chain)
    assert working_problem.observables[0].original is raw_function


def test_bind_functions_after_database_replaced() -> None:
    """Check that a new database is the only one recording after a rebind."""
    problem = Power2()
    problem.bind_functions()
    old_database = problem.database
    problem.database = Database(input_space=problem.input_space)
    problem.bind_functions()

    problem.evaluate_functions(
        input_value=array([0.5, 0.5, 0.5]), input_value_is_normalized=False
    )

    assert len(old_database) == 0
    assert len(problem.database) == 1


def test_bind_functions_carries_over_the_call_counter(
    enable_function_statistics,
) -> None:
    """Check that rebinding past the problem's own wrapper keeps the counter.

    Rebuilding the wrapper used to start a fresh counter at zero, so a
    second `scenario.execute()` counted only its own calls, even though
    nothing asked for the previous count to be dropped, e.g. two runs with
    no `reset()` in between.
    """
    problem = Power2()
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.5, 0.5, 0.5]), input_value_is_normalized=False
    )
    problem.evaluate_functions(
        input_value=array([0.4, 0.4, 0.4]), input_value_is_normalized=False
    )
    assert problem.objective.n_calls == 2

    # A second run rebinds the functions of the problem.
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.3, 0.3, 0.3]), input_value_is_normalized=False
    )

    assert problem.objective.n_calls == 3


def test_reset_function_calls_false_keeps_counting_across_a_rebind(
    enable_function_statistics,
) -> None:
    """Check that `reset(function_calls=False)` does not zero the counter.

    `problem.reset(function_calls=False)` clears the database without
    resetting the counters, and the count must still carry over once the
    functions are rebound for a second run.
    """
    problem = Power2()
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.5, 0.5, 0.5]), input_value_is_normalized=False
    )
    assert problem.objective.n_calls == 1

    problem.reset(function_calls=False)
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.4, 0.4, 0.4]), input_value_is_normalized=False
    )

    assert problem.objective.n_calls == 2


def test_reset_function_calls_true_still_zeroes_the_counter_across_a_rebind(
    enable_function_statistics,
) -> None:
    """Check that `reset(function_calls=True)` still zeroes the counter.

    The counter is carried over past the problem's own wrapper when it is
    rebuilt, so an explicit reset in between two runs must still be honoured.
    """
    problem = Power2()
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.5, 0.5, 0.5]), input_value_is_normalized=False
    )
    assert problem.objective.n_calls == 1

    problem.reset(function_calls=True)
    problem.bind_functions()
    problem.evaluate_functions(
        input_value=array([0.4, 0.4, 0.4]), input_value_is_normalized=False
    )

    assert problem.objective.n_calls == 1


def test_wrapper_original_stops_at_algebraic_operations() -> None:
    """Check that `original` does not reach through an offset."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=1, lower_bound=0.0, upper_bound=1.0)

    problem = EvaluationProblem(design_space)
    offset_function = ArrayFunction(sum, name="sum") - 1.0
    problem.add_observable(offset_function)

    problem.bind_functions()

    assert problem.observables[0].original is offset_function


@pytest.mark.parametrize(
    "variable_settings",
    [
        {"type_": "integer", "lower_bound": -2, "upper_bound": 2, "value": 1},
        {"lower_bound": 1.0, "upper_bound": 1.0, "value": 1.0},
    ],
    ids=["integer", "coinciding_bounds"],
)
def test_finite_differences_step_of_a_component_without_a_range(
    variable_settings,
) -> None:
    """Check the finite-difference step of a component the space does not normalize.

    The step is taken in the coordinates the algorithm works in
    and no longer scaled by the range of a variable,
    so an integer component and one whose bounds coincide,
    neither of which the working space normalizes,
    keep the step as it is instead of coming out as exactly `0.0`;
    scaling by the range used to divide the approximated Jacobian by zero
    and the run stopped at its starting point on a `NaN`.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=-2.0, upper_bound=2.0, value=1.0)
    design_space.add_variable("y", **variable_settings)

    problem = OptimizationProblem(
        design_space, differentiation_method="finite_differences"
    )
    problem.objective = ArrayFunction(
        lambda x: array([x[0] ** 2 + x[1] ** 2]), name="f"
    )

    result = execute_algo(
        problem,
        settings_model=SLSQP_Settings(max_iter=30, relax_integer_variables=True),
    )

    # The starting point is x=1, which the run must leave.
    assert_allclose(result.x_opt[0], 0.0, atol=1e-6)


@pytest.mark.parametrize("n_processes", [1, 2])
def test_a_run_counts_the_evaluations_of_its_functions(
    n_processes, enable_function_statistics
) -> None:
    """Check that a run whose counters are enabled evaluates and counts.

    A counter is built only for a function counting its calls,
    and a worker gets that function through pickle,
    which brings the count back as a plain integer
    rather than as the shared counter it was;
    a function without a counter to count in
    would stop the worker it is evaluated by.
    Whether the count of a worker reaches the process the user reads it from
    depends on the way the platform starts that worker,
    so a parallel run is checked on the samples it records.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", size=2, lower_bound=0.0, upper_bound=10.0)
    problem = EvaluationProblem(design_space)
    problem.add_observable(ArrayFunction(sum, name="sum"))

    CustomDOE().execute(
        problem,
        settings=CustomDOE_Settings(
            samples=array([[2.0, 3.0], [4.0, 5.0]]), n_processes=n_processes
        ),
    )

    assert_equal(problem.database.get_function_history("sum"), array([5.0, 9.0]))
    if n_processes == 1:
        assert problem.observables[0].n_calls == 2
