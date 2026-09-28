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

import warnings
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import isnan
from numpy import nan
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.evaluation_function import EvaluationFunction
from gemseo.core.function.linear_function import LinearFunction
from gemseo.core.function.transformed_input_function import TransformedInputFunction
from gemseo.core.problem.database import Database
from gemseo.core.problem.evaluation import EvaluationProblem
from gemseo.core.problem.termination_criterion import DesvarIsNan
from gemseo.core.problem.termination_criterion import FunctionIsNan
from gemseo.optimization.problem import OptimizationProblem
from gemseo.optimization.scipy_local.scipy_local import ScipyOpt
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.space.design import DesignSpace
from gemseo.space.transformation._working import create_working_transformation
from gemseo.space.transformation.base import BaseSpaceTransformation
from gemseo.space.transformation.composition import SpaceComposition
from gemseo.space.transformation.normalization import SpaceNormalization
from gemseo.util.testing.helper import assert_exception
from tests.space.transformation.test_transformation import FreezingTransformation

if TYPE_CHECKING:
    from gemseo.util.typing import NumberArray


@pytest.fixture
def design_space() -> DesignSpace:
    """A design space with one variable of two components."""
    space = DesignSpace()
    space.add_variable(
        "x", size=2, lower_bound=0.0, upper_bound=10.0, value=array([1.0, 2.0])
    )
    return space


@pytest.fixture
def problem(design_space) -> EvaluationProblem:
    """A problem with a single observable."""
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        ArrayFunction(
            lambda x: array([x @ x]), name="f", jac=lambda x: array([2.0 * x])
        )
    )
    return problem


@pytest.fixture
def transformation(design_space) -> SpaceComposition:
    """The normalization of the design space."""
    return SpaceComposition(design_space, SpaceNormalization)


def test_bind_functions_binds_the_functions(problem) -> None:
    """Check that recording binds the functions to the database."""
    problem.bind_functions()

    observable = problem.observables[0]
    assert isinstance(observable, EvaluationFunction)

    observable.func(array([1.0, 2.0]))
    assert len(problem.database) == 1


def test_bind_functions_is_idempotent(problem) -> None:
    """Check that a second call rebuilds rather than stacking a second wrapper."""
    problem.bind_functions()
    first = problem.observables[0]
    problem.bind_functions()
    second = problem.observables[0]

    assert second is not first
    assert second.original is first.original
    assert not isinstance(second.original, EvaluationFunction)


def test_bind_functions_keeps_the_wrapper_of_another_problem(
    problem, design_space
) -> None:
    """Check that a sub-problem does not discard the binding of its parent.

    A sub-problem built on a function of a recorded problem
    carries that problem's wrapper,
    whose evaluations have to reach that problem's database too;
    unwrapping any `EvaluationFunction` dropped that binding,
    so the parent's database stopped receiving those evaluations.
    """
    problem.bind_functions()

    sub_problem = EvaluationProblem(design_space)
    sub_problem.add_observable(problem.observables[0])
    sub_problem.bind_functions()

    sub_problem.observables[0].func(array([1.0, 2.0]))

    assert len(sub_problem.database) == 1
    assert len(problem.database) == 1


def test_bind_functions_keeps_the_wrapper_of_another_problem_on_rebinding(
    problem, design_space
) -> None:
    """Check that rebinding a second time still keeps the parent's wrapper.

    Calling `bind_functions()` again rebuilds the wrapper from the function it
    installed itself the first time, so it must unwrap exactly the one level
    it added, through `_wrapped_function`, and not through `original`, which
    is transitive and reaches past the parent's wrapper straight to the raw
    function; that dropped the forwarding to the parent's database that the
    first call preserved.
    """
    problem.bind_functions()

    sub_problem = EvaluationProblem(design_space)
    sub_problem.add_observable(problem.observables[0])
    sub_problem.bind_functions()
    sub_problem.bind_functions()

    sub_problem.observables[0].func(array([1.0, 2.0]))

    assert len(sub_problem.database) == 1
    assert len(problem.database) == 1


def test_bind_functions_without_database(problem) -> None:
    """Check that `use_database=False` disables the store and nothing else."""
    problem.bind_functions(use_database=False)

    problem.observables[0].func(array([1.0, 2.0]))

    assert len(problem.database) == 0


def test_bind_functions_without_database_stops_a_nan_point(
    design_space, snapshot
) -> None:
    """Check that a NaN design point stops a problem keeping no database.

    The input is checked whether or not the evaluations are recorded,
    so the function the user wrote is never handed a point holding a NaN,
    where it used to be evaluated at such a point when no database was kept.
    """
    calls = []
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        ArrayFunction(lambda x: (calls.append(x), array([0.0]))[1], name="f")
    )
    problem.bind_functions(use_database=False)

    with assert_exception(DesvarIsNan, snapshot):
        problem.observables[0].evaluate(array([nan, 2.0]))

    assert not calls
    assert len(problem.database) == 0


def test_bind_functions_original_stops_at_a_wrapper_of_another_problem(
    design_space,
) -> None:
    """Check that `get_originals` stops at a wrapper of another problem.

    A sub-algorithm rebuilding an inner problem from `get_originals`
    must get that wrapper back,
    so that the inner evaluations reach the database of the other problem.
    """
    top_problem = OptimizationProblem(design_space)
    top_problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f")
    top_problem.add_constraint(
        ArrayFunction(lambda x: array([x[0]]), name="g"), constraint_type="ineq"
    )
    top_problem.bind_functions()
    bound_constraint = top_problem.constraints[0]
    assert isinstance(bound_constraint, EvaluationFunction)

    sub_problem = OptimizationProblem(design_space)
    sub_problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f2")
    sub_problem.constraints.append(bound_constraint)
    sub_problem.bind_functions()

    assert next(iter(sub_problem.constraints.get_originals())) is bound_constraint


def test_transform_keeps_a_linear_original_reachable(
    design_space, transformation
) -> None:
    """Check that `original` reaches the user's function through both halves.

    `ScipyLinprog` and `ScipyMILP` read the coefficients
    of the `LinearFunction` behind `working_problem.objective.original`.
    """
    problem = OptimizationProblem(design_space)
    objective = LinearFunction(array([[1.0, 1.0]]), "f", input_names=["x"])
    problem.objective = objective
    problem.bind_functions()

    working_problem = problem.create_working_problem(transformation)

    assert isinstance(working_problem.objective, TransformedInputFunction)
    assert working_problem.objective.original is objective


def test_transform_leaves_the_original_problem_alone(problem, transformation) -> None:
    """Check the invariant the story exists for."""
    problem.bind_functions()
    functions_before = list(problem.observables)
    space_before = problem.input_space

    problem.create_working_problem(transformation)

    assert list(problem.observables) == functions_before
    assert problem.input_space is space_before


def test_transform_returns_a_problem_over_the_working_space(
    problem, transformation
) -> None:
    """Check that the working problem works in the working coordinates."""
    problem.bind_functions()
    working_problem = problem.create_working_problem(transformation)

    assert working_problem is not problem
    assert isinstance(working_problem.observables[0], TransformedInputFunction)
    assert_allclose(working_problem.input_space.get_upper_bounds(), array([1.0, 1.0]))


def test_transform_shares_the_database(problem, transformation) -> None:
    """Check that the working problem records into the database of the original."""
    problem.bind_functions()
    working_problem = problem.create_working_problem(transformation)

    assert working_problem.database is problem.database

    # The algorithm hands a working point; the database is keyed on the original one.
    working_problem.observables[0].func(array([0.1, 0.2]))

    assert len(problem.database) == 1
    assert_allclose(
        problem.database.get_function_value("f", array([1.0, 2.0])), array([5.0])
    )


def test_transform_keeps_the_original_reachable(problem, transformation) -> None:
    """Check that `original` reaches the user's function through both wrappers."""
    user_function = problem.observables[0]
    problem.bind_functions()
    working_problem = problem.create_working_problem(transformation)

    assert working_problem.observables[0].original is user_function


def test_transform_an_optimization_problem(design_space, transformation) -> None:
    """Check that the objective and the constraints are carried over."""
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f")
    problem.add_constraint(
        ArrayFunction(lambda x: array([x[0]]), name="g"), constraint_type="ineq"
    )
    problem.bind_functions()

    working_problem = problem.create_working_problem(transformation)

    assert isinstance(working_problem, OptimizationProblem)
    assert isinstance(working_problem.objective, TransformedInputFunction)
    assert len(working_problem.constraints) == 1
    assert isinstance(working_problem.constraints[0], TransformedInputFunction)


def test_transform_does_not_format_the_constraints_twice(
    design_space, transformation
) -> None:
    """Check that an offset constraint is not offset a second time.

    `add_constraint` formats a constraint at add time,
    so the working problem is built through the collections rather than through it.
    """
    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f")
    problem.add_constraint(
        ArrayFunction(lambda x: array([x[0]]), name="g"),
        constraint_type="ineq",
        value=3.0,
    )
    problem.bind_functions()
    original_value = problem.constraints[0].func(array([1.0, 2.0]))

    working_problem = problem.create_working_problem(transformation)
    working_value = working_problem.constraints[0].func(array([0.1, 0.2]))

    assert_allclose(working_value, original_value)


def test_transform_preserves_the_concrete_problem_class(transformation) -> None:
    """Check that a problem with its own constructor can be transformed.

    Every benchmark problem subclasses `OptimizationProblem`
    with a signature of its own,
    so the working problem is copied rather than constructed.
    """
    from gemseo.problem.optimization.rosenbrock import Rosenbrock

    problem = Rosenbrock()
    composition = create_working_transformation(problem.design_space, normalize=True)
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)

    assert type(working_problem) is Rosenbrock
    assert working_problem is not problem
    assert isinstance(working_problem.objective, TransformedInputFunction)
    # The original keeps its own functions and space.
    assert not isinstance(problem.objective, TransformedInputFunction)
    assert problem.design_space is not working_problem.input_space


def test_transform_does_not_share_the_collections(problem, transformation) -> None:
    """Check that the working problem owns its function collections.

    A collection is reachable from an attribute as well as from the sequence,
    so sharing one would let the working problem mutate the original.
    """
    problem.bind_functions()
    working_problem = problem.create_working_problem(transformation)

    assert working_problem.observables is not problem.observables
    assert working_problem._sequence_of_functions is not problem._sequence_of_functions
    for working, original in zip(
        working_problem._sequence_of_functions,
        problem._sequence_of_functions,
        strict=True,
    ):
        assert working is not original


def test_transform_keeps_the_declared_value_of_a_partially_valued_space() -> None:
    """Check the current value carried over when one variable has none.

    Gating the current value on the all-or-nothing `has_current_value`
    let one variable without a value discard the declared value
    of every normalized variable,
    so the run started from the center of the space instead.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=2.0)
    design_space.add_variable("y", lower_bound=0.0, upper_bound=10.0)

    problem = OptimizationProblem(design_space)
    problem.objective = ArrayFunction(
        lambda x: array([x @ x]), name="f", jac=lambda x: array([2.0 * x])
    )
    execute_algo(problem, settings_model=SLSQP_Settings(max_iter=1))

    # The variable without a value starts from the center of its range.
    assert_allclose(problem.database.get_x_vect(1), array([2.0, 5.0]))


def test_transform_passes_the_stop_if_nan_to_both_halves(
    design_space, transformation
) -> None:
    """Check that turning the stop off reaches the half that records.

    Once the functions are adapted,
    the collections of the problem hold the adapting half alone.
    Were the setting to stop there,
    the evaluation half would keep raising and end the run.
    """
    problem = EvaluationProblem(design_space)
    problem.add_observable(ArrayFunction(lambda x: array([nan]), name="f"))
    problem.bind_functions()
    working_problem = problem.create_working_problem(transformation)

    working_problem.stop_if_nan = False

    assert isnan(working_problem.observables[0].func(array([0.1, 0.2]))).all()


def test_doe_evaluates_every_sample_when_a_function_returns_nan(design_space) -> None:
    """Check that a NaN does not truncate the history of a DOE.

    A DOE turns the stop off before it runs,
    so a sample whose evaluation returns a NaN must neither stop the run
    nor be the last one recorded.
    """
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        ArrayFunction(
            lambda x: array([nan]) if x[0] > 5.0 else array([x @ x]), name="f"
        )
    )

    execute_algo(
        problem,
        algo_name="CustomDOE",
        samples=array([[1.0, 1.0], [8.0, 1.0], [2.0, 1.0]]),
        algo_type="doe",
    )

    assert len(problem.database) == 3


def test_doe_gives_the_stop_if_nan_back_to_the_problem_it_is_built_from(
    design_space, snapshot
) -> None:
    """Check that a DOE leaves the stop as it found it.

    A DOE turns the stop off on the problem it is handed,
    whose functions are the recording halves of the problem the user keeps:
    the two share them.
    Were the stop left off,
    that problem would report a stop that its own functions no longer honour,
    and evaluating one of them by hand would return a NaN in silence.
    """
    problem = EvaluationProblem(design_space)
    problem.add_observable(
        ArrayFunction(
            lambda x: array([nan]) if x[0] > 5.0 else array([x @ x]), name="f"
        )
    )

    execute_algo(
        problem,
        algo_name="CustomDOE",
        samples=array([[1.0, 1.0], [8.0, 1.0], [2.0, 1.0]]),
        algo_type="doe",
    )

    observable = problem.observables[0]
    assert problem.stop_if_nan
    assert observable.stop_if_nan
    with assert_exception(FunctionIsNan, snapshot):
        observable.evaluate(array([9.0, 1.0]))


def test_approximated_jacobian_perturbs_a_relaxed_component() -> None:
    """Check that a relaxed integer variable is perturbed as any float one.

    SLSQP does not handle integer variables,
    so the relaxation, asked for through `relax_integer_variables`,
    lets it explore one continuously,
    so the functions receive relaxed values
    and the evaluation half rounds none of its perturbations:
    rounding them would zero the derivative with respect to that variable.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)
    input_values = []

    def compute_output(input_value):
        """Compute an output value, recording the input value it is computed at.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        input_values.append(input_value.copy())
        return array([input_value @ input_value])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(compute_output, name="f")
    problem.differentiation_method = "finite_differences"
    problem.differentiation_step = 1e-7

    execute_algo(
        problem,
        settings_model=SLSQP_Settings(max_iter=3, relax_integer_variables=True),
    )

    assert input_values
    assert not all(float(input_value[1]).is_integer() for input_value in input_values)
    # The derivative of x @ x with respect to the relaxed component i is 2 i,
    # checked at every point the Jacobian was recorded at,
    # since not every recorded point has one.
    jacobians = [
        (x_vect.unwrap(), data["@f"])
        for x_vect, data in problem.database.items()
        if "@f" in data
    ]
    assert jacobians
    for x_vect, jacobian in jacobians:
        assert_allclose(jacobian.reshape(-1)[1], 2 * x_vect[1], atol=1e-6)


@pytest.mark.parametrize(
    ("relaxed_d", "projected_d"),
    [
        # The nearest choice is a real one, which rounding would not give.
        (3.2, 2.75),
        # The nearest choice is an integer one.
        (5.6, 5.0),
    ],
)
@pytest.mark.parametrize("normalize_design_space", [False, True])
def test_driver_projects_a_relaxed_optimum_onto_the_declared_domain(
    relaxed_d, projected_d, normalize_design_space
) -> None:
    """Check that the optimum written back into the design space is projected.

    The relaxation,
    asked for through `relax_integer_variables` and `relax_discrete_variables`,
    lets the algorithm explore the integer and discrete variables continuously,
    so the optimum it finds is relaxed.
    The driver projects it onto the domain the user declared once,
    when writing it back into the design space:
    it rounds an integer component,
    snaps a discrete one to its nearest choice
    and leaves a float one alone,
    while `x_opt` and `f_opt` themselves keep the relaxed optimum.
    The projected counterparts, `x_opt_projected` and `f_opt_projected`,
    hold the value the design space receives and the objective evaluated
    there, and `is_feasible_projected` its feasibility, `True` here since
    the problem has no constraint.
    With normalization on,
    the projection goes through the normalization of the composition too.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)
    space.add_discrete_variable("d", [1, 2.75, 5, 8.5], value=1)
    relaxed_optimum = array([2.5, 3.4, relaxed_d])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(
        lambda x: array([(x - relaxed_optimum) @ (x - relaxed_optimum)]),
        name="f",
        jac=lambda x: array([2.0 * (x - relaxed_optimum)]),
    )

    execute_algo(
        problem,
        algo_name="L_BFGS_B",
        max_iter=100,
        relax_integer_variables=True,
        relax_discrete_variables=True,
        normalize_design_space=normalize_design_space,
    )

    assert_allclose(problem.solution.x_opt, relaxed_optimum, atol=1e-4)
    current_value = space.get_current_value()
    assert_allclose(current_value[0], 2.5, atol=1e-4)
    assert_equal(current_value[1:], array([3.0, projected_d]))

    assert_allclose(problem.solution.x_opt_projected, current_value, atol=1e-4)
    difference = current_value - relaxed_optimum
    assert_allclose(
        problem.solution.f_opt_projected, difference @ difference, atol=1e-3
    )
    assert problem.solution.is_feasible_projected is True


def test_driver_result_without_relaxation_copies_the_optimum(monkeypatch) -> None:
    """Check that a run relaxing nothing copies the optimum instead of projecting.

    The composition built for a run that relaxes nothing is empty,
    so its `project()` returns `x_opt` itself, unchanged,
    and `_post_run` recognizes that nothing changed:
    it copies the projected fields from the ones of the optimum
    instead of evaluating the problem a second time,
    so `evaluate_functions` is never called on the user's own problem,
    only, through the algorithm itself, on the working one.
    """
    original_evaluate_functions = OptimizationProblem.evaluate_functions
    calls_on_the_original_problem = []

    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(
        lambda x: array([(x - 3.0) @ (x - 3.0)]),
        name="f",
        jac=lambda x: array([2.0 * (x - 3.0)]),
    )

    def spy(self, *args, **kwargs):
        """Record a call made on the original problem, then run it for real."""
        if self is problem:
            calls_on_the_original_problem.append(1)
        return original_evaluate_functions(self, *args, **kwargs)

    monkeypatch.setattr(OptimizationProblem, "evaluate_functions", spy)

    execute_algo(problem, algo_name="L_BFGS_B", max_iter=100)

    assert not calls_on_the_original_problem
    solution = problem.solution
    assert solution.x_opt_projected is solution.x_opt
    assert solution.x_opt_projected_as_dict is solution.x_opt_as_dict
    assert solution.f_opt_projected is solution.f_opt
    assert solution.is_feasible_projected is solution.is_feasible


def test_driver_result_projection_evaluation_failure_leaves_fields_none(
    caplog,
) -> None:
    """Check that a NaN at the projected optimum leaves the projected fields `None`.

    `evaluate_functions` raises a termination criterion when the objective it
    evaluates is `NaN`;
    `_post_run` catches it, logs a warning and leaves `f_opt_projected` and
    `is_feasible_projected` at `None`,
    while `x_opt_projected` and `x_opt_projected_as_dict` stay set,
    since the projection itself never fails.
    """
    space = DesignSpace()
    space.add_variable("x", type_="integer", lower_bound=0, upper_bound=10, value=5)

    def objective(x):
        """Return NaN at the point the relaxed optimum rounds to.

        Args:
            x: The input value.

        Returns:
            The output value.
        """
        if x[0] == 3.0:
            return array([nan])
        return array([(x[0] - 3.4) ** 2])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(objective, name="f")
    problem.differentiation_method = "finite_differences"

    execute_algo(
        problem,
        settings_model=SLSQP_Settings(max_iter=100, relax_integer_variables=True),
    )

    solution = problem.solution
    assert_allclose(solution.x_opt_projected, array([3.0]))
    assert solution.f_opt_projected is None
    assert solution.is_feasible_projected is None
    assert "Could not evaluate the functions" in caplog.text


def test_driver_result_projection_arbitrary_exception_leaves_fields_none(
    caplog,
) -> None:
    """Check that any exception at the projected optimum leaves the fields `None`.

    `evaluate_functions` may raise something other than a `TerminationCriterion`
    at the projected point, e.g. a discipline failure or a plain `ValueError`
    the objective itself raises; `_post_run` must catch it too, log a warning
    naming the error and leave `f_opt_projected` and `is_feasible_projected` at
    `None`, instead of letting it escape `execute()` after a run that otherwise
    converged.
    """
    space = DesignSpace()
    space.add_variable("x", type_="integer", lower_bound=0, upper_bound=10, value=5)

    def objective(x):
        """Raise at the point the relaxed optimum rounds to.

        Args:
            x: The input value.

        Returns:
            The output value.
        """
        if x[0] == 3.0:
            msg = "Boom"
            raise ValueError(msg)
        return array([(x[0] - 3.4) ** 2])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(objective, name="f")
    problem.differentiation_method = "finite_differences"

    execute_algo(
        problem,
        settings_model=SLSQP_Settings(max_iter=100, relax_integer_variables=True),
    )

    solution = problem.solution
    assert_allclose(solution.x_opt_projected, array([3.0]))
    assert solution.f_opt_projected is None
    assert solution.is_feasible_projected is None
    assert "Could not evaluate the functions" in caplog.text
    assert "Boom" in caplog.text


def test_driver_releases_the_perturbation_transformation_after_a_run() -> None:
    """Check that a run releases the map it told the user's own functions.

    `create_working_problem` tells the evaluation half of the objective the
    transformation for the duration of the run,
    so an approximated Jacobian of the working problem perturbs a working
    point that denormalizes to a step of `1e5` on the huge range of the
    variable, `1e12`.
    `execute()` releases that map once the run ends,
    so asking the user's own objective for a Jacobian afterwards,
    at a point the run never visited,
    so the database holds no cached value for it,
    perturbs the step of `1e-7` this problem was built with directly,
    in the user's coordinates,
    which falls below the float resolution of a value as large as `2.5e11`
    and returns a Jacobian of zeros instead of the one a working,
    normalized step would give.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1e12, value=5e11)
    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    problem.differentiation_method = "finite_differences"

    execute_algo(
        problem,
        algo_name="L_BFGS_B",
        max_iter=2,
        normalize_design_space=True,
    )

    assert problem.objective._EvaluationFunction__transformation is None
    unvisited_point = array([2.5e11])
    assert unvisited_point not in problem.database
    assert_allclose(problem.objective.jac(unvisited_point), array([0.0]))


def test_driver_releases_the_perturbation_transformation_when_the_algorithm_raises(
    monkeypatch,
) -> None:
    """Check that a raising algorithm still releases the map.

    The `finally` block of `execute()` releases the map whatever the run does,
    an algorithm raising included,
    so a caller catching the exception still finds the user's own objective
    perturbing in its own coordinates,
    checked the same way as
    `test_driver_releases_the_perturbation_transformation_after_a_run`,
    at a point the run never visited.
    """

    def _raise(self, problem) -> None:
        msg = "The algorithm failed."
        raise RuntimeError(msg)

    monkeypatch.setattr(ScipyOpt, "_run", _raise)

    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1e12, value=5e11)
    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(lambda x: x**2, name="f")
    problem.differentiation_method = "finite_differences"

    with pytest.raises(RuntimeError, match="The algorithm failed"):
        execute_algo(problem, algo_name="L_BFGS_B", normalize_design_space=True)

    assert problem.objective._EvaluationFunction__transformation is None
    unvisited_point = array([2.5e11])
    assert unvisited_point not in problem.database
    assert_allclose(problem.objective.jac(unvisited_point), array([0.0]))


def test_approximated_jacobian_of_a_new_iter_observable_rounds() -> None:
    """Check that an observable of each new iteration rounds its perturbations too.

    Such an observable carries no adaptation half,
    since it is evaluated at a point that has just been recorded,
    so its evaluation half was told the map by `bind_functions`
    like every other one.
    The map rounding here is the normalization,
    built on a space with an integer variable since nothing relaxes it.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)
    input_values = []

    def compute_observable(input_value):
        """Compute an observable, recording the input value it is computed at.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        input_values.append(input_value.copy())
        return array([input_value @ input_value])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f")
    problem.add_observable(ArrayFunction(compute_observable, name="o"))
    problem.differentiation_method = "finite_differences"
    problem.differentiation_step = 1e-7

    execute_algo(
        problem,
        algo_name="PYDOE_FULLFACT",
        algo_type="doe",
        n_samples=4,
        evaluate_observable_jacobian=True,
        relax_integer_variables=False,
        normalize_design_space=True,
    )

    assert input_values
    assert all(float(input_value[1]).is_integer() for input_value in input_values)


def test_approximated_jacobian_rounds_when_normalization_does() -> None:
    """Check that the perturbations follow the value path when it rounds.

    Denormalizing a vector rounds its integer components
    when the space it is built on has some,
    so an explicit `relax_integer_variables=False`, which relaxes nothing,
    does not stop the value path from rounding.
    The perturbations must round too,
    otherwise a function would receive a non-integral value for an integer variable
    as soon as normalization is on.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=100.0, value=10.0)
    space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)
    input_values = []

    def compute_output(input_value):
        """Compute an output value, recording the input value it is computed at.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        input_values.append(input_value.copy())
        return array([input_value @ input_value])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(compute_output, name="f")
    problem.differentiation_method = "finite_differences"
    problem.differentiation_step = 1e-7

    execute_algo(
        problem,
        algo_name="CustomDOE",
        algo_type="doe",
        samples=array([[0.1, 0.5], [0.2, 0.6]]),
        eval_jac=True,
        relax_integer_variables=False,
        normalize_design_space=True,
    )

    assert input_values
    assert all(float(input_value[1]).is_integer() for input_value in input_values)


def test_transform_refuses_a_function_that_has_not_been_bound(
    problem, transformation, snapshot
) -> None:
    """Check that the adaptation half is refused without the evaluation one.

    It is built on the evaluation half,
    so adapting a function the problem has not bound
    would build a working problem whose run leaves no history behind.
    """
    with assert_exception(ValueError, snapshot):
        problem.create_working_problem(transformation)


def test_transform_refuses_a_transformation_built_for_another_space(
    problem, snapshot
) -> None:
    """Check that a map built for another space than the problem's is refused.

    The working problem is built over the working space of the map,
    so a map built for another space would map the values of this problem
    with the bounds of that other one.
    """
    other_space = DesignSpace()
    other_space.add_variable(
        "x", size=2, lower_bound=0.0, upper_bound=10.0, value=array([1.0, 2.0])
    )
    composition = SpaceComposition(other_space)
    problem.bind_functions()

    with assert_exception(ValueError, snapshot):
        problem.create_working_problem(composition)


@pytest.mark.parametrize("differentiation_method", ["user", "finite_differences"])
def test_transform_accepts_a_transformation_changing_the_dimension(
    problem, monkeypatch, differentiation_method
) -> None:
    """Check that the working space is free to have a dimension of its own.

    The adaptation half maps a working point back to the original coordinates
    and hands it to the evaluation half,
    which records it there,
    so the working problem takes the dimension of the working space
    while the shared database keeps the one the user declared.
    An approximated Jacobian goes through the very same map on the way back:
    the frozen component it drops on the way in
    comes back as the zero column `inverse_transform_jacobian` appends.
    """
    monkeypatch.setattr(EvaluationProblem, "enable_working_database", True)
    problem.differentiation_method = differentiation_method
    composition = SpaceComposition(problem.input_space, FreezingTransformation)
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)

    assert working_problem.input_space.dimension == 1
    observable = working_problem.observables[0]
    assert_allclose(observable.func(array([1.0])), array([5.0]))
    assert_allclose(observable.jac(array([1.0])), array([2.0]), rtol=1e-5)
    assert_allclose(problem.database.get_x_vect_history(), [array([1.0, 2.0])])
    assert_allclose(problem.working_database.get_x_vect_history(), [array([1.0])])
    if differentiation_method == "finite_differences":
        # The Jacobian recorded in the coordinates the user declared
        # keeps the dimension of the original space:
        # the frozen component comes back as the zero column that
        # `inverse_transform_jacobian` appends.
        gradient_name = Database.get_gradient_name("f")
        assert_allclose(
            problem.database.get_function_value(gradient_name, array([1.0, 2.0])),
            array([[2.0, 0.0]]),
            atol=1e-5,
        )


def test_approximated_jacobian_perturbs_in_the_working_coordinates() -> None:
    """Check that the step is taken at the working point.

    `create_working_problem` tells the evaluation half the map,
    and the approximator perturbs the working point it is handed,
    so a step of `1e-7` on a working range of `[0, 1]` still moves
    a variable whose original range is `1e12`;
    taken in the original coordinates,
    it would fall below the float resolution of the value it perturbs
    and return a Jacobian of zeros.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1e12, value=5e11)
    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(lambda x: x**2, name="f"))
    problem.differentiation_method = "finite_differences"
    composition = create_working_transformation(space, normalize=True)
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)

    assert_allclose(
        working_problem.observables[0].jac(array([0.5])), array([[1e24]]), rtol=1e-6
    )


class _ScalingTransformation(BaseSpaceTransformation[DesignSpace]):
    """A user map dividing the values of its original space by `1e12`.

    Defined at module level, rather than inside a test,
    so that it stands next to the transformations gemseo provides,
    with maps of its own rather than one of gemseo's flags:
    `set_perturbation_transformation` reads no flag,
    so a user map settles the perturbation the same way a gemseo one does.
    """

    def _create_working_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space.filter(list(self._original_space), copy=True)
        for name in space:
            space.set_lower_bound(name, 0.0)
            space.set_upper_bound(name, 1.0)

        return space

    def transform_value(self, value: NumberArray) -> NumberArray:  # noqa: D102
        return value / 1e12

    def inverse_transform_value(  # noqa: D102
        self, value: NumberArray, no_check: bool = False
    ) -> NumberArray:
        return value * 1e12

    def transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian * 1e12

    def inverse_transform_jacobian(  # noqa: D102
        self, jacobian: NumberArray, value: NumberArray | None = None
    ) -> NumberArray:
        return jacobian / 1e12


def test_approximated_jacobian_follows_a_user_transformation() -> None:
    """Check that a user map settles the perturbation just as gemseo's own do.

    `set_perturbation_transformation` is handed
    whatever `create_working_problem` was given,
    so a map of the user's own,
    with no flag to declare and maps of its own,
    moves the working point the same way one of gemseo's does.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1e12, value=5e11)
    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(lambda x: x**2, name="f"))
    problem.differentiation_method = "finite_differences"
    transformation = _ScalingTransformation(space)
    problem.bind_functions()

    working_problem = problem.create_working_problem(transformation)

    assert_allclose(
        working_problem.observables[0].jac(array([0.5])), array([[1e24]]), rtol=1e-6
    )


@pytest.mark.parametrize("with_integer", [False, True])
def test_complex_step_perturbation_keeps_the_imaginary_part_through_the_backward_map(
    with_integer,
) -> None:
    """Check that a complex-step perturbation survives the round trip through a map.

    A perturbed working point is complex,
    while the current value the normalization reads its common dtype from is real,
    so denormalizing the point must keep its dtype rather than cast it,
    whether the normalization is built on the space the user declared
    or on the relaxed one an integer variable brings in.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=1e12, value=5e11)
    if with_integer:
        space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)

    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(lambda x: array([x[0] ** 2]), name="f"))
    problem.differentiation_method = "complex_step"
    composition = create_working_transformation(
        space, normalize=True, relax_integer=True
    )
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)
    working_point = composition.transform_value(space.get_current_value())

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jacobian = working_problem.observables[0].jac(working_point)

    # The derivative with respect to "x", whether the Jacobian comes back
    # flat or as a single row.
    assert_allclose(jacobian.reshape(-1)[0], 1e24, rtol=1e-6)


def _compute_squared_norm(input_value):
    """Compute the squared norm of an input value.

    Defined at the module level so that a function built on it can be
    pickled, which a process pool requires.

    Args:
        input_value: The input value.

    Returns:
        The squared norm.
    """
    return array([input_value @ input_value])


def test_approximated_jacobian_differentiates_in_parallel(
    design_space, transformation
) -> None:
    """Check that a parallel approximation perturbs in the working coordinates too.

    The approximator pickles a bound method of the evaluation half to hand
    it to a worker process, exactly the one the sequential path calls,
    so a worker must perturb the working point and map it back the same way.
    """
    problem = EvaluationProblem(design_space)
    problem.add_observable(ArrayFunction(_compute_squared_norm, name="f"))
    problem.differentiation_method = "finite_differences"
    problem.parallel_differentiation = True
    problem.parallel_differentiation_options = {"n_processes": 2}
    problem.bind_functions()

    working_problem = problem.create_working_problem(transformation)

    jacobian = working_problem.observables[0].jac(array([0.1, 0.2]))

    assert_allclose(jacobian.reshape(-1), array([20.0, 40.0]), rtol=1e-4)


def test_forward_differences_flip_at_the_working_upper_bound() -> None:
    """Check that the flip at a bound reads the working space, not the original one.

    The approximator is built on the working space once `create_working_problem` tells
    the evaluation half the map, so the bound it flips a forward step
    against is the exact one of that space, `1.0`,
    rather than the image of the original bound under the map,
    which may fall just short of it: every perturbed point still reaches
    the wrapped function within the original bound, `10.0`.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=10.0)
    input_values = []

    def compute_output(input_value):
        """Compute an output value, recording the input value it is computed at.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        input_values.append(input_value.copy())
        return array([input_value @ input_value])

    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(compute_output, name="f"))
    problem.differentiation_method = "finite_differences"
    composition = create_working_transformation(space, normalize=True)
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)
    jacobian = working_problem.observables[0].jac(array([1.0]))

    assert input_values
    assert all(input_value[0] <= 10.0 for input_value in input_values)
    assert_allclose(jacobian, array([[200.0]]), rtol=1e-4)


def test_transform_stops_before_an_integer_normalization_hides_a_nan() -> None:
    """Check that a NaN working point is caught before it is mapped back.

    Denormalizing a vector casts it to the common dtype of the space,
    `int64` for an integer-only one,
    which turns a NaN into a huge negative integer instead of raising,
    so the check of the evaluation half,
    which runs after that map,
    would see an ordinary value and pass it through to the user's function.
    The adaptation half must check the working point itself,
    before the map that can hide the NaN this way.
    """
    space = DesignSpace()
    space.add_variable("x", type_="integer", lower_bound=0, upper_bound=10, value=3)
    calls = []
    jac_calls = []

    def objective(x):
        """Compute an output value, recording the input value it is called at.

        Args:
            x: The input value.

        Returns:
            The output value.
        """
        calls.append(x)
        return array([0.0])

    def jac(x):
        """Compute a Jacobian, recording the input value it is called at.

        Args:
            x: The input value.

        Returns:
            The Jacobian.
        """
        jac_calls.append(x)
        return array([1.0])

    problem = OptimizationProblem(space)
    problem.objective = ArrayFunction(objective, name="f", jac=jac, dim=1)
    problem.bind_functions()
    composition = create_working_transformation(space, normalize=True)
    working_problem = problem.create_working_problem(composition)

    with pytest.raises(DesvarIsNan):
        working_problem.objective.evaluate(array([nan]))

    assert not calls

    with pytest.raises(DesvarIsNan):
        working_problem.objective.jac(array([nan]))

    assert not jac_calls


def test_relaxed_integer_perturbed_like_a_float() -> None:
    """Check that a relaxed integer component is perturbed like a float one.

    Relaxing then normalizing turns the integer component into a
    continuous one over the same range as the float one,
    so a working step of `1e-7` moves both original components by the
    same amount, `1e-6`.
    """
    space = DesignSpace()
    space.add_variable("x", lower_bound=0.0, upper_bound=10.0, value=5.0)
    space.add_variable("i", type_="integer", lower_bound=0, upper_bound=10, value=5)
    input_values = []

    def compute_output(input_value):
        """Compute an output value, recording the input value it is computed at.

        Args:
            input_value: The input value.

        Returns:
            The output value.
        """
        input_values.append(input_value.copy())
        return array([input_value @ input_value])

    problem = EvaluationProblem(space)
    problem.add_observable(ArrayFunction(compute_output, name="f"))
    problem.differentiation_method = "finite_differences"
    composition = create_working_transformation(
        space, normalize=True, relax_integer=True
    )
    problem.bind_functions()

    working_problem = problem.create_working_problem(composition)
    base_point = space.get_current_value()
    working_problem.observables[0].jac(composition.transform_value(base_point))

    perturbations = [abs(input_value - base_point) for input_value in input_values[1:]]
    assert perturbations
    for perturbation in perturbations:
        moved = perturbation[perturbation.nonzero()]
        assert_allclose(moved, 1e-6, atol=1e-9)
