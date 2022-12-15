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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                        documentation
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""An adapter from `.OptimizationProblem` to a pSeven ProblemGeneric."""
from __future__ import annotations

from typing import Mapping
from typing import NamedTuple
from typing import Optional

from da import p7core
from numpy import atleast_1d
from numpy import full
from numpy import ndarray

from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
from gemseo.core.parallel_execution import ParallelExecution

OutputValues = list[Optional[float]]
OutputMask = list[bool]


class PSevenProblem(p7core.gtopt.ProblemGeneric):
    """Adapter of OptimizationProblem to da.p7core.gtopt.ProblemGeneric.

    The methods prepare_problem() and evaluate() are defined according to pSeven's
    requirements. Refer to the API documentation of pSeven Core for more information.
    """

    def __init__(
        self,
        problem: OptimizationProblem,
        evaluation_cost_type: str | Mapping[str, str] | None = None,
        expensive_evaluations: Mapping[str, int] | None = None,
        lower_bounds: ndarray | None = None,
        upper_bounds: ndarray | None = None,
        initial_point: ndarray | None = None,
        use_gradient: bool = True,
        use_threading: bool = False,
        normalize_design_space: bool = True,
    ) -> None:
        # noqa:D205,D212,D415
        """
        Args:
            problem: The optimization problem to be adapted to pSeven.
            evaluation_cost_type: The evaluation cost type of each function of the
                problem: "Cheap" or "Expensive".
                If a string, then the same cost type is set for all the functions.
                If None, the evaluation cost types are set by pSeven.
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem.
                If None, this number is set automatically by pSeven.
            lower_bounds: The lower bounds on the design variables.
                If None, the lower bounds are read from the design space.
            upper_bounds: The upper bounds on the design variables.
                If None, the upper bounds are read from the design space.
            initial_point: The initial values of the design variables.
                If None, the initial values are read from the design space.
            use_gradient: Whether to use the derivatives of the functions.
                If the functions have no derivative then this value has no effect.
            use_threading: Whether to use threads instead of processes to parallelize
                the execution.
            normalize_design_space: Whether the design variables are normalized.
        """  # noqa: D205, D212, D415
        self.__normalize_design_space = normalize_design_space
        self.__problem = problem
        self.__use_gradient = use_gradient
        self.__use_threading = use_threading

        if evaluation_cost_type is None:
            self.__evaluation_cost_type = dict()
        elif isinstance(evaluation_cost_type, str):
            self.__evaluation_cost_type = {
                name: evaluation_cost_type for name in problem.get_all_functions_names()
            }
        else:
            self.__evaluation_cost_type = evaluation_cost_type

        if expensive_evaluations is None:
            self.__expensive_evaluations = dict()
        else:
            self.__expensive_evaluations = expensive_evaluations

        # Set the design variables bounds and initial values
        design_space = problem.design_space

        if lower_bounds is None:
            self.__lower_bounds = design_space.get_lower_bounds()
        else:
            self.__lower_bounds = lower_bounds

        if upper_bounds is None:
            self.__upper_bounds = design_space.get_upper_bounds()
        else:
            self.__upper_bounds = upper_bounds

        if initial_point is not None:
            self.__initial_point = initial_point
        elif design_space.has_current_value():
            self.__initial_point = design_space.get_current_value()
        else:
            self.__initial_point = full(design_space.dimension, None)

    def prepare_problem(self) -> None:
        """Initialize the problem for pSeven."""
        self.__add_variables()
        self.__add_objectives()
        self.__add_constraints()

    def __add_variables(self) -> None:
        """Add the design variables to the pSeven problem."""
        design_space = self.__problem.design_space
        for var_name in design_space.variables_names:
            var_indexes = design_space.get_variables_indexes([var_name])
            lower_bound = self.__lower_bounds[var_indexes]
            upper_bound = self.__upper_bounds[var_indexes]
            current_x = self.__initial_point[var_indexes]
            indexed_names = design_space.get_indexed_var_name(var_name)
            for index in range(len(indexed_names)):
                bounds = (lower_bound[index], upper_bound[index])
                initial_guess = current_x[index]
                pseven_type = self.__get_p7_variable_type(var_name, index)
                hints = {"@GT/VariableType": pseven_type}
                self.add_variable(bounds, initial_guess, indexed_names[index], hints)

    def __add_objectives(self) -> None:
        """Add the objectives to the pSeven problem."""
        objective = self.__problem.objective
        hints = self.__get_p7_function_hints(objective)
        dimension = self.__problem.get_function_dimension(objective.name)

        if dimension > 1:
            for index in range(dimension):
                name = objective.get_indexed_name(index)
                self.add_objective(name, hints)
        else:
            self.add_objective(objective.name, hints)

        # Add the objectives gradients
        if self.__use_gradient and objective.has_jac():
            self.enable_objectives_gradient()

    def __add_constraints(self) -> None:
        """Add the constraints to the pSeven problem."""
        problem = self.__problem

        for constraint in problem.constraints:
            bounds = self.__get_p7_constraint_bounds(constraint)
            hints = self.__get_p7_function_hints(constraint)
            dimension = problem.get_function_dimension(constraint.name)
            if dimension > 1:
                for index in range(dimension):
                    name = constraint.get_indexed_name(index)
                    self.add_constraint(bounds, name, hints)
            else:
                self.add_constraint(bounds, constraint.name, hints)

        # Add the constraints gradients
        if self.__use_gradient:
            differentiable = all(
                constraint.has_jac() for constraint in problem.constraints
            )
            if problem.has_constraints() and differentiable:
                self.enable_constraints_gradient()

    def __get_p7_variable_type(
        self,
        variable_name: str,
        index: int,
    ) -> str:
        """Return the pSeven variable type associated with a design variable component.

        Args:
            variable_name: The name of the design variable.
            index: The index of the variable component.

        Returns:
            The pSeven variable type.

        Raises:
            TypeError: If the type of the design variable is not supported by pSeven.
        """
        var_type = self.__problem.design_space.get_type(variable_name)[index]
        if var_type == DesignSpace.FLOAT.value:
            return "Continuous"
        if var_type == DesignSpace.INTEGER.value:
            return "Integer"
        raise TypeError(f"Unsupported design variable type: {var_type}.")
        # TODO: For future reference, pSeven also supports discrete and categorical
        #  variables.

    def __get_p7_function_hints(
        self,
        function: MDOFunction,
    ) -> dict[str, str | int]:
        """Return the pSeven hints associated with a function.

        Args:
            function: The function.

        Returns:
            The pSeven hints.
        """
        linearity_type = PSevenProblem.__get_p7_linearity_type(function)
        hints = {"@GTOpt/LinearityType": linearity_type}
        name = function.name
        if name in self.__evaluation_cost_type:
            hints["@GTOpt/EvaluationCostType"] = self.__evaluation_cost_type[name]
        if name in self.__expensive_evaluations:
            hints["@GTOpt/ExpensiveEvaluations"] = self.__expensive_evaluations[name]
        return hints

    @staticmethod
    def __get_p7_linearity_type(
        function: MDOFunction,
    ) -> str:
        """Return the pSeven linearity type of a function.

        Args:
            function: The function.

        Returns:
            The pSeven linearity type.
        """
        if isinstance(function, MDOLinearFunction):
            return "Linear"
        if isinstance(function, MDOQuadraticFunction):
            return "Quadratic"
        return "Generic"

    def __get_p7_constraint_bounds(
        self,
        constraint: MDOFunction,
    ) -> tuple[float | None, float]:
        """Return the pSeven bounds associated with a constraint.

        Args:
            constraint: The constraint.

        Returns:
            The lower bound, the upper bound.

        Raises:
            ValueError: If the constraint type is invalid.
        """
        if constraint.f_type == MDOFunction.TYPE_EQ:
            return -self.__problem.eq_tolerance, self.__problem.eq_tolerance

        if constraint.f_type == MDOFunction.TYPE_INEQ:
            return None, self.__problem.ineq_tolerance

        raise ValueError("Invalid constraint type.")
        # TODO: For future reference, pSeven support constraints bounded from both
        #  sides.

    def evaluate(
        self, queryx: ndarray, querymask: ndarray
    ) -> tuple[list[OutputValues], list[OutputMask]]:
        """Compute the values of the objectives and the constraints for pSeven.

        Args:
            queryx: The points to evaluate. (2D-array where each row is a point.)
            querymask: The evaluation request mask.

        Returns:
            The evaluation result. (2D-array-like with one row per point.),
            The evaluation masks. (Idem.)
        """
        number_of_processes = queryx.shape[0]
        if number_of_processes == 1:
            output_values, output_mask, _, _ = Worker(
                self.__problem, self.__use_gradient
            )((queryx[0], querymask[0]))
            return [output_values], [output_mask]

        # Create the workers for parallel evaluations
        if self.__use_threading:
            workers = [
                Worker(self.__problem, self.__use_gradient)
                for _ in range(number_of_processes)
            ]
        else:
            workers = Worker(self.__problem, self.__use_gradient)

        # Create a callback to fill the database on-the-fly
        def storing_callback(index: int, outputs: Outputs) -> None:
            """Store the outputs in the database.

            Args:
                index: The index of the sample.
                outputs: The outputs of the worker.
            """
            _, _, outputs, jacobians = outputs
            for name, value in jacobians.items():
                outputs[Database.get_gradient_name(name)] = value

            x = queryx[index]
            if self.__normalize_design_space:
                x = self.__problem.design_space.unnormalize_vect(x)

            self.__problem.database.store(x, outputs)

        # Evaluate the samples in parallel
        functions_batch, output_masks_batch, _, _ = zip(
            *ParallelExecution(
                workers, number_of_processes, self.__use_threading
            ).execute(list(zip(queryx, querymask)), exec_callback=storing_callback)
        )
        return list(functions_batch), list(output_masks_batch)


class Outputs(NamedTuple):
    """The outputs of a worker."""

    output_values: OutputValues
    """The output values to be passed to pSeven."""
    output_mask: OutputMask
    """The mask of the output values to be passed to pSeven."""
    values: dict[str, float | ndarray]
    """The function values computed by GEMSEO."""
    jacobians_values: dict[str, ndarray]
    """The Jacobians computed by GEMSEO."""


class Worker:
    """A worker to evaluate the functions of a problem."""

    def __init__(self, problem: OptimizationProblem, use_gradient: bool) -> None:
        """
        Args:
            problem: The optimization problem to be adapted to pSeven.
            use_gradient: Whether to use the derivatives of the functions.
                If the functions have no derivative then this value has no effect.
        """  # noqa: D205, D212, D415
        self.__problem = problem
        self.__use_gradient = use_gradient

    def __call__(self, inputs: tuple[ndarray, ndarray]) -> Outputs:
        """Execute the worker.

        Evaluate the functions of the problem.

        Args:
            inputs: The points to evaluate and their masks.

        Returns:
            The outputs and their mask.
        """
        x, mask = inputs
        dimensions = self.__problem.get_functions_dimensions()
        objective_dimension = dimensions[self.__problem.objective.name]

        # Get the names of the constraints to evaluate.
        constraints_names = list()
        mask_index = objective_dimension
        for name in self.__problem.get_constraints_names():
            dimension = dimensions[name]
            if True in mask[mask_index : mask_index + dimension]:
                constraints_names.append(name)

            mask_index += dimension

        # Get the names of the functions to differentiate
        jacobians_names = list()
        if self.__use_gradient:
            for function in [self.__problem.objective] + self.__problem.constraints:
                dimension = self.__problem.dimension * dimensions[function.name]
                if (
                    function.has_jac()
                    and True in mask[mask_index : mask_index + dimension]
                ):
                    jacobians_names.append(function.name)

                mask_index += dimension

        # Evaluate the functions and compute the Jacobian matrices
        values, jacobians = self.__problem.evaluate_functions(
            x,
            eval_obj=True in mask[:objective_dimension],
            constraints_names=constraints_names,
            jacobians_names=jacobians_names,
            normalize=self.__problem.preprocess_options.get(
                "is_function_input_normalized", False
            ),
        )

        return self.__get_output_and_mask(values, jacobians)

    def __get_output_and_mask(
        self, values: dict[str, float | ndarray], jacobians: dict[str, ndarray]
    ) -> Outputs:
        """Return the outputs and their mask.

        Args:
            values: The values of the functions.
            jacobians: The Jacobian matrices of the functions.

        Returns:
            The outputs and their mask.
        """
        output_values = list()
        output_mask = list()
        functions = [self.__problem.objective] + self.__problem.constraints
        dimensions = self.__problem.get_functions_dimensions()

        # Get the values of the functions
        for function in functions:
            dimension = dimensions[function.name]
            if function.name in values:
                output_values.extend(atleast_1d(values[function.name]).tolist())
                output_mask.extend([True] * dimension)
            else:
                output_values.extend([None] * dimension)
                output_mask.extend([False] * dimension)

        # Get the derivatives of the functions
        for function in functions:
            size = self.__problem.dimension * dimensions[function.name]
            if function.name in jacobians:
                output_values.extend(jacobians[function.name].flatten().tolist())
                output_mask.extend([True] * size)
            elif self.__use_gradient and function.has_jac():
                output_values.extend([None] * size)
                output_mask.extend([False] * size)

        return Outputs(output_values, output_mask, values, jacobians)
