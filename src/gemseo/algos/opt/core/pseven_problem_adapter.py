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

from da import p7core
from numpy import array
from numpy import atleast_1d
from numpy import concatenate
from numpy import full
from numpy import full_like
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction


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
        """
        self.__problem = problem
        self.__use_gradient = use_gradient

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
        self,
        queryx: ndarray,
        querymask: ndarray,
    ) -> tuple[list[list[float]], list[ndarray]]:
        """Compute the values of the objectives and the constraints for pSeven.

        Args:
            queryx: The points to evaluate. (2D-array where each row is a point.)
            querymask: The evaluation request mask.

        Returns:
            The evaluation result. (2D-array-like with one row per point.),
            The evaluation masks. (Idem.)
        """
        functions_batch = list()
        output_masks_batch = list()
        for x, mask in zip(queryx, querymask):
            # Evaluate the functions, unless a stopping criterion is satisfied
            functions = list()
            # Compute the objectives values
            objectives, obj_mask = self.__compute_objectives(x, mask)
            functions.extend(objectives)
            # Compute the constraints values
            constraints, constr_mask = self.__compute_constraints(
                x, mask[len(functions) :]
            )
            functions.extend(constraints)
            # Compute the objectives gradients
            obj_grads, obj_grads_mask = self.__compute_objectives_gradients(
                x, mask[len(functions) :]
            )
            functions.extend(obj_grads)
            # Compute the constraints gradients
            constr_grads, constr_grads_mask = self.__compute_constraints_gradients(
                x, mask[len(functions) :]
            )
            functions.extend(constr_grads)
            functions_batch.append(functions)
            output_mask = concatenate(
                [obj_mask, constr_mask, obj_grads_mask, constr_grads_mask]
            )
            output_masks_batch.append(output_mask)

        return functions_batch, output_masks_batch

    def __compute_objectives(
        self,
        x_vec: ndarray,
        mask: ndarray,
    ) -> tuple[list[float], ndarray]:
        obj_dim = self.__problem.get_function_dimension(self.__problem.objective.name)

        if True in mask[:obj_dim]:
            objectives = atleast_1d(self.__problem.objective(x_vec)).tolist()
            output_mask = full_like(mask[:obj_dim], True)
        else:
            objectives = [None] * obj_dim
            output_mask = full_like(mask[:obj_dim], False)

        return objectives, output_mask

    def __compute_constraints(
        self,
        x_vec: ndarray,
        mask: ndarray,
    ) -> tuple[list[float], ndarray]:
        constraints = list()
        output_mask = list()
        n_inds = 0

        for constraint in self.__problem.constraints:
            constr_dim = self.__problem.get_function_dimension(constraint.name)
            if True in mask[n_inds : n_inds + constr_dim]:
                constraints.extend(atleast_1d(constraint(x_vec)).tolist())
                output_mask.extend([True] * constr_dim)
            else:
                constraints.extend([None] * constr_dim)
                output_mask.extend([False] * constr_dim)
            n_inds += constr_dim

        return constraints, array(output_mask)

    def __compute_objectives_gradients(
        self,
        x_vec: ndarray,
        mask: ndarray,
    ) -> tuple[list[float], ndarray]:
        if not self.__use_gradient or not self.__problem.objective.has_jac():
            return [], array([])

        pb_dim = self.__problem.dimension
        obj_dim = self.__problem.get_function_dimension(self.__problem.objective.name)
        n_values = obj_dim * pb_dim

        if True in mask[:n_values]:
            obj_grads = self.__problem.objective.jac(x_vec).flatten().tolist()
            output_mask = full_like(mask[:n_values], True)
        else:
            obj_grads = [None] * n_values
            output_mask = full_like(mask[:n_values], False)

        return obj_grads, output_mask

    def __compute_constraints_gradients(
        self,
        x_vec: ndarray,
        mask: ndarray,
    ) -> tuple[list[float], ndarray]:
        constr_grads = list()
        output_mask = list()

        if not self.__use_gradient or any(
            not constraint.has_jac() for constraint in self.__problem.constraints
        ):
            return constr_grads, array(output_mask)

        pb_dim = self.__problem.dimension
        n_inds = 0

        for constraint in self.__problem.constraints:
            constr_dim = self.__problem.get_function_dimension(constraint.name)
            n_values = constr_dim * pb_dim
            if True in mask[n_inds : n_inds + n_values]:
                constr_grads.extend(constraint.jac(x_vec).flatten().tolist())
                output_mask.extend([True] * n_values)
            else:
                constr_grads.extend([None] * n_values)
                output_mask.extend([False] * n_values)
            n_inds += n_values

        return constr_grads, array(output_mask)
