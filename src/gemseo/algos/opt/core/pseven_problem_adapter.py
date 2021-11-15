# -*- coding: utf-8 -*-
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

from enum import Enum
from typing import Dict, List, Mapping, Optional, Tuple, Union

from da import p7core
from numpy import array, atleast_1d, concatenate, full, full_like, ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.core.mdofunctions.mdo_function import (
    MDOFunction,
    MDOLinearFunction,
    MDOQuadraticFunction,
)


class CostType(Enum):
    """The evaluation cost type of pSeven functions."""

    CHEAP = "Cheap"
    EXPENSIVE = "Expensive"


class PSevenProblem(p7core.gtopt.ProblemGeneric):
    """Adapter of OptimizationProblem to da.p7core.gtopt.ProblemGeneric.

    The methods prepare_problem() and evaluate() are defined according to pSeven's
    requirements. Refer to the API documentation of pSeven Core for more information.
    """

    def __init__(
        self,
        problem,  # type: OptimizationProblem
        evaluation_cost_type=None,  # type: Optional[Mapping[str, CostType]]
        expensive_evaluations=None,  # type: Optional[Mapping[str, int]]
        lower_bounds=None,  # type: Optional[ndarray]
        upper_bounds=None,  # type: Optional[ndarray]
        initial_point=None,  # type: Optional[ndarray]
    ):  # type: (...) -> None
        # noqa:D205,D212,D415
        """
        Args:
            problem: The optimization problem to be adapted to pSeven.
            evaluation_cost_type: The evaluation cost type of each function of the
                problem.
                If None, the evaluation cost types default to "Cheap".
            expensive_evaluations: The maximal number of expensive evaluations for
                each function of the problem.
                If None, this number is set automatically by pSeven.
            lower_bounds: The lower bounds on the design variables.
                If None, the lower bounds are read from the design space.
            upper_bounds: The upper bounds on the design variables.
                If None, the upper bounds are read from the design space.
            initial_point: The initial values of the design variables.
                If None, the initial values are read from the design space.
        """
        self.__problem = problem

        if evaluation_cost_type is None:
            self.__evaluation_cost_type = dict()
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
        elif design_space.has_current_x():
            self.__initial_point = design_space.get_current_x()
        else:
            self.__initial_point = full(design_space.dimension, None)

    def prepare_problem(self):  # type: (...) -> None
        """Initialize the problem for pSeven."""
        self.__add_variables()
        self.__add_objectives()
        self.__add_constraints()

    def __add_variables(self):  # type: (...) -> None
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

    def __add_objectives(self):  # type: (...) -> None
        """Add the objectives to the pSeven problem."""
        objective = self.__problem.objective
        hints = self.__get_p7_function_hints(objective)
        dimension = self.__get_function_dimension(objective)

        if dimension > 1:
            for index in range(dimension):
                name = self.__get_component_name(objective, index)
                self.add_objective(name, hints)
        else:
            self.add_objective(objective.name, hints)

        # Add the objectives gradients
        if objective.has_jac():
            self.enable_objectives_gradient()

    def __add_constraints(self):  # type: (...) -> None
        """Add the constraints to the pSeven problem."""
        problem = self.__problem

        for constraint in problem.constraints:
            bounds = self.__get_p7_constraint_bounds(constraint)
            hints = self.__get_p7_function_hints(constraint)
            dimension = self.__get_function_dimension(constraint)
            if dimension > 1:
                for index in range(dimension):
                    name = self.__get_component_name(constraint, index)
                    self.add_constraint(bounds, name, hints)
            else:
                self.add_constraint(bounds, constraint.name, hints)

        # Add the constraints gradients
        differentiable = all(constraint.has_jac() for constraint in problem.constraints)
        if problem.has_constraints() and differentiable:
            self.enable_constraints_gradient()

    def __get_p7_variable_type(
        self,
        variable_name,  # type: str
        index,  # type: int
    ):  # type: (...) -> str
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
        raise TypeError("Unsupported design variable type: {}".format(var_type))
        # TODO: For future reference, pSeven also supports discrete and categorical
        #  variables.

    def __get_p7_function_hints(
        self,
        function,  # type: MDOFunction
    ):  # type: (...) -> Dict[str, Union[str, int]]
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
            hints["@GTOpt/EvaluationCostType"] = self.__evaluation_cost_type[name].value
        if name in self.__expensive_evaluations:
            hints["@GTOpt/ExpensiveEvaluations"] = self.__expensive_evaluations[name]
        return hints

    @staticmethod
    def __get_p7_linearity_type(
        function,  # type: MDOFunction
    ):  # type: (...) -> str
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

    def __get_function_dimension(
        self,
        function,  # type: MDOFunction
    ):  # type: (...) -> int
        """Return the dimension of a function.

        Args:
            function: The function.

        Returns:
            The dimension of the function.

        Raises:
            RuntimeError: If the function dimension is unavailable.
        """
        design_space = self.__problem.design_space
        if function.has_dim():
            return function.dim
        elif design_space.has_current_x():
            current_x = design_space.get_current_x()
            return atleast_1d(function(current_x)).size
        else:
            raise RuntimeError("The function dimension is not available.")

    @staticmethod
    def __get_component_name(
        function,  # type: MDOFunction
        index,  # type: int
    ):  # type: (...) -> str
        """Return the name of function component.

        Args:
            function: The function.
            index: The index of the function component.

        Returns:
            The name of the function component.
        """
        return "{}{}{}".format(function.name, DesignSpace.SEP, index)

    @staticmethod
    def __get_p7_constraint_bounds(
        constraint,  # type: MDOFunction
    ):  # type: (...) -> Tuple[Optional[float], float]
        """Return the pSeven bounds associated with a constraint.

        Args:
            constraint: The constraint.

        Returns:
            The lower bound, the upper bound.

        Raises:
            ValueError: If the constraint type is invalid.
        """
        if constraint.f_type == MDOFunction.TYPE_EQ:
            return 0.0, 0.0
        if constraint.f_type == MDOFunction.TYPE_INEQ:
            return None, 0.0
        raise ValueError("Invalid constraint type.")
        # TODO: For future reference, pSeven support constraints bounded from both
        #  sides.

    def evaluate(
        self,
        queryx,  # type: ndarray
        querymask,  # type: ndarray
    ):  # type: (...) -> Tuple[List[List[float]], List[ndarray]]
        """Compute the values of the objectives and the constraints for pSeven.

        Args:
            queryx: The points to evaluate. (2D-array where each row is a point.)
            querymask: The evaluation request mask.

        Returns:
            The evaluation result. (2D-array-like with one row per point.),
            The evaluation masks. (Idem.)

        Raises:
            p7core.UserTerminated: If a termination criterion is reached.
        """
        functions_batch = list()
        output_masks_batch = list()
        for x, mask in zip(queryx, querymask):
            # Evaluate the functions, unless a stopping criterion is satisfied
            functions = list()
            try:
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
            except TerminationCriterion:
                # Interrupt pSeven
                raise p7core.UserTerminated("Gemseo stopping criterion satisfied")
            else:
                functions.extend(constr_grads)
                functions_batch.append(functions)
                output_mask = concatenate(
                    [obj_mask, constr_mask, obj_grads_mask, constr_grads_mask]
                )
                output_masks_batch.append(output_mask)

        return functions_batch, output_masks_batch

    def __compute_objectives(
        self,
        x_vec,  # type: ndarray
        mask,  # type: ndarray
    ):  # type: (...) -> Tuple[List[float], ndarray]
        obj_dim = self.__get_function_dimension(self.__problem.objective)

        if True in mask[:obj_dim]:
            objectives = atleast_1d(self.__problem.objective(x_vec)).tolist()
            output_mask = full_like(mask[:obj_dim], True)
        else:
            objectives = [None] * obj_dim
            output_mask = full_like(mask[:obj_dim], False)

        return objectives, output_mask

    def __compute_constraints(
        self,
        x_vec,  # type: ndarray
        mask,  # type: ndarray
    ):  # type: (...) -> Tuple[List[float], ndarray]
        constraints = list()
        output_mask = list()
        n_inds = 0

        for constraint in self.__problem.constraints:
            constr_dim = self.__get_function_dimension(constraint)
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
        x_vec,  # type: ndarray
        mask,  # type: ndarray
    ):  # type: (...) -> Tuple[List[float], ndarray]
        pb_dim = self.__problem.dimension
        obj_dim = self.__get_function_dimension(self.__problem.objective)
        n_values = obj_dim * pb_dim

        if self.__problem.objective.has_jac() and True in mask[:n_values]:
            obj_grads = self.__problem.objective.jac(x_vec).flatten().tolist()
            output_mask = full_like(mask[:n_values], True)
        elif self.__problem.objective.has_jac():
            obj_grads = [None] * n_values
            output_mask = full_like(mask[:n_values], False)
        else:
            obj_grads = []
            output_mask = array([])

        return obj_grads, output_mask

    def __compute_constraints_gradients(
        self,
        x_vec,  # type: ndarray
        mask,  # type: ndarray
    ):  # type: (...) -> Tuple[List[float], ndarray]
        constr_grads = list()
        output_mask = list()
        pb_dim = self.__problem.dimension
        n_inds = 0

        if all(constraint.has_jac() for constraint in self.__problem.constraints):
            for constraint in self.__problem.constraints:
                constr_dim = self.__get_function_dimension(constraint)
                n_values = constr_dim * pb_dim
                if True in mask[n_inds : n_inds + n_values]:
                    constr_grads.extend(constraint.jac(x_vec).flatten().tolist())
                    output_mask.extend([True] * n_values)
                else:
                    constr_grads.extend([None] * n_values)
                    output_mask.extend([False] * n_values)
                n_inds += n_values

        return constr_grads, array(output_mask)
