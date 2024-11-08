# Copyright 2022 Airbus SAS
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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Damien Guenot
#       :author: Francois Gallard, Charlie Vanaret, Benoit Pauwels
#       :author: Gabriel Max De Mendonça Abrantes
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Optimization history."""

from __future__ import annotations

import logging
from numbers import Real
from typing import TYPE_CHECKING
from typing import NamedTuple

from numpy import argmin
from numpy import array
from numpy import inf
from numpy import isnan
from numpy import ndarray
from numpy.linalg import norm

from gemseo.algos.database import Database
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.typing import RealArray
from gemseo.typing import RealOrComplexArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.mdo_functions.collections.constraints import Constraints


BestInfeasiblePointType = tuple[RealArray, RealArray, bool, dict[str, RealArray]]

LOGGER = logging.getLogger(__name__)


class OptimizationHistory:
    """An optimization history."""

    __constraints: Constraints
    """The constraints."""

    __database: Database
    """The database attached to the optimization problem.."""

    __design_space: DesignSpace
    """The design space."""

    objective_name: str
    """The name of the objective."""

    class Solution(NamedTuple):
        """A solution of the problem."""

        objective: float | RealArray
        """The value of the objective."""

        design: RealArray
        """The value of the design vector."""

        is_feasible: bool
        """Whether the solution is feasible."""

        constraints: dict[str, RealArray]
        """The values of the constraints."""

        constraint_jacobian: dict[str, RealArray]
        """The Jacobian matrices of the constraints."""

    def __init__(
        self, constraints: Constraints, database: Database, design_space: DesignSpace
    ) -> None:
        """
        Args:
            constraints: The constraints of the optimization problem.
            database: The database of the optimization problem.
            design_space: The design space.
        """  # noqa: D205, D212
        self.objective_name = ""
        self.__constraints = constraints
        self.__database = database
        self.__design_space = design_space

    @property
    def feasible_points(
        self,
    ) -> tuple[list[RealOrComplexArray], list[dict[str, float | list[int]]]]:
        """The feasible points within a given tolerance.

        This tolerance is defined by
        :attr:`.OptimizationProblem.tolerances.equality` for equality constraints and
        :attr:`.OptimizationProblem.tolerances.inequality` for inequality ones.

        Raises:
            ValueError: When the database is empty.
        """
        self.__raise_when_database_is_empty()
        x_history = []
        f_history = []
        for input_value, output_values in self.__database.items():
            # if all constraints are satisfied, store the vector
            if self.__constraints.is_point_feasible(output_values):
                x_history.append(input_value.unwrap())
                f_history.append(output_values)

        return x_history, f_history

    def check_design_point_is_feasible(
        self,
        x_vect: RealArray,
    ) -> tuple[bool, float]:
        r"""Check if a design point is feasible and measure its constraint violation.

        The constraint violation measure at a design point :math:`x` is

        .. math::

           \lVert\max(g(x)-\varepsilon_{\text{ineq}},0)\rVert_2^2
           +\lVert|\max(|h(x)|-\varepsilon_{\text{eq}},0)\rVert_2^2

        where :math:`\|.\|_2` is the Euclidean norm,
        :math:`g(x)` is the inequality constraint vector,
        :math:`h(x)` is the equality constraint vector,
        :math:`\varepsilon_{\text{ineq}}` is the tolerance
        for the inequality constraints
        and
        :math:`\varepsilon_{\text{eq}}` is the tolerance for the equality constraints.

        If the design point is feasible, the constraint violation measure is 0.

        Args:
            x_vect: The design point :math:`x`.

        Returns:
            Whether the design point is feasible,
            and its constraint violation measure.

        Raises:
            ValueError: When the database is empty.
        """
        self.__raise_when_database_is_empty()
        violation = 0.0
        x_vect_is_feasible = True
        output_names_to_values = self.__database.get(x_vect)
        constraints = self.__constraints
        for constraint in constraints:
            constraint_value = output_names_to_values.get(constraint.name)
            if constraint_value is None:
                break

            f_type = constraint.f_type
            if constraints.is_constraint_satisfied(f_type, constraint_value):
                continue

            x_vect_is_feasible = False
            if isnan(constraint_value).any():
                return x_vect_is_feasible, inf

            if f_type == MDOFunction.ConstraintType.INEQ:
                tolerance = constraints.tolerances.inequality
            else:
                tolerance = constraints.tolerances.equality
                constraint_value = abs(constraint_value)

            if isinstance(constraint_value, ndarray):
                constraint_value = constraint_value[constraint_value > tolerance]

            violation += norm(constraint_value - tolerance) ** 2

        return x_vect_is_feasible, violation

    def __get_best_infeasible_point(self) -> BestInfeasiblePointType:
        """Returns the best infeasible point within a given tolerance.

        Returns:
            The design variables values, the objective function value,
            the feasibility of the point and the functions values.
        """
        x_history = []
        f_history = []
        is_feasible = []
        viol_criteria = []
        for x_vect, out_val in self.__database.items():
            is_pt_feasible, f_violation = self.check_design_point_is_feasible(x_vect)
            is_feasible.append(is_pt_feasible)
            viol_criteria.append(f_violation)
            x_history.append(x_vect.unwrap())
            f_history.append(out_val)

        best_i = int(argmin(array(viol_criteria)))
        outputs_opt = f_history[best_i]
        f_opt = outputs_opt.get(self.objective_name)
        if isinstance(f_opt, ndarray) and len(f_opt) == 1:
            f_opt = f_opt[0]

        return x_history[best_i], f_opt, is_feasible[best_i], outputs_opt

    @property
    def optimum(self) -> Solution:
        """The optimum solution within a given feasibility tolerance.

        This solution is defined by:

        - the value of the objective function,
        - the value of the design variables,
        - the indicator of feasibility of the optimal solution,
        - the value of the constraints,
        - the value of the gradients of the constraints.

        Raises:
            ValueError: When the database is empty.
        """
        self.__raise_when_database_is_empty()
        feas_x, feas_f = self.feasible_points
        constraints = self.__constraints

        # Case 1: there is no feasible point; we return the least infeasible one.
        if not feas_x:
            msg = (
                "Optimization found no feasible point; "
                "the least infeasible point is selected."
            )
            LOGGER.warning(msg)
            x_opt, f_opt, _, f_history = self.__get_best_infeasible_point()
            c_opt = {c.name: f_history.get(c.name) for c in constraints}
            func = Database.get_gradient_name
            c_opt_grad = {c.name: f_history.get(func(c.name)) for c in constraints}
            return self.Solution(f_opt, x_opt, False, c_opt, c_opt_grad)

        # Case 2: the solution is feasible; we return it.
        f_opt, x_opt = inf, array([])
        c_opt = {}
        c_opt_grad = {}
        obj_name = self.objective_name
        for i, output_values in enumerate(feas_f):
            obj_value = output_values.get(obj_name)
            if obj_value is None:
                continue

            if not isinstance(obj_value, Real) and obj_value.size > 1:
                obj_value = norm(obj_value)

            if obj_value < f_opt:
                f_opt = obj_value
                x_opt = feas_x[i]
                for constraint in constraints:
                    c_name = constraint.name
                    c_opt[c_name] = output_values.get(c_name)
                    c_key = Database.get_gradient_name(c_name)
                    c_opt_grad[constraint.name] = output_values.get(c_key)

        if isinstance(f_opt, ndarray) and len(f_opt) == 1:
            f_opt = f_opt[0]

        return self.Solution(f_opt, x_opt, True, c_opt, c_opt_grad)

    def get_data_by_names(
        self,
        names: str | Iterable[str],
        as_dict: bool = True,
        filter_non_feasible: bool = False,
    ) -> RealArray | dict[str, RealArray]:
        """Return the data for specific names of variables.

        Args:
            names: The names of the variables.
            as_dict: If ``True``, return values as dictionary.
            filter_non_feasible: If ``True``, remove the non-feasible points from
                the data.

        Returns:
            The data related to the variables.

        Raises:
            ValueError: When the database is empty.
        """
        self.__raise_when_database_is_empty()
        data = self.__database.to_dataset(name="OptimizationProblem").get_view(
            variable_names=names
        )
        data = data.to_dict(orient="list") if as_dict else data.to_numpy()
        if not filter_non_feasible:
            return data

        feasible_indices = [
            self.__database.get_iteration(x) - 1 for x in self.feasible_points[0]
        ]
        if not as_dict:
            return data[feasible_indices, :]

        return {key: array(value)[feasible_indices] for key, value in data.items()}

    @property
    def last_point(self) -> Solution:
        """The last point.

        The last point is defined by:

        - the value of the objective function,
        - the value of the design variables,
        - the indicator of feasibility of the last point,
        - the value of the constraints,
        - the value of the gradients of the constraints.

        Raises:
            ValueError: When the database is empty.
        """
        database = self.__raise_when_database_is_empty()
        x_last = database.get_x_vect(-1)
        output_last = database[x_last]
        f_last = database.get_function_value(self.objective_name, -1)
        constraints = self.__constraints
        is_feas = constraints.is_point_feasible(output_last)
        c_last = {c.name: output_last.get(c.name) for c in constraints}
        func = Database.get_gradient_name
        c_last_grad = {c.name: output_last.get(func(c.name)) for c in constraints}
        return self.Solution(f_last, x_last, is_feas, c_last, c_last_grad)

    def __raise_when_database_is_empty(self) -> Database:
        """Raise an exception when the database is empty.

        Raises:
            ValueError: When the database is empty.
        """
        database = self.__database
        if not database:
            msg = "The optimization history is empty."
            raise ValueError(msg)

        return database
