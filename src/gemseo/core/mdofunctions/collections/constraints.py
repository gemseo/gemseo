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
"""A mutable sequence of constraints."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import abs as np_abs
from numpy import absolute
from numpy import all as np_all
from numpy import atleast_1d

from gemseo.algos.aggregation.aggregation_func import aggregate_iks
from gemseo.algos.aggregation.aggregation_func import aggregate_lower_bound_ks
from gemseo.algos.aggregation.aggregation_func import aggregate_max
from gemseo.algos.aggregation.aggregation_func import aggregate_positive_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_sum_square
from gemseo.algos.aggregation.aggregation_func import aggregate_upper_bound_ks
from gemseo.core.mdofunctions.collections.functions import Functions
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.mdo_quadratic_function import MDOQuadraticFunction
from gemseo.disciplines.constraint_aggregation import ConstraintAggregation
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.algos.constraint_tolerances import ConstraintTolerances
    from gemseo.algos.design_space import DesignSpace
    from gemseo.typing import RealArray
    from gemseo.typing import RealOrComplexArray


class Constraints(Functions):
    """A mutable sequence of constraints."""

    AggregationFunction = ConstraintAggregation.EvaluationFunction

    _AGGREGATION_FUNCTION_MAP: Final[str] = {
        AggregationFunction.IKS: aggregate_iks,
        AggregationFunction.LOWER_BOUND_KS: aggregate_lower_bound_ks,
        AggregationFunction.UPPER_BOUND_KS: aggregate_upper_bound_ks,
        AggregationFunction.POS_SUM: aggregate_positive_sum_square,
        AggregationFunction.MAX: aggregate_max,
        AggregationFunction.SUM: aggregate_sum_square,
    }

    _F_TYPES: ClassVar[tuple[str, str]] = (
        MDOFunction.ConstraintType.EQ,
        MDOFunction.ConstraintType.INEQ,
    )

    __design_space: DesignSpace
    """The design space on which the constraints are evaluated."""

    __aggregated_constraint_indices: list[int]
    """The indices of the aggregated constraints."""

    __tolerances: ConstraintTolerances
    """The constraint tolerances."""

    def __init__(
        self, design_space: DesignSpace, tolerances: ConstraintTolerances
    ) -> None:
        """
        Args:
            design_space: The design space.
            tolerances: The constraint tolerances.
        """  # noqa: D202, D205, D212
        super().__init__()
        self.__design_space = design_space
        self.__aggregated_constraint_indices = []
        self.__tolerances = tolerances

    @property
    def tolerances(self) -> ConstraintTolerances:
        """The constraint tolerances."""
        return self.__tolerances

    @property
    def aggregated_constraint_indices(self) -> list[int]:
        """The indices of the aggregated constraints."""
        return self.__aggregated_constraint_indices

    def aggregate(
        self,
        constraint_index: int,
        method: Callable[[RealArray], float]
        | AggregationFunction = AggregationFunction.MAX,
        groups: Iterable[Sequence[int]] | None = None,
        **options: Any,
    ) -> None:
        """Aggregate a constraint to generate a reduced dimension constraint.

        Args:
            constraint_index: The index of the constraint in :attr:`.constraints`.
            method: The aggregation method, e.g. ``"max"``, ``"lower_bound_KS"``,
                ``"upper_bound_KS"``or ``"IKS"``.
            groups: The groups of components of the constraint to aggregate
                to produce one aggregation constraint per group of components;
                if ``None``, a single aggregation constraint is produced.
            **options: The options of the aggregation method.

        Raises:
            KeyError: When the given index is greater or equal
                to the number of constraints.
        """
        n_constraints = len(self)
        if constraint_index >= n_constraints:
            msg = (
                f"The index of the constraint ({constraint_index}) must be lower "
                f"than the number of constraints ({n_constraints})."
            )
            raise KeyError(msg)

        constraint = self[constraint_index]
        if callable(method):
            aggregate_constraints = method
        else:
            aggregate_constraints = self._AGGREGATION_FUNCTION_MAP[method]

        del self[constraint_index]
        if groups is None:
            self.insert(constraint_index, aggregate_constraints(constraint, **options))
            self.__aggregated_constraint_indices.append(constraint_index)
        else:
            aggregated_constraints = [
                aggregate_constraints(constraint, indices, **options)
                for indices in groups
            ]
            self[constraint_index:constraint_index] = aggregated_constraints
            self.__aggregated_constraint_indices.extend(
                list(
                    range(
                        constraint_index,
                        constraint_index + len(aggregated_constraints) + 1,
                    )
                )
            )

    def format(
        self,
        function: MDOFunction,
        value: float = 0.0,
        constraint_type: MDOFunction.ConstraintType | None = None,
        positive: bool = False,
    ) -> MDOFunction:
        r"""Format a constraint.

        An equality constraint is written as :math:`c(x)=a`,
        a positive inequality constraint is written as :math:`c(x)\geq a`
        and a negative inequality constraint is written as :math:`c(x)\leq a`.

        Args:
            function: The function :math:`c`.
            value: The value :math:`a`.
            constraint_type: The type of the constraint.
                If ``None``,
                ``function.f_type`` must be either
                ``MDOFunction.ConstraintType.INEQ``
                or ``MDOFunction.ConstraintType.EQ``.
            positive: Whether the inequality constraint is positive.

        Returns:
            A formatted constraint ready to be added to the sequence.

        Raises:
            TypeError: When the constraint of a linear optimization problem
                is not an :class:`.MDOLinearFunction`.
            ValueError: When the type of the constraint is missing.
        """
        func_name = function.name
        has_default_name = function.has_default_name
        ctype = constraint_type or function.f_type
        cstr_repr = self.__get_string_representation(function, ctype, value, positive)
        if value != 0:
            function = function.offset(-value)
        if positive:
            function = -function

        if constraint_type is not None:
            function.f_type = constraint_type
        elif not function.is_constraint():
            msg = (
                "Constraint type must be provided, "
                "either when defining the function or when adding it to the problem."
            )
            raise ValueError(msg)

        function.special_repr = cstr_repr
        if not has_default_name:
            function.name = func_name
            if function.output_names:
                output_names = "#".join(function.output_names)
                cstr_repr = cstr_repr.replace(func_name, output_names)
                function.expr = function.expr.replace(func_name, output_names)
                function.special_repr = f"{func_name}: {cstr_repr}"

        return function

    @staticmethod
    def __get_string_representation(
        function: MDOFunction,
        constraint_type: MDOFunction.ConstraintType,
        value: float | None = None,
        positive: bool = False,
    ) -> str:
        """Express a constraint as a string expression.

        Args:
            function: The constraint function.
            constraint_type: The type of the constraint.
            value: The value for which the constraint is active.
                If ``None``, this value is 0.
            positive: If ``True``, then the inequality constraint is positive.

        Returns:
            A string representation of the constraint.
        """
        if value is None:
            value = 0.0
        str_repr = function.name
        if function.input_names:
            arguments = ", ".join(function.input_names)
            str_repr += f"({arguments})"

        if constraint_type == MDOFunction.ConstraintType.EQ:
            sign = " == "
        elif positive:
            sign = " >= "
        else:
            sign = " <= "

        if function.expr:
            str_repr += ": "
            expr = function.expr
            n_char = len(str_repr)
            # Remove empty lines with filter
            expr_spl = [_f for _f in expr.split("\n") if _f]
            str_repr = str_repr + expr_spl[0] + sign + str(value)
            if isinstance(function, (MDOLinearFunction, MDOQuadraticFunction)):
                for repre in expr_spl[1:]:
                    str_repr += "\n" + " " * n_char + repre
            else:
                for repre in expr_spl[1:]:
                    str_repr += "\n" + " " * n_char + repre + sign + str(value)
        else:
            str_repr += sign + str(value)
        return str_repr

    def get_equality_constraints(self) -> Iterator[MDOFunction]:
        """Return the equality constraints.

        Yields:
            The equality constraints.
        """
        for constraint in self._functions:
            if constraint.f_type == constraint.ConstraintType.EQ:
                yield constraint

    def get_inequality_constraints(self) -> Iterator[MDOFunction]:
        """Return the inequality constraints.

        Yields:
            The inequality constraints.
        """
        for constraint in self._functions:
            if constraint.f_type == constraint.ConstraintType.INEQ:
                yield constraint

    def get_active(
        self,
        x_vect: RealArray,
        tol: float = 1e-6,
    ) -> dict[MDOFunction, RealArray]:
        """Indicate the active components of the different inequality constraints.

        Args:
            x_vect: The vector of design variables.
            tol: The tolerance for deciding whether a constraint is active.

        Returns:
            For each constraint,
            a boolean indicator of activation of its different components.
        """
        design_space = self.__design_space
        design_space.check_membership(x_vect)
        if self._functions and self._functions[0].expects_normalized_inputs:
            x_vect = design_space.normalize_vect(x_vect)

        return {
            ineq_constraint: atleast_1d((ineq_constraint.evaluate(x_vect)) >= -tol)
            for ineq_constraint in self.get_inequality_constraints()
        }

    def is_constraint_satisfied(
        self,
        constraint_type: MDOFunction.ConstraintType,
        constraint_value: RealArray,
    ) -> bool:
        """Determine if an evaluation satisfies a constraint within a given tolerance.

        Args:
            constraint_type: The type of the constraint.
            constraint_value: The value of the constraint.

        Returns:
            Whether a value satisfies a constraint.
        """
        if constraint_type == MDOFunction.ConstraintType.EQ:
            return np_all(np_abs(constraint_value) <= self.__tolerances.equality)

        return np_all(constraint_value <= self.__tolerances.inequality)

    def is_point_feasible(
        self,
        point: Mapping[str, RealOrComplexArray],
    ) -> bool:
        """Check if a point is feasible.

        Note:
            If the value of a constraint is absent from this point,
            then this constraint will be considered satisfied.

        Args:
            point: An optimization point defined by variable values.

        Returns:
            The feasibility of the point.
        """
        feasible = True
        for constraint in self._functions:
            constraint_value = point.get(constraint.name, None)
            if constraint_value is None or not self.is_constraint_satisfied(
                constraint.f_type, constraint_value
            ):
                return False

        return feasible

    def get_number_of_unsatisfied_constraints(
        self,
        values: Mapping[str, float | RealArray] = READ_ONLY_EMPTY_DICT,
    ) -> int:
        """Return the number of scalar constraints not satisfied by design variables.

        Args:
            values: The values of the constraints.

        Returns:
            The number of unsatisfied scalar constraints.
        """
        n_unsatisfied = 0
        for constraint in self._functions:
            if constraint.name not in values:
                continue

            value = atleast_1d(values[constraint.name])
            if constraint.f_type == MDOFunction.ConstraintType.EQ:
                value = absolute(value)
                tolerance = self.__tolerances.equality
            else:
                tolerance = self.__tolerances.inequality

            n_unsatisfied += sum(value > tolerance)

        return n_unsatisfied
