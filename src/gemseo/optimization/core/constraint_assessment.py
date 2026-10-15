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
"""The assessment of a constraint at a point, against its tolerance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Final

from numpy import abs as np_abs
from numpy import atleast_1d
from numpy import count_nonzero
from numpy import inf
from numpy import isnan
from numpy import maximum
from numpy.linalg import norm

from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization.core.constraint_status import ConstraintStatus

if TYPE_CHECKING:
    from gemseo.util.typing import RealOrComplexArray


@dataclass(frozen=True)
class ConstraintAssessment:
    r"""The assessment of a constraint at a point, against its tolerance.

    For a constraint value,
    the residual $r$ is `value - tolerance` for an inequality
    and `abs(value) - tolerance` for an equality,
    component-wise.
    The violation $v=\max(r,0)$ is also component-wise:
    a negative residual is not a violation.
    """

    status: ConstraintStatus
    """The status of the constraint."""

    squared_violation: float = 0.0
    r"""The squared violation $\lVert v\rVert_2^2$.

    This is the sum of the squares of the violations
    (Nocedal and Wright, *Numerical Optimization*, 2006),
    so its unit is the square of the unit of the constraint,
    whereas `max_violation` is in the unit of the constraint.
    It is `inf` when the value contains NaN or is missing.
    """

    max_violation: float = 0.0
    r"""The maximum violation $\max(v)$.

    This is the largest violation,
    in the unit of the constraint.
    It is `inf` when the value contains NaN or is missing.
    """

    n_violated_components: int = 0
    """The number of violated scalar components, NaN ones included."""

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.status.value}, "
            f"squared_violation={self.squared_violation:.6g}, "
            f"max_violation={self.max_violation:.6g}, "
            f"n_violated_components={self.n_violated_components})"
        )

    @classmethod
    def from_value(
        cls,
        value: RealOrComplexArray | float,
        constraint_type: ArrayFunction.ConstraintType,
        tolerance: float,
    ) -> ConstraintAssessment:
        """Assess a value of a constraint against a tolerance.

        A constraint is satisfied within its tolerance
        if and only if all the components of its value
        are less than or equal to the tolerance,
        in absolute value for an equality constraint;
        a NaN component is never satisfied.

        Args:
            value: The value of the constraint.
            constraint_type: The type of the constraint.
            tolerance: The tolerance of the constraint.

        Returns:
            The assessment of the constraint.

        Note:
            This uses the same comparison, `value <= tolerance`, as
            [Constraints.is_constraint_satisfied]
            [gemseo.optimization.core.constraints.Constraints.is_constraint_satisfied].
        """
        # A complex value, e.g. from the complex-step method,
        # is assessed on its real part.
        value = atleast_1d(value).real
        if constraint_type == ArrayFunction.ConstraintType.EQ:
            value = np_abs(value)

        # A NaN component is not satisfied.
        is_satisfied = value <= tolerance
        if is_satisfied.all():
            return cls(ConstraintStatus.SATISFIED)

        n_violated_components = int(value.size - count_nonzero(is_satisfied))
        if isnan(value).any():
            return cls(ConstraintStatus.VIOLATED, inf, inf, n_violated_components)

        violation = maximum(value - tolerance, 0.0)
        return cls(
            ConstraintStatus.VIOLATED,
            float(norm(violation) ** 2),
            float(violation.max()),
            n_violated_components,
        )


_missing_assessment: Final[ConstraintAssessment] = ConstraintAssessment(
    ConstraintStatus.MISSING, inf, inf
)
"""The assessment of a constraint whose value is absent from a point."""
