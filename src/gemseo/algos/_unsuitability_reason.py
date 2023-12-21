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
"""A Enum handling the reasons why an algorithm is unsuited for a problem."""

from __future__ import annotations

from typing import Final

from strenum import StrEnum

_LINEAR_SOLVER_TEMPLATE: Final[str] = "the left-hand side of the problem is not {}"
"""The template of the reason why an algorithm is unsuited for a problem."""

_OPTIMIZER_TEMPLATE: Final[str] = "it does not handle {}"
"""The template of the reason why an algorithm is unsuited for a problem."""


class _UnsuitabilityReason(StrEnum):
    """The reason why an algorithm is unsuited for a problem."""

    NO_REASON = ""

    # DriverLibrary
    EMPTY_DESIGN_SPACE = "the design space is empty"

    # LinearSolverLibrary
    NOT_SYMMETRIC = _LINEAR_SOLVER_TEMPLATE.format("symmetric")
    NOT_POSITIVE_DEFINITE = _LINEAR_SOLVER_TEMPLATE.format("positive definite")
    NOT_LINEAR_OPERATOR = _LINEAR_SOLVER_TEMPLATE.format("a linear operator")

    # OptimizationLibrary
    NON_LINEAR_PROBLEM = _OPTIMIZER_TEMPLATE.format("non-linear problems")
    INEQUALITY_CONSTRAINTS = _OPTIMIZER_TEMPLATE.format("inequality constraints")
    EQUALITY_CONSTRAINTS = _OPTIMIZER_TEMPLATE.format("equality constraints")

    # PyDOE
    SMALL_DIMENSION = (
        "the dimension of the problem is lower than the minimum dimension it can handle"
    )

    def __bool__(self) -> bool:
        return self != self.NO_REASON
