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
#        :author: Isabelle Santos
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Result of an ODE problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class ODEResult:
    """The result of an ODE problem."""

    time_vector: NDArray[float]
    """The vector of times for the solution."""

    state_vector: NDArray[float]
    """The vector of states for the solution.

    This array contains one state for each time. As such, it has one line per dimension
    to the state of the problem, and one column per time contained in the time vector.
    """

    n_func_evaluations: int
    """The number of evaluations of the right-hand side."""

    n_jac_evaluations: int
    """The number of evaluations of the Jacobian of the right-hand side."""

    solver_message: str
    """The solver's termination message."""

    is_converged: bool
    """Whether the algorithm has converged."""

    solver_options: dict[str, Any]
    """The options passed to the solver."""

    solver_name: str
    """The name of the solver."""
