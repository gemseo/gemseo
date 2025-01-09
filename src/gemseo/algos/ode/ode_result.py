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
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Result of an ODE problem."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@dataclass
class ODEResult:
    r"""The result of the resolution of an ODE problem.

    In the ODE
    :math:`\frac{d\mathbf{s}(t)}{dt}=f(t,\mathbf{s}(t))`,
    the right-hand side (RHS) function is noted :math:`f`
    and the state variable at time :math:`t` is noted
    :math:`\mathbf{s}(t)\in\mathbb{R}^d`.
    """

    times: RealArray
    r"""The times_eval :math:`t_1,\ldots,t_N`."""

    state_trajectories: RealArray
    r"""The states at times_eval :math:`t_1,\ldots,t_N`.

    Shaped as ``(d,N)``
    where ``d`` is the state dimension.
    """

    n_func_evaluations: int
    """The number of evaluations of the RHS function :math:`f`."""

    n_jac_evaluations: int
    """The number of differentiations of the RHS function :math:`f`."""

    algorithm_has_converged: bool
    """Whether the algorithm has converged."""

    algorithm_name: str
    """The name of the ODE solver."""

    algorithm_settings: dict[str, Any]
    """The settings of the ODE solver."""

    algorithm_termination_message: str
    """The termination message of the ODE solver."""

    jac_wrt_desvar: RealArray
    """The Jacobian of the final state with respect to the design variables."""

    jac_wrt_initial_state: RealArray
    """The Jacobian of the final state with respect to the initial state."""

    terminal_event_index: int | None
    """The index of the event function responsible for the termination of the
    integration.

    If ``None``,
    the integration has been performed on the entire time interval without interruption.
    """

    final_state: RealArray
    """The state at the instant of interruption of the integration in time.

    If a terminal event occurs, it is the state during such occurrence. Otherwise, it is
    the state at the final time of integration.
    """

    termination_time: float
    """The time of interruption of the integration in time.

    If a terminal event occurs, it is the time of such occurrence. Otherwise, it is the
    final time of integration.
    """
