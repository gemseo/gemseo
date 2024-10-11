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
"""ODE problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import NamedTuple
from typing import Union

from numpy import asarray
from numpy import empty

from gemseo.algos.base_problem import BaseProblem
from gemseo.algos.ode.ode_result import ODEResult
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike


RHSFuncType = Callable[[Union[RealArray, float], RealArray], RealArray]
RHSJacType = Union[
    Callable[[Union[RealArray, float], RealArray], RealArray], RealArray, None
]


class DifferentiationFunctions(NamedTuple):
    """Functions to differentiate the right-hand side (RHS) of the ODE.

    Either a constant matrix
    or a function to compute it at a given time and state.

    If ``None``,
    it will be approximated.
    """

    desvar: RHSJacType
    """The function to differentiate the RHS with respect to the design variables."""

    state: RHSJacType
    """The function to differentiate the RHS with respect to state."""

    time_state: RHSJacType = None
    """The function to differentiate the RHS with respect to time and state."""


class TimeInterval(NamedTuple):
    """A time interval."""

    initial: float
    """The initial time."""

    final: float
    """The final time."""


class ODEProblem(BaseProblem):
    r"""First-order ordinary differential equation (ODE).

    A first-order ODE is written as

    .. math:: \frac{ds(t)}{dt} = f(t, s(t)).

    where :math:`f` is called the right-hand side (RHS) of the ODE
    and :math:`s(t)` is the state vector at time :math:`t`.
    """

    rhs_function: RHSFuncType
    """The RHS function :math:`f`."""

    jac: DifferentiationFunctions
    """The functions to compute the Jacobian of :math:`f`."""

    adjoint: DifferentiationFunctions
    """The functions to compute the adjoint of :math:`f`."""

    initial_state: RealArray
    """The state at the initial time."""

    solve_at_algorithm_times: bool
    """Whether to solve ODE only at time of interest.

    Otherwise, use times chosen by the algorithm.
    """

    result: ODEResult
    """The result of the ODE problem."""

    time_interval: TimeInterval
    """The initial and final times."""

    event_functions: Iterable[RHSFuncType]
    """The event functions, for which the integration stops when they get equal to 0."""

    __time_check: float
    """Used for fixing the time instant while checking the Jacobian with respect to
    time."""

    __times: RealArray | None
    """The times of interest where the state is computed.

    If ``None``, the ODE is integrated in the interval [0, 1] by default,
    and the state is evaluated in the instants chosen by the solving algorithm.
    """

    def __init__(
        self,
        func: RHSFuncType | RealArray,
        initial_state: RealArray,
        times: ArrayLike,
        jac_wrt_time_state: RHSJacType = None,
        jac_wrt_state: RHSJacType = None,
        jac_wrt_desvar: RHSJacType = None,
        adjoint_wrt_state: RHSJacType = None,
        adjoint_wrt_desvar: RHSJacType = None,
        solve_at_algorithm_times: bool | None = None,
        event_functions: Iterable[RHSFuncType] = (),
    ) -> None:
        """
        Args:
            func: The RHS function :math:`f`.
            initial_state: The initial state.
            times: Either the initial and final times
                or the times of interest where the state must be stored,
                including the initial and final times.
                When only initial and final times are provided,
                the times of interest are the instants chosen by the ODE solver
                to compute the state trajectories.
            jac_wrt_time_state: The Jacobian of :math:`f` for time and state.
                Either a constant matrix
                or a function to compute it
                at a given time and state.
                If ``None``,
                it will be approximated.
            jac_wrt_state: The Jacobian of :math:`f` with respect to state.
                Either a constant matrix
                or a function to compute it
                at a given time and state.
                If ``None``,
                it will be approximated.
            jac_wrt_desvar:  The Jacobian of :math:`f`
                with respect to the design variables.
                Either a constant matrix
                or a function to compute it
                at a given time and state.
                If ``None``,
                it will be approximated.
            adjoint_wrt_state: The adjoint relative to the state
                when using an adjoint-based ODE solver.
            adjoint_wrt_desvar: The adjoint relative to the design variables
                when using an adjoint-based ODE solver.
            solve_at_algorithm_times: Whether to solve the ODE chosen by the algorithm.
                Otherwise, use times defined in the vector `times`.
                If ``None``,
                it is initialized as ``False``
                if no terminal event is considered,
                and ``True`` otherwise.
            event_functions: The event functions,
                for which the integration stops when they get equal to 0.
                If empty,
                the solver will solve the ODE for the entire assigned time interval.
        """  # noqa: D205, D212, D415
        self.rhs_function = func

        # Define the functions computing the Jacobian.
        if jac_wrt_state is not None:
            jac_wrt_state = jac_wrt_state
        elif jac_wrt_time_state is None:
            jac_wrt_state = None
        else:
            jac_wrt_state = self._jac_wrt_state_from_jac_wrt_time_state

        self.jac = DifferentiationFunctions(
            desvar=jac_wrt_desvar, state=jac_wrt_state, time_state=jac_wrt_time_state
        )

        # Define the functions computing the adjoint.
        self.adjoint = DifferentiationFunctions(
            state=adjoint_wrt_state, desvar=adjoint_wrt_desvar
        )

        self.initial_state = initial_state

        # Define times and time interval
        self.__times = asarray(times)
        self.__times.sort()
        self.time_interval = TimeInterval(
            initial=float(self.__times[0]), final=float(self.__times[-1])
        )

        # Define event functions
        self.event_functions = event_functions
        for event_function in event_functions:
            # Remind: event_function is a Python function.
            event_function.terminal = True

        if solve_at_algorithm_times is None:
            self.solve_at_algorithm_times = not event_functions
        else:
            self.solve_at_algorithm_times = solve_at_algorithm_times

        self.result = ODEResult(
            times=empty(0),
            state_trajectories=empty(0),
            n_func_evaluations=0,
            n_jac_evaluations=0,
            terminal_event_time=0.0,
            terminal_event_index=None,
            terminal_event_state=empty(0),
            algorithm_termination_message="",
            algorithm_has_converged=False,
            algorithm_name="",
            algorithms_options={},
        )

        self.__time_check = self.time_interval[0]

    def _jac_wrt_state_from_jac_wrt_time_state(
        self, time: RealArray, state: RealArray
    ) -> RealArray:
        """Compute the Jacobian of the RHS function with respect to the state.

        This uses the function computing the Jacobian of the RHS function
        with respect to the time and the state.

        Args:
            time: The current time.
            state: The current state.

        Returns:
            The Jacobian of the RHS function with respect to the state.
        """
        jac = self.jac.time_state
        jacobian = jac(time, state) if callable(jac) else jac
        return jacobian[:, 1:]

    def check(self) -> None:
        """Ensure the parameters of the problem are consistent.

        Raises:
            ValueError: If the state and time shapes are inconsistent.
        """
        data = self.result.state_trajectories
        if data.size != 0 and data.shape[1] != self.result.times.size:
            msg = "Inconsistent state and time shapes."
            raise ValueError(msg)

    @property
    def times(self) -> RealArray | None:
        """Getter for the vector __times.

        Returns: times
        """
        return self.__times
