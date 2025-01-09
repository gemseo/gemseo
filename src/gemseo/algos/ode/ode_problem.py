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
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.typing import RealArray
from gemseo.utils.derivatives.approximation_modes import ApproximationMode

if TYPE_CHECKING:
    from collections.abc import Iterable


RHSFuncType = Callable[[Union[RealArray, float], RealArray], RealArray]
RHSJacType = Union[Callable[[Union[RealArray, float], RealArray], RealArray], RealArray]


class TimeInterval(NamedTuple):
    """A time interval."""

    initial: float
    """The initial time."""

    final: float
    """The final time."""


class ODEProblem(BaseProblem):
    r"""First-order Ordinary Differential Equation (ODE).

    A first-order ODE is written as

    .. math:: \frac{ds(t)}{dt} = f(t, s(t)).

    where :math:`f` is called the right-hand side (RHS) of the ODE
    and :math:`s(t)` is the state vector at time :math:`t`.
    """

    event_functions: Iterable[RHSFuncType]
    """The event functions, for which the integration stops when they get equal to 0."""

    initial_state: RealArray
    """The state at the initial time."""

    jac_function_wrt_state: RHSFuncType
    """The function to compute the Jacobian of :math:`f` with respect to the state."""

    jac_function_wrt_desvar: RHSFuncType
    """The function to compute the Jacobian of :math:`f` with respect to the design
    variables."""

    result: ODEResult
    """The result of the ODE problem."""

    rhs_function: RHSFuncType
    """The RHS function :math:`f`, function of the time and state."""

    solve_at_algorithm_times: bool
    """Whether to solve the ODE only at the times of interest."""

    __time_interval: TimeInterval
    """The initial and final times."""

    __jacobian_check_time: float
    """Used for fixing the time instant while checking the Jacobian."""

    __evaluation_times: RealArray | None
    """The times of interest where the state is computed.

    If ``None``, the ODE is integrated in the interval [0, 1].
    """

    def __init__(
        self,
        func: RHSFuncType | RealArray,
        initial_state: RealArray,
        times: RealArray,
        jac_function_wrt_state: RHSJacType = None,
        jac_function_wrt_desvar: RHSJacType = None,
        adjoint_wrt_state: RHSJacType = None,
        adjoint_wrt_desvar: RHSJacType = None,
        solve_at_algorithm_times: bool = False,
        event_functions: Iterable[RHSFuncType] = (),
    ) -> None:
        """
        Args:
            func: The RHS function :math:`f`.
            initial_state: The initial state.
            times: The time interval of integration and  the times of interest where
                the state must be stored.
            jac_function_wrt_state: The Jacobian of :math:`f` with respect to state.
                Either a constant matrix
                or a function to compute it
                at a given time and state.
                If ``None``,
                it will be approximated.
            jac_function_wrt_desvar:  The Jacobian of :math:`f`
                with respect to the design variables.
                Either a constant matrix
                or a function to compute it
                at a given time and state.
                If ``None``,
                it will be approximated.
                Since the design variables are supposed fixed
                during the integration of the ODE,
                this Jacobian cannot be checked by the method :meth:`.check_jacobian`.
            adjoint_wrt_state: The adjoint relative to the state
                when using an adjoint-based ODE solver.
            adjoint_wrt_desvar: The adjoint relative to the design variables
                when using an adjoint-based ODE solver.
            solve_at_algorithm_times: Whether to solve the ODE chosen by the algorithm.
            event_functions: The event functions,
                for which the integration stops when they get equal to 0.
                If empty,
                the solver will solve the ODE for the entire assigned time interval.
        """  # noqa: D205, D212, D415
        self.rhs_function = func
        self.compute_trajectory = solve_at_algorithm_times or len(times) > 2

        self.jac_function_wrt_state = jac_function_wrt_state
        self.jac_function_wrt_desvar = jac_function_wrt_desvar

        self.adjoint_wrt_state = adjoint_wrt_state
        self.adjoint_wrt_desvar = adjoint_wrt_desvar

        self.initial_state = initial_state

        evaluation_times = times if len(times) > 2 else None
        self.update_times(float(times[0]), float(times[-1]), evaluation_times)

        self.event_functions = event_functions
        for event_function in event_functions:
            # Remind: event_function is a Python function.
            event_function.terminal = True

        self.solve_at_algorithm_times = solve_at_algorithm_times
        self.result = ODEResult(
            times=empty(0),
            state_trajectories=empty(0),
            n_func_evaluations=0,
            n_jac_evaluations=0,
            termination_time=0.0,
            terminal_event_index=None,
            final_state=empty(0),
            jac_wrt_desvar=empty(0),
            jac_wrt_initial_state=empty(0),
            algorithm_termination_message="",
            algorithm_has_converged=False,
            algorithm_name="",
            algorithm_settings={},
        )

        self.__jacobian_check_time = self.__time_interval.initial

    def check(self) -> None:
        """Ensure the parameters of the problem are consistent.

        Raises:
            ValueError: If the state and time shapes are inconsistent.
        """
        data = self.result.state_trajectories
        if data.size != 0 and data.shape[1] != self.result.times.size:
            msg = "Inconsistent state and time shapes."
            raise ValueError(msg)

    def check_jacobian(
        self,
        state: RealArray,
        time: float | None = None,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-6,
        error_max: float = 1e-8,
    ) -> None:
        """Check if the analytical Jacobian with respect to the state is correct.

        Args:
            state: The state.
            time: The time of evaluation of the function.
                If ``None``, use :attr:`.self.__time_interval.initial`.
            approximation_mode: The approximation mode.
            step: The step for the approximation of the gradients.
            error_max: The maximum value of the error.

        Raises:
            ValueError: Either if the approximation method is unknown,
                if the shapes of
                the analytical and approximated Jacobian matrices
                are inconsistent
                or if the analytical gradients are wrong.
        """
        if time is None:
            time = self.__time_interval.initial
        self.__jacobian_check_time = time

        function_of_state = MDOFunction(
            self._compute_func_of_state,
            name="f",
            jac=self._compute_jac_of_state,
        )
        function_of_state.check_grad(state, approximation_mode, step, error_max)

    @property
    def evaluation_times(self) -> RealArray | None:
        """The times of interest where the state is computed.

        If ``None``, the ODE is integrated in the interval [0, 1] by default.
        """
        return self.__evaluation_times

    @property
    def time_interval(self) -> RealArray | None:
        """The time interval in which the ODE is solved.

        If ``None``, the ODE is integrated in the interval [0, 1] by default.
        """
        return self.__time_interval

    def update_times(
        self,
        initial_time: float | None = None,
        final_time: float | None = None,
        times: RealArray | None = None,
    ) -> None:
        """Update the solution times of the ODE.

        Args:
            initial_time: The initial time.
            final_time: The final time.
            times: The vector of time instants (sorted in ascending order).

        Raises:
            ValueError: If the initial time is not lower than the final time.
        """
        t_0 = initial_time if initial_time is not None else self.__time_interval.initial
        t_f = final_time if final_time is not None else self.__time_interval.final

        if self.compute_trajectory and times is not None:
            t_0 = min(t_0, times[0])
            t_f = max(t_f, times[-1])

        if t_0 > t_f:
            msg = "The initial time must be lower than the final time."
            raise ValueError(msg)

        self.__time_interval = TimeInterval(initial=float(t_0), final=float(t_f))

        if self.compute_trajectory and times is not None:
            self.__evaluation_times = asarray(times)

    def _compute_func_of_state(self, state: RealArray) -> RealArray:
        """Evaluate the RHS function as a function of the state.

        Args:
            state: The state.

        Returns:
            The evaluation of the RHS function.
        """
        return asarray(self.rhs_function(self.__jacobian_check_time, state))

    def _compute_jac_of_state(self, state: RealArray) -> RealArray:
        """Evaluate the Jacobian with respect to the state as a function of the state.

        Args:
            state: The state.

        Returns:
            The evaluation of the Jacobian of the RHS function.

        Raises:
            AttributeError: If the jacobian with respect to the state is not available.

        """
        if self.jac_function_wrt_state is None:
            msg = "The function jac_function_wrt_state is not available."
            raise AttributeError(msg)

        if callable(self.jac_function_wrt_state):
            return asarray(
                self.jac_function_wrt_state(self.__jacobian_check_time, state)
            )
        return self.jac_function_wrt_state
