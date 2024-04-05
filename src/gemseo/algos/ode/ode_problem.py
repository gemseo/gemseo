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
"""Ordinary differential equation problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from numpy import asarray
from numpy import empty

from gemseo.algos.base_problem import BaseProblem
from gemseo.algos.ode.ode_result import ODEResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.derivatives.approximation_modes import ApproximationMode

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

    from gemseo.typing import NumberArray


class ODEProblem(BaseProblem):
    r"""First-order ordinary differential equation (ODE).

    A first-order ODE is written as

    .. math:: \frac{ds(t)}{dt} = f(t, s(t)).

    where :math:`f` is called the right-hand side (RHS) of the ODE
    and :math:`s(t)` is the state vector at time :math:`t`.
    """

    rhs_function: Callable[[float, NumberArray], NumberArray]
    """The function :math:`f`."""

    jac: Callable[[float, NumberArray], NumberArray] | NumberArray | None
    """The function to compute the Jacobian of :math:`f`.

    If ``Callable``,
    the Jacobian is assumed to be dependent on time and state.
    It will be called as ``jac(time, state)`` as necessary.
    If ``NumberArray``,
    the Jacobian is assumed to be constant.
    If ``None``,
    the Jacobian will be approximated by finite differences.
    """

    jac_desvar: Callable[[float, NumberArray], NumberArray]
    """The function to compute the Jacobian of :math:`f` relative to the design
    variables."""  # noqa: E501

    adjoint_wrt_state: NumberArray
    """The adjoint of the problem relative to the state."""

    adjoint_wrt_desvar: NumberArray
    """The adjoint of the problem relative to the design variables."""

    initial_state: NumberArray
    r"""The initial conditions :math:`(t_0,s_0)` of the ODE."""

    __time_vector: NumberArray
    """The times at which the solution should be evaluated."""

    integration_interval: tuple[float, float]
    """The interval of integration."""

    result: ODEResult
    """The result of the ODE problem."""

    def __init__(
        self,
        func: Callable[[float, NumberArray], NumberArray],
        initial_state: ArrayLike,
        initial_time: float,
        final_time: float,
        jac: Callable[[float, NumberArray], NumberArray] | NumberArray | None = None,
        jac_desvar: Callable[[float, NumberArray], NumberArray] | None = None,
        adjoint_wrt_state: NumberArray | None = None,
        adjoint_wrt_desvar: NumberArray | None = None,
        time_vector: NumberArray | None = None,
    ) -> None:
        """
        Args:
            func: The RHS function :math:`f`. It will be called as `func(time, state)`
                as necessary.
            initial_state: The initial state of the ODE.
            initial_time: The start of the integration interval.
            final_time: The end of the integration interval.
            jac: The function to compute the Jacobian of :math:`f`.
                If ``Callable``,
                the Jacobian is assumed to be dependent on time and state.
                It will be called as ``jac(time, state)`` as necessary.
                If ``NumberArray``,
                the Jacobian is assumed to be constant.
                If ``None``,
                the Jacobian will be approximated by finite differences.
            jac_desvar: The function to compute the Jacobian of :math:`f`
                relative to the design variables.
                If ``None``,
                use a solver that doesn't require the adjoint.
            adjoint_wrt_state: The adjoint relative to the state.
                If ``None``,
                use a solver that doesn't require the adjoint.
            adjoint_wrt_desvar: The adjoint relative to the design variables.
                If ``None``,
                use a solver that doesn't require the adjoint.
            time_vector: The time vector for the solution.
                If ``None``,
                the solver will select times for which the computed solution is stored.
        """  # noqa: D205, D212, D415
        self.rhs_function = func
        self.jac = jac
        self.jac_desvar = jac_desvar
        self.adjoint_wrt_state = adjoint_wrt_state
        self.adjoint_wrt_desvar = adjoint_wrt_desvar
        self.initial_state = asarray(initial_state)
        self.__time_vector = time_vector
        self.integration_interval = (initial_time, final_time)
        self.result = ODEResult(
            time_vector=empty(0),
            state_vector=empty(0),
            n_func_evaluations=0,
            n_jac_evaluations=0,
            solver_message="",
            is_converged=False,
            solver_name="",
            solver_options={},
        )

    @property
    def time_vector(self):
        """The times at which the solution shall be evaluated."""
        return self.__time_vector

    def check(self) -> None:
        """Ensure the parameters of the problem are consistent.

        Raises:
            ValueError: If the state and time shapes are inconsistent.
        """
        if (
            self.result.state_vector.size != 0
            and self.result.state_vector.shape[1] != self.result.time_vector.size
        ):
            msg = "Inconsistent state and time shapes."
            raise ValueError(msg)

    def _func(self, state: NumberArray) -> NumberArray:
        """Evaluate :math:`f` at a given state.

        Args:
            state: The state of the system.

        Returns:
            The function :math:`f` at `state`.
        """
        return asarray(self.rhs_function(self.result.time_vector, state))

    def _jac(self, state: NumberArray) -> NumberArray:
        """Compute the Jacobian of :math:`f` at a given state.

        Args:
            state: The state of the system.

        Returns:
            The Jacobian of :math:`f` at `state`.
        """
        return asarray(self.jac(self.result.time_vector, state))

    def check_jacobian(
        self,
        state_vector: ArrayLike,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-6,
        error_max: float = 1e-8,
    ) -> None:
        """Check if the Jacobian function is correct.

        At a given state,
        compare the value of the Jacobian
        computed by the function provided by ther user
        to an approximated value
        computed by finite-differences for example.

        Args:
            state_vector: The state at which the Jacobian is checked.
            approximation_mode: The approximation mode.
            step: The step used to approximate the gradients.
            error_max: The error threshold above which the Jacobian is deemed to
                be incorrect.

        Raises:
            ValueError: When the Jacobian function is wrong.
        """
        if self.jac is not None:
            function = MDOFunction(self._func, "f", jac=self._jac)
            function.check_grad(
                asarray(state_vector), approximation_mode, step, error_max
            )
