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
from numpy import ndarray

from gemseo.algos.base_problem import BaseProblem
from gemseo.algos.ode.ode_result import ODEResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.utils.derivatives.approximation_modes import ApproximationMode

if TYPE_CHECKING:
    from numpy.typing import ArrayLike
    from numpy.typing import NDArray


class ODEProblem(BaseProblem):
    r"""First-order ordinary differential equation (ODE).

    .. math:: \frac{ds(t)}{dt} = f(t, s(t)).

    :math:`f` is called the right-hand side of the ODE.
    """

    rhs_function: Callable[[NDArray[float], NDArray[float]], NDArray[float]]
    """The right-hand side of the ODE."""

    jac: Callable[[NDArray[float], NDArray[float]], NDArray[float]]
    """The Jacobian function of the right-hand side of the ODE."""

    initial_state: NDArray[float]
    r"""The initial conditions :math:`(t_0,s_0)` of the ODE."""

    __time_vector: NDArray[float]
    """The times at which the solution should be evaluated."""

    integration_interval: tuple[float, float]
    """The interval of integration."""

    result: ODEResult
    """The result of the ODE problem."""

    def __init__(
        self,
        func: Callable[[NDArray[float], NDArray[float]], NDArray[float]]
        | NDArray[float],
        initial_state: ArrayLike,
        initial_time: float,
        final_time: float,
        jac: Callable[[NDArray[float], NDArray[float]], NDArray[float]]
        | NDArray[float]
        | None = None,
        time_vector: NDArray[float] | None = None,
    ) -> None:
        """
        Args:
            func: The right-hand side of the ODE.
            initial_state: The initial state of the ODE.
            initial_time: The start of the integration interval.
            final_time: The end of the integration interval.
            jac: The Jacobian of the right-hand side of the ODE.
            time_vector: The time vector for the solution.
        """  # noqa: D205, D212, D415
        self.rhs_function = func
        self.jac = jac
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
            raise ValueError("Inconsistent state and time shapes.")

    def _func(self, state) -> ndarray:
        return asarray(self.rhs_function(self.result.time_vector, state))

    def _jac(self, state) -> ndarray:
        return asarray(self.jac(self.result.time_vector, state))

    def check_jacobian(
        self,
        state_vector: ArrayLike,
        approximation_mode: ApproximationMode = ApproximationMode.FINITE_DIFFERENCES,
        step: float = 1e-6,
        error_max: float = 1e-8,
    ) -> None:
        """Check if the analytical jacobian is correct.

        Compare the value of the analytical jacobian to a finite-element
        approximation of the jacobian at user-specified points.

        Args:
            state_vector: The state vector at which the jacobian is checked.
            approximation_mode: The approximation mode.
            step: The step used to approximate the gradients.
            error_max: The error threshold above which the jacobian is deemed to
                be incorrect.

        Raises:
            ValueError: Either if the approximation method is unknown, if the
                shapes of the analytical and approximated Jacobian matrices
                are inconsistent or if the analytical gradients are wrong.

        Returns:
            Whether the jacobian is correct.
        """
        if self.jac is not None:
            function = MDOFunction(self._func, "f", jac=self._jac)
            function.check_grad(
                asarray(state_vector), approximation_mode, step, error_max
            )
