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
"""A class to check the Jacobian of the right-hand side of an ODE."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import asarray

from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.algos.ode.ode_problem import DifferentiationFunctions
    from gemseo.algos.ode.ode_problem import RHSFuncType
    from gemseo.typing import RealArray


class RHSJacobianChecking:
    """A class to check the Jacobian of the right-hand side of an ODE."""

    default_time: float
    """The default time value."""

    __jac: DifferentiationFunctions
    """The functions to compute the Jacobian of the RHS."""

    __rhs_function: RHSFuncType
    """The function to compute the right-hand side (RHS) of the ODE."""

    function_of_state: MDOFunction
    """A function computing the RHS given the state for the default time."""

    function_of_time_and_state: MDOFunction
    """A function computing the RHS given the time and state."""

    def __init__(
        self,
        rhs_function: RHSFuncType,
        jac: DifferentiationFunctions,
        default_time: float = 0.0,
    ) -> None:
        """
        Args:
            rhs_function: The function to compute the right-hand side (RHS) of the ODE.
            jac: The functions to compute the Jacobian of the RHS.
            default_time: The default time value.
        """  # noqa: D205, D212, D415
        self.default_time = default_time
        self.__jac = jac
        self.__rhs_function = rhs_function
        self.function_of_time_and_state = MDOFunction(
            self._func_of_time_and_state,
            name="f",
            jac=self._jac_of_time_and_state,
        )
        self.function_of_state = MDOFunction(
            self._func_of_state,
            name="f",
            jac=self._jac_of_state,
        )

    def _func_of_time_and_state(self, time_and_state: RealArray) -> RealArray:
        """Evaluate the RHS function at given time and state.

        Args:
            time_and_state: The time and state.

        Returns:
            The RHS value.
        """
        return asarray(self.__rhs_function(time_and_state[0], time_and_state[1:]))

    def _jac_of_time_and_state(self, time_and_state) -> RealArray:
        """Differentiate the RHS function at given time and state.

        Args:
            time_and_state: The time and state.

        Returns:
            The Jacobian of the RHS function.
        """
        if self.__jac.time_state is None:
            msg = "The function jac.time_state is not available."
            raise ValueError(msg)

        if callable(self.__jac.time_state):
            return asarray(self.__jac.time_state(time_and_state[0], time_and_state[1:]))
        return self.__jac.time_state

    def _func_of_state(self, state: RealArray) -> RealArray:
        """Evaluate the RHS function at a given state and a default time.

        Args:
            state: The state.

        Returns:
            The Jacobian of the RHS function.
        """
        return asarray(self.__rhs_function(self.default_time, state))

    def _jac_of_state(self, state: RealArray) -> RealArray:
        """Differentiate the RHS function at a given state and a default time.

        Args:
            state: The state.

        Returns:
            The Jacobian of the RHS function.
        """
        if self.__jac.state is None:
            msg = "The function jac.state is not available."
            raise ValueError(msg)

        if callable(self.__jac.state):
            return asarray(self.__jac.state(self.default_time, state))
        return self.__jac.state
