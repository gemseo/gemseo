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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)
#        :author: Gilberto Ruiz Jimenez
"""An MDOFunction subclass to support formulations."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import ndarray

from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class NormFunction(MDOFunction):
    """An :class:`.MDOFunction` object to be evaluated from an input vector."""

    def __init__(
        self,
        orig_func: MDOFunction,
        normalize: bool,
        round_ints: bool,
        optimization_problem: OptimizationProblem,
    ) -> None:
        """
        Args:
            orig_func: The original function.
            normalize: Whether to unnormalize the input vector
                before evaluating the original function.
            round_ints: If True, then round the integer variables.
            optimization_problem: The optimization problem object that contains
                the function.

        Raises:
            ValueError: If the original function does not provide a Jacobian matrix.
        """  # noqa: D205, D212, D415
        self.__normalize = normalize
        self.__orig_func = orig_func
        self.__round_ints = round_ints
        self.__optimization_problem = optimization_problem
        # For performance
        design_space = self.__optimization_problem.design_space
        self.__unnormalize_vect = design_space.unnormalize_vect
        self.__round_vect = design_space.round_vect
        self.__normalize_grad = design_space.normalize_grad
        self.__evaluate_orig_func = self.__orig_func.evaluate

        super().__init__(
            self._func_to_wrap,
            name=orig_func.name,
            jac=self._jac_to_wrap,
            f_type=orig_func.f_type,
            expr=orig_func.expr,
            args=orig_func.args,
            dim=orig_func.dim,
            outvars=orig_func.outvars,
            special_repr=orig_func.special_repr,
        )

    def _func_to_wrap(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Evaluate the original function.

        Args:
            x_vect: An input vector.

        Returns:
            The value of the function at this input vector.
        """
        if self.__normalize:
            x_vect = self.__unnormalize_vect(x_vect)
        if self.__round_ints:
            x_vect = self.__round_vect(x_vect)
        return self.__evaluate_orig_func(x_vect)

    def _jac_to_wrap(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Evaluate the gradient of the original function.

        Args:
            x_vect: An input vector.

        Returns:
            The value of the gradient of the original function at this input vector.

        Raises:
            ValueError: If the original function does not provide a Jacobian matrix.
        """
        if not self.__orig_func.has_jac():
            raise ValueError(
                "Selected user gradient but function {} "
                "has no Jacobian matrix !".format(self.__orig_func)
            )
        if self.__normalize:
            x_vect = self.__unnormalize_vect(x_vect)
        if self.__round_ints:
            x_vect = self.__round_vect(x_vect)
        g_u = self.__orig_func.jac(x_vect)
        if self.__normalize:
            return self.__normalize_grad(g_u)
        return g_u

    @property
    def expects_normalized_inputs(self) -> bool:  # noqa:D102
        return self.__normalize
