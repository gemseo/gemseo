# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

import logging
from typing import TYPE_CHECKING

from numpy import any as np_any
from numpy import isnan as np_isnan
from numpy import ndarray

from gemseo.algos.database import Database
from gemseo.algos.stop_criteria import (
    DesvarIsNan,
    FunctionIsNan,
    MaxIterReachedException,
)
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.algos.opt_problem import OptimizationProblem

LOGGER = logging.getLogger(__name__)


class NormDBFunction(MDOFunction):
    """An :class:`.MDOFunction` object to be evaluated from a database."""

    def __init__(
        self,
        orig_func,  # type: MDOFunction
        normalize,  # type: bool
        is_observable,  # type: bool
        optimization_problem,  # type: OptimizationProblem
    ):  # type: (...) -> None
        """
        Args:
            orig_func: The original function to be wrapped.
            normalize: If True, then normalize the function's input vector.
            is_observable: If True, new_iter_listeners are not called
                when function is called (avoid recursive call).
            optimization_problem: The optimization problem object that contains
                the function.
        """
        self.__normalize = normalize
        self.__orig_func = orig_func
        self.__is_observable = is_observable
        self.__optimization_problem = optimization_problem

        super(NormDBFunction, self).__init__(
            self._func,
            orig_func.name,
            jac=self._jac,
            f_type=orig_func.f_type,
            expr=orig_func.expr,
            args=orig_func.args,
            dim=orig_func.dim,
            outvars=orig_func.outvars,
        )

    def _func(
        self,
        x_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Compute the function to be passed to the optimizer.

        Args:
            x_vect: The value of the design variables.

        Returns:
            The evaluation of the function for this value of the design variables.

        Raises:
            DesvarIsNan: If the design variables contain a NaN value.
            FunctionIsNan: If a function returns a NaN value when evaluated.
            MaxIterReachedException: If the maximum number of iterations has been
                reached.
        """
        if np_any(np_isnan(x_vect)):
            raise DesvarIsNan("Design Variables contain a NaN value: {}".format(x_vect))

        if self.__normalize:
            xn_vect = x_vect
            xu_vect = self.__optimization_problem.design_space.unnormalize_vect(xn_vect)
        else:
            xu_vect = x_vect
            xn_vect = self.__optimization_problem.design_space.normalize_vect(xu_vect)

        # try to retrieve the evaluation
        value = self.__optimization_problem.database.get_f_of_x(
            self.__orig_func.name, xu_vect
        )

        if value is None:
            new_eval = self.__optimization_problem.database.is_new_eval(xu_vect)
            if new_eval and self.__optimization_problem.is_max_iter_reached():
                raise MaxIterReachedException()

            # if not evaluated yet, evaluate
            if self.__normalize:
                value = self.__orig_func(xn_vect)
            else:
                value = self.__orig_func(xu_vect)
            if self.__optimization_problem.stop_if_nan and np_any(np_isnan(value)):
                raise FunctionIsNan(
                    "The function {} is NaN for x={}".format(
                        self.__orig_func.name, xu_vect
                    )
                )
            func_name_to_value = {self.__orig_func.name: value}
            # store (x, f(x)) in database
            self.__optimization_problem.database.store(xu_vect, func_name_to_value)

        return value

    def _jac(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> ndarray
        """Compute the gradient of the function to be passed to the optimizer.

        Args:
            x_vect: The value of the design variables.

        Returns:
            The evaluation of the gradient for this value of the design variables.

        Raises:
            FunctionIsNan: If the design variables contain a NaN value.
                If the evaluation of the jacobian results in a NaN value.
        """
        if np_any(np_isnan(x_vect)):
            raise FunctionIsNan(
                "Design Variables contain a NaN value: {}".format(x_vect)
            )
        if self.__normalize:
            xn_vect = x_vect
            xu_vect = self.__optimization_problem.design_space.unnormalize_vect(xn_vect)
        else:
            xu_vect = x_vect
            xn_vect = self.__optimization_problem.design_space.normalize_vect(xu_vect)

        # try to retrieve the evaluation
        jac_u = self.__optimization_problem.database.get_f_of_x(
            Database.get_gradient_name(self.__orig_func.name), xu_vect
        )

        if jac_u is None:

            new_eval = self.__optimization_problem.database.is_new_eval(xu_vect)
            if new_eval and self.__optimization_problem.is_max_iter_reached():
                raise MaxIterReachedException()

            # if not evaluated yet, evaluate
            if self.__normalize:
                jac_n = self.__orig_func.jac(xn_vect).real
                jac_u = self.__optimization_problem.design_space.unnormalize_grad(jac_n)

            else:
                jac_u = self.__orig_func.jac(xu_vect).real
                jac_n = self.__optimization_problem.design_space.normalize_grad(jac_u)
            if np_any(np_isnan(jac_n)) and self.__optimization_problem.stop_if_nan:
                raise FunctionIsNan(
                    "Function {}'s Jacobian is NaN "
                    "for x={}".format(self.__orig_func.name, xu_vect)
                )
            func_name_to_value = {
                Database.get_gradient_name(self.__orig_func.name): jac_u
            }
            # store (x, j(x)) in database
            self.__optimization_problem.database.store(xu_vect, func_name_to_value)
        else:
            jac_n = self.__optimization_problem.design_space.normalize_grad(jac_u)

        if self.__normalize:
            return jac_n

        else:
            return jac_u
