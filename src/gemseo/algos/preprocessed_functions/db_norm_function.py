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
"""A database-assisted function."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable

from gemseo.algos.preprocessed_functions.base_preprocessed_function import (
    BasePreprocessedFunction,
)
from gemseo.algos.preprocessed_functions.norm_function_mixin import NormFunctionMixin
from gemseo.algos.preprocessed_functions.utils import check_function_output_includes_nan
from gemseo.algos.stop_criteria import MaxIterReachedException

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.database import Database
    from gemseo.algos.design_space import DesignSpace
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction
    from gemseo.utils.compatibility.scipy import ArrayType


class DBNormFunction(NormFunctionMixin, BasePreprocessedFunction):
    """A database-assisted function."""

    __database: Database
    """The database to store and retrieve function evaluations."""

    __design_space: DesignSpace
    """The design space attached to the evaluation problem."""

    __evaluation_problem: EvaluationProblem
    """The evaluation problem."""

    __gradient_name: str
    """The name of the gradient function."""

    __normalize_grad: Callable[[ArrayType], ArrayType]
    """Normalize an unnormalized gradient."""

    __normalize_vect: Callable[[ArrayType, bool, ArrayType | None], ArrayType]
    """Normalize a vector of the design space."""

    __unnormalize_vect: Callable[[ArrayType, bool, bool, ArrayType | None], ArrayType]
    """Unnormalize a normalized vector of the design space."""

    def __init__(
        self, function: MDOFunction, evaluation_problem: EvaluationProblem
    ) -> None:
        """
        Args:
            evaluation_problem: The evaluation problem
                to which ``function`` is attached.
        """  # noqa: D205, D212, D415
        super().__init__(function)
        design_space = evaluation_problem.design_space
        self.__evaluation_problem = evaluation_problem
        self.__database = evaluation_problem.database
        self.__unnormalize_vect = design_space.unnormalize_vect
        self.__normalize_vect = design_space.normalize_vect
        self.__unnormalize_grad = design_space.unnormalize_grad
        self.__gradient_name = self.__database.get_gradient_name(self.name)
        self.__design_space = evaluation_problem.design_space

    def _compute_output(self, x_vect: ndarray) -> ndarray:
        check_function_output_includes_nan(x_vect)
        xn_vect = x_vect
        xu_vect = self.__unnormalize_vect(xn_vect)
        database = self.__database
        hashed_xu = database.get_hashable_ndarray(xu_vect)
        value = database.get_function_value(self.name, hashed_xu)
        if value is None:
            if (
                not database.get(hashed_xu)
                and self.__evaluation_problem.evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            value = self._original_func(xn_vect)
            check_function_output_includes_nan(
                value, self.__evaluation_problem.stop_if_nan, self.name, xu_vect
            )
            database.store(hashed_xu, {self.name: value})

        return value

    def _compute_jacobian(self, x_vect: ndarray) -> ndarray:
        check_function_output_includes_nan(x_vect)
        xn_vect = x_vect
        xu_vect = self.__unnormalize_vect(xn_vect)
        database = self.__database
        hashed_xu = database.get_hashable_ndarray(xu_vect)
        jac_u = database.get_function_value(self.__gradient_name, hashed_xu)
        if jac_u is None:
            if (
                not database.get(hashed_xu)
                and self.__evaluation_problem.evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            jac_n = self._original_jac(xn_vect)
            jac_u = self.__unnormalize_grad(jac_n)
            check_function_output_includes_nan(
                jac_u.data,
                self.__evaluation_problem.stop_if_nan,
                self.__gradient_name,
                xu_vect,
            )
            database.store(hashed_xu, {self.__gradient_name: jac_u})
        else:
            jac_n = self.__design_space.normalize_grad(jac_u)

        return jac_n.real
