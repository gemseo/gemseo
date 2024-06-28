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
from gemseo.algos.preprocessed_functions.utils import check_function_output_includes_nan
from gemseo.algos.stop_criteria import MaxIterReachedException

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.algos.database import Database
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.core.mdofunctions.mdo_function import MDOFunction


class DBFunction(BasePreprocessedFunction):
    """A database-assisted function."""

    __database: Database
    """The database to store and retrieve function evaluations."""

    __gradient_name: str
    """The name of the gradient function."""

    __evaluation_problem: EvaluationProblem
    """The evaluation problem to which ``function`` is attached."""

    def __init__(
        self, function: MDOFunction, evaluation_problem: EvaluationProblem
    ) -> None:
        """
        Args:
            evaluation_problem: The evaluation problem
                to which ``function`` is attached.
        """  # noqa: D205, D212, D415
        super().__init__(function)
        self.__expects_normalized_inputs = function.expects_normalized_inputs
        self.__evaluation_problem = evaluation_problem
        self.__database = evaluation_problem.database
        self.__gradient_name = self.__database.get_gradient_name(self.name)

    def _compute_output(self, x_vect: ndarray) -> ndarray:
        return self._preprocess(x_vect, self.name, self._original_func)

    def _compute_jacobian(self, x_vect: ndarray) -> ndarray:
        return self._preprocess(x_vect, self.__gradient_name, self._original_jac).real

    def _preprocess(
        self, x_vect: ndarray, name: str, func: Callable[[ndarray], ndarray]
    ) -> ndarray:
        """A preprocessed function.

        Args:
            x_vect: The input value at which to evaluate the function.
            name: The name of the function.
            func: The function.

        Returns:
            The value of the output of the preprocessed function.
        """
        check_function_output_includes_nan(x_vect)
        database = self.__database
        hashed_xu = database.get_hashable_ndarray(x_vect)
        value = database.get_function_value(name, hashed_xu)
        if value is None:
            if (
                not database.get(hashed_xu)
                and self.__evaluation_problem.evaluation_counter.maximum_is_reached
            ):
                raise MaxIterReachedException

            value = func(x_vect)
            check_function_output_includes_nan(
                value, self.__evaluation_problem.stop_if_nan, name, x_vect
            )
            database.store(hashed_xu, {name: value})
        return value
