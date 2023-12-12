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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The main discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import eye
from numpy import newaxis
from numpy import zeros

from gemseo.problems.scalable.parametric.core.disciplines.base_discipline import (
    BaseDiscipline,
)
from gemseo.problems.scalable.parametric.core.variable_names import OBJECTIVE_NAME
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_constraint_name
from gemseo.problems.scalable.parametric.core.variable_names import get_coupling_name

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MainDiscipline(BaseDiscipline):
    r"""The main discipline of the scalable problem.

    It computes the objective :math:`x_0^Tx_0 + \sum_{i=1}^N y_i^Ty_i`. and the left-
    hand side of the constraints :math:`t_1-y_1\leq 0,\ldots,t_N-y_N\leq 0`.
    """

    __n_scalable_disciplines: int
    r"""The number of scalable disciplines :math:`N`."""

    __y_i_names: list[str]
    r"""The names of the coupling variables :math:`y_1,\ldots,y_N`."""

    __c_i_names: list[str]
    r"""The names of the constraint variables :math:`c_1,\ldots,c_N`."""

    __t_i: tuple[NDArray[float]]
    r"""The threshold vectors :math:`t_1,\ldots,t_N`."""

    def __init__(
        self,
        *t_i: NDArray[float],
        **default_input_values: NDArray[float],
    ) -> None:
        r"""
        Args:
            *t_i: The threshold vectors :math:`t_1,\ldots,t_N`.
            **default_input_values: The default values of the input variables.
        """  # noqa: D205 D212
        self.name = self.__class__.__name__
        self.input_names_to_default_values = default_input_values
        self.__n_scalable_disciplines = len(t_i)
        scalable_discipline_indices = range(1, self.__n_scalable_disciplines + 1)
        self.__y_i_names = [
            get_coupling_name(scalable_discipline_index)
            for scalable_discipline_index in scalable_discipline_indices
        ]
        self.__c_i_names = [
            get_constraint_name(scalable_discipline_index)
            for scalable_discipline_index in scalable_discipline_indices
        ]
        self.__y_i_names_to_default_values = {
            coupling_name: self.input_names_to_default_values[coupling_name]
            for coupling_name in self.__y_i_names
        }
        self.__t_i = t_i
        self.input_names = sorted(self.input_names_to_default_values.keys())
        self.output_names = [OBJECTIVE_NAME]
        self.output_names.extend(self.__c_i_names)
        self.names_to_sizes = {
            input_name: default_value.size
            for input_name, default_value in default_input_values.items()
        }
        for cstr_name, cpl_name in zip(self.__c_i_names, self.__y_i_names):
            self.names_to_sizes[cstr_name] = self.names_to_sizes[cpl_name]
        self.names_to_sizes[OBJECTIVE_NAME] = 1

    def __call__(
        self,
        x_0: NDArray[float] | None = None,
        compute_jacobian: bool = False,
        **y_i: NDArray[float],
    ) -> dict[str, NDArray[float] | dict[str, NDArray[float]]]:
        r"""Compute objective and constraints or their derivatives.

        Args:
            x_0: The value of the shared design variable :math:`x_0`.
                If ``None``, use the default one.
            compute_jacobian: Whether to compute the values of the objective and
                constraints, or their derivatives.
            **y_i: The values of the coupling variables :math:`y_1,\ldots,y_N`.
                If missing, use the default ones.

        Returns:
            Either the values of the objective and constraints or their derivatives.
        """
        if x_0 is None:
            x_0 = self.input_names_to_default_values[SHARED_DESIGN_VARIABLE_NAME]

        _y_i = self.__y_i_names_to_default_values.copy()
        _y_i.update(y_i)

        if compute_jacobian:
            jacobian = {}
            for output_name in self.output_names:
                jacobian[output_name] = {
                    input_name: zeros((
                        self.names_to_sizes[output_name],
                        self.names_to_sizes[input_name],
                    ))
                    for input_name in self.input_names
                }

            jacobian[OBJECTIVE_NAME][SHARED_DESIGN_VARIABLE_NAME] = 2 * x_0[newaxis, :]
            for y_i_name, c_i_name in zip(self.__y_i_names, self.__c_i_names):
                jacobian[OBJECTIVE_NAME][y_i_name] = 2 * _y_i[y_i_name][newaxis, :]
                jacobian[c_i_name][y_i_name] = -eye(self.names_to_sizes[y_i_name])

            return jacobian

        output_names_to_values = {
            OBJECTIVE_NAME: array([
                sum([(__y_i**2).sum() for __y_i in _y_i.values()]) + (x_0**2).sum()
            ])
        }
        for c_i_name, __y_i, t_i in zip(self.__c_i_names, _y_i.values(), self.__t_i):
            output_names_to_values[c_i_name] = t_i - __y_i

        return output_names_to_values
