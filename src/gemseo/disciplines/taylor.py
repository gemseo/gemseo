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
"""A discipline to create Taylor polynomials from another discipline."""

from __future__ import annotations

from copy import deepcopy
from types import MappingProxyType
from typing import TYPE_CHECKING

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class TaylorDiscipline(MDODiscipline):
    r"""The first-order polynomial of a discipline.

    The first-order polynomial
    of a function :math:`f`
    at an expansion point :math:`a`
    is :math:`f(a)+\sum_{i=1}^d\frac{\partial f(a)}{\partial x_i}(x_i-a_i)`.
    """

    __offset: Mapping[str, NDArray[float]]
    """The offset of the polynomial."""

    def __init__(
        self,
        discipline: MDODiscipline,
        input_data: Mapping[str, NDArray[float]] = MappingProxyType({}),
        name: str = "",
    ) -> None:
        """
        Args:
            discipline: The discipline to be approximated by a Taylor polynomial.
            input_data: The point of expansion.
                If empty, use the default inputs of ``discipline``.

        Raises:
            ValueError: If neither ``input_data`` nor ``discipline.default_inputs``
                is specified.
        """  # noqa: D205 D212
        input_names = set(discipline.get_input_data_names())
        if (
            (input_data and (input_data.keys() < input_names))
            or not input_data
            and discipline.default_inputs.keys() < input_names
        ):
            raise ValueError(
                "All the discipline input values must be "
                "specified either in input_data or in discipline.default_inputs."
            )

        discipline.linearize(compute_all_jacobians=True, input_data=input_data)
        super().__init__(name=name)
        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(discipline.get_output_data_names())
        self.default_inputs = input_data or discipline.default_inputs
        self.__offset = {}
        for output_name in self.get_output_data_names():
            self.__offset[output_name] = discipline.local_data[output_name] - sum(
                discipline.jac[output_name][input_name] @ input_value
                for input_name, input_value in self.default_inputs.items()
            )
        self.jac = deepcopy(discipline.jac)
        self._is_linearized = True

    def _run(self) -> None:
        input_data = self.get_input_data()
        for output_name in self.get_output_data_names():
            self.local_data[output_name] = self.__offset[output_name] + sum(
                self.jac[output_name][input_name] @ input_value
                for input_name, input_value in input_data.items()
            )
