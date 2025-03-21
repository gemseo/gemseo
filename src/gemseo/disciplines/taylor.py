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
from typing import TYPE_CHECKING

from gemseo.core.discipline import Discipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray

    from gemseo.typing import StrKeyMapping


class TaylorDiscipline(Discipline):
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
        discipline: Discipline,
        input_data: Mapping[str, NDArray[float]] = READ_ONLY_EMPTY_DICT,
        name: str = "",
    ) -> None:
        """
        Args:
            discipline: The discipline to be approximated by a Taylor polynomial.
            input_data: The point of expansion.
                If empty, use the default inputs of ``discipline``.

        Raises:
            ValueError: If neither ``input_data`` nor
            ``discipline.io.input_grammar.defaults`` is specified.
        """  # noqa: D205 D212
        input_names = set(discipline.io.input_grammar)
        if (input_data and (input_data.keys() < input_names)) or (
            not input_data and discipline.io.input_grammar.defaults.keys() < input_names
        ):
            msg = (
                "All the discipline input values must be specified either in "
                "input_data or in discipline.io.input_grammar.defaults."
            )
            raise ValueError(msg)

        discipline.linearize(compute_all_jacobians=True, input_data=input_data)
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(discipline.io.input_grammar)
        self.io.output_grammar.update_from_names(discipline.io.output_grammar)
        self.io.input_grammar.descriptions = discipline.io.input_grammar.descriptions
        self.io.output_grammar.descriptions = discipline.io.output_grammar.descriptions
        self.io.input_grammar.defaults = (
            input_data or discipline.io.input_grammar.defaults
        )
        self.__offset = {}
        for output_name in self.io.output_grammar:
            defaults = self.io.input_grammar.defaults
            self.__offset[output_name] = discipline.io.data[output_name] - sum(
                discipline.jac[output_name][input_name] @ defaults[input_name]
                for input_name in sorted(defaults)
            )
        self.jac = deepcopy(discipline.jac)
        self._has_jacobian = True

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        output_data = {}
        for output_name in self.io.output_grammar:
            output_data[output_name] = self.__offset[output_name] + sum(
                self.jac[output_name][input_name] @ input_data[input_name]
                for input_name in sorted(input_data)
            )
        return output_data
