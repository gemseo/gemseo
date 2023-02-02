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
"""Discipline computing a linear combination of its inputs."""
from __future__ import annotations

from typing import Iterable

from numpy import eye

from gemseo.core.discipline import MDODiscipline


class LinearCombination(MDODiscipline):
    r"""Discipline computing a linear combination of its inputs.

    The user can specify the coefficients related to the variables
    as well as the offset.

    E.g.,
    a discipline
    computing the output :math:`y`
    from :math:`d` inputs :math:`x_1,\ldots,x_d`
    with the function
    :math:`f(x_1,\ldots,x_d)=a_0+\sum_{i=1}^d a_i x_i`.

    When the offset :math:`a_0` is equal to 0
    and the coefficients :math:`a_1,\ldots,a_d` are equal to 1,
    the discipline simply sums the inputs.

    Note:
        By default,
        the :class:`.LinearCombination` simply sums the inputs.

    Example:
        >>> discipline = LinearCombination(["alpha", "beta", "gamma"], "delta",
        input_coefficients={"beta": 2.})
        >>> input_data = {"alpha": array([1.0]), "beta": array([2.0])}
        >>> discipline.execute(input_data)
        >>> delta = discipline.local_data["delta"]  # delta = array([5.])
    """
    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_LinearCombination__offset",
        "_LinearCombination__coefficients",
        "_LinearCombination__output_name",
    )

    def __init__(
        self,
        input_names: Iterable[str],
        output_name: str,
        input_coefficients: dict[str, float] = None,
        offset: float = 0.0,
    ):
        """
        Args:
            input_names: The names of input variables.
            output_name: The name of the output variable.
            input_coefficients: The coefficients related to the input variables.
                If ``None``, use 1 for all the input variables.
            offset: The output value when all the input variables are equal to zero.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.__offset = offset
        self.__coefficients = input_coefficients
        self.__output_name = output_name
        self.input_grammar.update(list(input_names))
        self.output_grammar.update([output_name])
        self.__coefficients = {name: 1.0 for name in self.get_input_data_names()}
        if input_coefficients:
            self.__coefficients.update(input_coefficients)

    def _run(self) -> None:
        self.local_data[self.__output_name] = self.__offset
        for input_name, input_value in self.get_input_data().items():
            self.local_data[self.__output_name] += (
                self.__coefficients[input_name] * input_value
            )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(with_zeros=True)
        self.jac = {}
        jac = self.jac[self.__output_name] = {}
        one_matrix = eye(self.local_data[self.__output_name].size)
        for input_name in self.get_input_data_names():
            jac[input_name] = self.__coefficients[input_name] * one_matrix
