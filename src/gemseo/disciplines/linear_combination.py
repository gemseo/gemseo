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

from typing import TYPE_CHECKING

from numpy import zeros
from scipy.sparse import eye

from gemseo.core.discipline import Discipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.typing import StrKeyMapping


class LinearCombination(Discipline):
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

    Notes:
        By default,
        the :class:`.LinearCombination` simply sums the inputs.

    Examples:
        >>> discipline = LinearCombination(["alpha", "beta", "gamma"], "delta",
                input_coefficients={"alpha": 1.,"beta": 2.,"gamma": 3.})
        >>> input_data = {"alpha": array([1.0]), "beta": array([1.0]),
                "gamma": array([1.0])}
        >>> discipline.execute(input_data)
        >>> delta = discipline.io.data["delta"]  # delta = array([6.])
    """

    __offset: float
    r"""The offset :math:`a_0` in :math:`a_0+\sum_{i=1}^d a_i x_i`."""

    __coefficients: dict[str, float]
    r"""The coefficients :math:`a_1,\ldots,a_d` in :math:`a_0+\sum_{i=1}^d a_i x_i`."""

    __output_name: str
    """The name of the output."""

    def __init__(
        self,
        input_names: Iterable[str],
        output_name: str,
        input_coefficients: Mapping[str, float] = READ_ONLY_EMPTY_DICT,
        offset: float = 0.0,
        input_size: int = 1,
        average: bool = False,
    ) -> None:
        """
        Args:
            input_names: The names of the input variables.
            output_name: The name of the output variable.
            input_coefficients: The coefficients related to the input variables.
                If empty,
                use 1 for all the :math:`d` input variables
                when ``average`` is ``False``.
                or :math:`1/d` when ``average`` is ``True``.
            offset: The output value when all the input variables are equal to zero.
            input_size: The size of the inputs.
            average: Whether to average the inputs
                when ``input_coefficients`` is empty,
        """  # noqa: D205, D212, D415
        super().__init__()
        input_grammar = self.io.input_grammar
        input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names((output_name,))

        # TODO: API: remove this line in GEMSEO v7.
        default_size = 1 if input_size is None else input_size
        input_grammar.defaults.update({
            input_name: zeros(default_size) for input_name in input_names
        })

        if input_coefficients:
            self.__coefficients = dict.fromkeys(input_names, 1.0)
            self.__coefficients.update(input_coefficients)
        elif average:
            self.__coefficients = dict.fromkeys(input_names, 1.0 / len(input_grammar))
        else:
            self.__coefficients = dict.fromkeys(input_names, 1.0)

        self.__offset = offset
        self.__output_name = output_name

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        output_data = {self.__output_name: self.__offset}
        for input_name, input_value in input_data.items():
            output_data[self.__output_name] += (
                self.__coefficients[input_name] * input_value
            )
        return output_data

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._init_jacobian(
            input_names, output_names, init_type=self.InitJacobianType.SPARSE
        )
        identity = eye(self.io.data[self.__output_name].size, format="csr")

        jac = self.jac[self.__output_name]
        for input_name in self.io.input_grammar:
            jac[input_name] = self.__coefficients[input_name] * identity
