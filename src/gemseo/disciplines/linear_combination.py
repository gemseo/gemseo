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

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable


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

    Notes:
        By default,
        the :class:`.LinearCombination` simply sums the inputs.

    Examples:
        >>> discipline = LinearCombination(["alpha", "beta", "gamma"], "delta",
                input_coefficients={"alpha": 1.,"beta": 2.,"gamma": 3.})
        >>> input_data = {"alpha": array([1.0]), "beta": array([1.0]),
                "gamma": array([1.0])}
        >>> discipline.execute(input_data)
        >>> delta = discipline.local_data["delta"]  # delta = array([6.])
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
        input_coefficients: dict[str, float] | None = None,
        offset: float = 0.0,
        input_size: int | None = None,
    ) -> None:
        """
        Args:
            input_names: The names of input variables.
            output_name: The name of the output variable.
            input_coefficients: The coefficients related to the input variables.
                If ``None``, use 1 for all the input variables.
            offset: The output value when all the input variables are equal to zero.
            input_size: The size of the inputs.
                If ``None``, the default inputs are initialized with size 1 arrays.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names([output_name])

        default_size = 1 if input_size is None else input_size
        self.default_inputs.update({
            input_name: zeros(default_size) for input_name in input_names
        })

        self.__coefficients = {input_name: 1.0 for input_name in input_names}
        if input_coefficients:
            self.__coefficients.update(input_coefficients)

        self.__offset = offset
        self.__output_name = output_name

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
        self._init_jacobian(inputs, outputs, init_type=self.InitJacobianType.SPARSE)
        identity = eye(self.local_data[self.__output_name].size, format="csr")

        jac = self.jac[self.__output_name]
        for input_name in self.get_input_data_names():
            jac[input_name] = self.__coefficients[input_name] * identity
