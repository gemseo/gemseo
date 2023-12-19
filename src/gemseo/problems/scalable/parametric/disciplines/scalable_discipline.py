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
"""The scalable discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.problems.scalable.parametric.core.disciplines.scalable_discipline import (
    ScalableDiscipline as _ScalableDiscipline,
)
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_u_local_name
from gemseo.problems.scalable.parametric.core.variable_names import get_x_local_name
from gemseo.problems.scalable.parametric.disciplines.base_discipline import (
    BaseDiscipline,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import NDArray


class ScalableDiscipline(BaseDiscipline):
    r"""A scalable discipline.

    It computes the output
    :math:`y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j`.
    """

    _CORE_DISCIPLINE_CLASS = _ScalableDiscipline

    __x_i_name: str
    r"""The name of the local design variable :math:`x_i`."""

    __u_i_name: str
    r"""The name of the local uncertain variable :math:`u_i`."""

    def __init__(
        self,
        index: int,
        a_i: NDArray,
        D_i0: NDArray,  # noqa: N803
        D_ii: NDArray,  # noqa: N803
        C_ij: Mapping[str, NDArray[float]],  # noqa: N803
        **default_input_values: NDArray[float],
    ) -> None:
        r"""
        Args:
            index: The index :math:`i` of the scalable discipline.
            a_i: The offset vector :math:`a_i`.
            D_i0: The coefficient matrix :math:`D_{i,0}`
                to multiply the shared design variable :math:`x_0`.
            D_ii: The coefficient matrix :math:`D_{i,i}`
                to multiply the local design variable :math:`x_i`.
            C_ij: The coefficient matrices
                :math:`\left(C_{i,j}\right)_{j=1\atop j\neq i}^N`
                where :math:`C_{i,j}` is used
                to multiply the coupling variable :math:`y_j`.
            **default_input_values: The default values of the input variables.
        """  # noqa: D205 D212
        super().__init__(index, a_i, D_i0, D_ii, C_ij, **default_input_values)
        self.__u_i_name = get_u_local_name(self._discipline.index)
        self.__x_i_name = get_x_local_name(self._discipline.index)

    def _run(self) -> None:
        if self.__u_i_name in self.input_grammar:
            u_i = self.get_inputs_by_name(self.__u_i_name)
        else:
            u_i = 0
        self.store_local_data(
            **self._discipline(
                self._local_data[SHARED_DESIGN_VARIABLE_NAME],
                self._local_data[self.__x_i_name],
                u_i,
                **{
                    y_j_name: self._local_data[y_j_name]
                    for y_j_name in self._discipline.coefficients.C_ij
                },
            )
        )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._init_jacobian(inputs, outputs)
        jac = self._discipline(
            self._local_data[SHARED_DESIGN_VARIABLE_NAME],
            self._local_data[self.__x_i_name],
            {
                y_j_name: self._local_data[y_j_name]
                for y_j_name in self._discipline.coefficients.C_ij
            },
            compute_jacobian=True,
        )
        for output_name in jac:
            sub_jac = jac[output_name]
            self_sub_jac = self.jac[output_name]
            for input_name in jac[output_name]:
                self_sub_jac[input_name] = sub_jac[input_name]
