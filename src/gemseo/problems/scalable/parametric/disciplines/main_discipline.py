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

from gemseo.problems.scalable.parametric.core.disciplines.main_discipline import (
    MainDiscipline as _MainDiscipline,
)
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_coupling_name
from gemseo.problems.scalable.parametric.disciplines.base_discipline import (
    BaseDiscipline,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import NDArray


class MainDiscipline(BaseDiscipline):
    r"""The main discipline of the scalable problem.

    It computes the objective :math:`x_0^Tx_0 + \sum_{i=1}^N y_i^Ty_i`. and the left-
    hand side of the constraints :math:`t_1-y_1\leq 0,\ldots,t_N-y_N\leq 0`.
    """

    _CORE_DISCIPLINE_CLASS = _MainDiscipline

    __n_scalable_disciplines: int
    r"""The number of scalable disciplines :math:`N`."""

    __y_i_names: list[str]
    r"""The names of the coupling variables :math:`y_1,\ldots,y_N`."""

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
        self.__n_scalable_disciplines = len(t_i)
        self.__y_i_names = [
            get_coupling_name(index) for index in range(1, len(t_i) + 1)
        ]
        super().__init__(*t_i, **default_input_values)

    def _run(self) -> None:
        self.store_local_data(
            **self._discipline(
                self._local_data[SHARED_DESIGN_VARIABLE_NAME],
                **{
                    y_i_name: self._local_data[y_i_name]
                    for y_i_name in self.__y_i_names
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
            self.local_data[SHARED_DESIGN_VARIABLE_NAME],
            compute_jacobian=True,
            **{y_i_name: self.local_data[y_i_name] for y_i_name in self.__y_i_names},
        )
        for output_name in jac:
            self_sub_jac = self.jac[output_name]
            sub_jac = jac[output_name]
            for input_name in jac[output_name]:
                self_sub_jac[input_name] = sub_jac[input_name]
