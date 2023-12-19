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
"""A scalable discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

from numpy import eye
from numpy import zeros

from gemseo.problems.scalable.parametric.core.disciplines.base_discipline import (
    BaseDiscipline,
)
from gemseo.problems.scalable.parametric.core.variable_names import (
    SHARED_DESIGN_VARIABLE_NAME,
)
from gemseo.problems.scalable.parametric.core.variable_names import get_coupling_name
from gemseo.problems.scalable.parametric.core.variable_names import get_u_local_name
from gemseo.problems.scalable.parametric.core.variable_names import get_x_local_name

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy.typing import NDArray


class Coefficients(NamedTuple):
    r"""The coefficients of a scalable discipline.

    The output of a scalable discipline indexed by :math:`i` is computed as
    :math:`y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^NC_{i,j}y_j`.
    """

    a_i: NDArray[float]
    r"""The coefficient vector :math:`a_i`."""

    D_i0: NDArray[float]
    r"""The coefficient matrix :math:`D_{i,0}` to multiply :math:`x_0`."""

    D_ii: NDArray[float]
    r"""The coefficient matrix :math:`D_{i,i}` to multiply :math:`x_i`."""

    C_ij: Mapping[str, NDArray[float]]
    r"""The coefficient matrix :math:`C_{i,j}` to multiply :math:`y_j`."""


class ScalableDiscipline(BaseDiscipline):
    r"""A scalable discipline.

    It computes the output
    :math:`y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j`.
    """

    index: int
    """The index of the scalable discipline."""

    coefficients: Coefficients
    """The coefficient matrices defining the scalable discipline."""

    __x_i_name: str
    r"""The name of the local design variable :math:`x_i`."""

    __u_i_name: str
    r"""The name of the local uncertain variable :math:`u_i`."""

    __y_i_name: str
    r"""The name of the coupling variable :math:`y_i`."""

    __output_size: int
    r"""The size of the coupling variable :math:`y_i`."""

    def __init__(
        self,
        index: int,
        a_i: NDArray[float],
        D_i0: NDArray[float],  # noqa: N803
        D_ii: NDArray[float],  # noqa: N803
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
        self.name = f"{self.__class__.__name__}[{index}]"
        self.index = index
        self.input_names_to_default_values = default_input_values
        self.coefficients = Coefficients(a_i, D_i0, D_ii, C_ij)
        self.__x_i_name = get_x_local_name(index)
        self.__u_i_name = get_u_local_name(index)
        self.__y_i_name = get_coupling_name(index)
        self.__output_size = a_i.size
        self.input_names = sorted(self.input_names_to_default_values.keys())
        self.output_names = [self.__y_i_name]
        self.names_to_sizes = {
            input_name: default_value.size
            for input_name, default_value in default_input_values.items()
        }
        self.names_to_sizes.update({self.output_names[0]: len(D_ii)})

    def __call__(
        self,
        x_0: NDArray[float] | None = None,
        x_i: NDArray[float] | None = None,
        u_i: NDArray[float] | None = None,
        compute_jacobian: bool = False,
        **y_j: NDArray[float],
    ) -> dict[str, NDArray[float] | dict[str, NDArray[float]]]:
        r"""Compute the coupling variable :math:`y_i` or its derivatives.

        Args:
            x_0: The value of the shared design variable :math:`x_0`.
                If ``None``, use the default one.
            x_i: The value of the local design variable :math:`x_i`.
                If ``None``, use the default one.
            u_i: The constant vector :math:`u_i` added to the output.
                If ``None``, use the default one.
            compute_jacobian: Whether to compute
                the value of the coupling variable :math:`y_i`
                or that of its derivatives.
            **y_j: The values of the coupling variables
                :math:`(y_j)_{1\leq j\neq i\leq N}`.
                If missing, use the default ones.

        Returns:
            Either the value of math:`y_i` or that of its derivatives.
        """
        if x_0 is None:
            x_0 = self.input_names_to_default_values[SHARED_DESIGN_VARIABLE_NAME]

        if x_i is None:
            x_i = self.input_names_to_default_values[self.__x_i_name]

        if u_i is None:
            u_i = self.input_names_to_default_values.get(self.__u_i_name, 0.0)

        _y_j = {
            name: self.input_names_to_default_values[name]
            for name in self.coefficients.C_ij
        }
        _y_j.update(y_j)

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
            coupling_size = self.names_to_sizes[self.__y_i_name]
            jac = jacobian[self.__y_i_name]
            jac[SHARED_DESIGN_VARIABLE_NAME] = -self.coefficients.D_i0
            jac[self.__x_i_name] = -self.coefficients.D_ii
            jac[self.__u_i_name] = eye(coupling_size)
            for y_j_name, _C_ij in self.coefficients.C_ij.items():  # noqa: N806
                jac[y_j_name] = _C_ij

            return jacobian

        y_i = (
            self.coefficients.a_i.ravel()
            - self.coefficients.D_i0 @ x_0
            - self.coefficients.D_ii @ x_i
        )
        for y_j_name, _C_ij in self.coefficients.C_ij.items():  # noqa: N806
            y_i += _C_ij @ _y_j[y_j_name]

        return {self.output_names[0]: y_i + u_i}
