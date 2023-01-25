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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)
#        :author: Gilberto Ruiz Jimenez
"""The MDOFunction consistency constraint subclass to support formulations."""
from __future__ import annotations

import logging
from typing import Sequence
from typing import TYPE_CHECKING

from numpy import eye
from numpy import newaxis
from numpy import ones_like
from numpy import zeros

from gemseo.core.mdofunctions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.core.formulation import MDOFormulation

LOGGER = logging.getLogger(__name__)


class ConsistencyCstr(MDOFunction):
    """An :class:`.MDOFunction` object to compute the consistency constraints."""

    def __init__(
        self,
        output_couplings: Sequence[str],
        formulation: MDOFormulation,
    ) -> None:
        """
        Args:
            output_couplings: The names of the output couplings.
            formulation: The IDF formulation of the problem.
        """  # noqa: D205, D212, D415
        self.__formulation = formulation
        self.__output_couplings = output_couplings
        self.__coupl_func = FunctionFromDiscipline(
            self.__output_couplings, self.__formulation
        )
        self.__dv_names_of_disc = self.__coupl_func.args

        if self.__formulation.normalize_constraints:
            self.__norm_fact = self.__formulation._get_normalization_factor(
                output_couplings
            )
        else:
            self.__norm_fact = 1.0

        self.__dv_len = self.__formulation.design_space.variables_sizes

        expr = ""
        for out_c in self.__output_couplings:
            expr += f"{out_c}({', '.join(self.__dv_names_of_disc)}) - {out_c}\n"

        super().__init__(
            self._func_to_wrap,
            self.__coupl_func.name,
            args=self.__dv_names_of_disc,
            expr=expr,
            jac=self._jac_to_wrap,
            outvars=self.__coupl_func.outvars,
            f_type=MDOFunction.TYPE_EQ,
        )

    def _func_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the consistency constraints.

        Args:
            x_vect: The design variable vector.

        Returns:
            The value of the consistency constraints.
            Equal to zero if the disciplines are at equilibrium.
        """
        x_sw = self.__formulation.mask_x_swap_order(self.__output_couplings, x_vect)
        coupl = self.__coupl_func(x_vect)
        if self.__formulation.normalize_constraints:
            return (coupl - x_sw) / self.__norm_fact
        return coupl - x_sw

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the gradient of the consistency constraints.

        Args:
            x_vect: The design variable vector.

        Returns:
            The value of the gradient of the consistency constraints.
        """
        coupl_jac = self.__coupl_func.jac(x_vect)  # pylint: disable=E1102

        if len(coupl_jac.shape) > 1:
            # In this case it is harder since a block diagonal
            # matrix with -Id should be placed for each output
            # coupling, at the right place.
            n_outs = coupl_jac.shape[0]
            x_jac_2d = zeros((n_outs, len(x_vect)), dtype=x_vect.dtype)
            x_names = self.__formulation.get_optim_variables_names()
            o_min = 0
            o_max = 0
            for out in self.__output_couplings:
                o_len = self.__dv_len[out]
                i_min = 0
                i_max = 0
                o_max += o_len
                for x_i in x_names:
                    x_len = self.__dv_len[x_i]
                    i_max += x_len
                    if x_i == out:
                        x_jac_2d[o_min:o_max, i_min:i_max] = eye(x_len)
                    i_min = i_max
                o_min = o_max
            x_jac = x_jac_2d
        else:
            # This is surprising but there is a duality between the masking
            # operation in the function inputs and the unmasking of its
            # outputs
            x_jac = self.__formulation.unmask_x_swap_order(
                self.__output_couplings, ones_like(x_vect)
            )

        if self.__formulation.normalize_constraints:
            return (coupl_jac - x_jac) / self.__norm_fact[:, newaxis]
        return coupl_jac - x_jac
