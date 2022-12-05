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
"""The MDOFunction subclass to create a function from an MDODiscipline."""
from __future__ import annotations

import logging
from typing import Iterable
from typing import Sequence
from typing import TYPE_CHECKING

from numpy import empty

from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.mdofunctions.mdo_function import ArrayType
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from gemseo.core.discipline import MDODiscipline
    from gemseo.core.base_formulation import BaseFormulation

LOGGER = logging.getLogger(__name__)


class FunctionFromDiscipline(MDOFunction):
    """An :class:`.MDOFunction` object from an :class:`.MDODiscipline`."""

    def __init__(
        self,
        output_names: Sequence[str],
        mdo_formulation: BaseFormulation,
        discipline: MDODiscipline | None = None,
        top_level_disc: bool = True,
        x_names: Sequence[str] | None = None,
        all_data_names: Iterable[str] | None = None,
        differentiable: bool = True,
    ) -> None:
        """
        Args:
            output_names: The names of the outputs.
            mdo_formulation: The MDOFormulation object in which the function is
                located.
            discipline: The discipline computing these outputs.
                If None, the discipline is detected from the inner disciplines.
            top_level_disc: If True, search the discipline among the top level ones.
            x_names: The names of the design variables.
                If None, use self.get_x_names_of_disc(discipline).
            all_data_names: The reference data names for masking x.
                If None, use self.get_optim_variables_names().
            differentiable: If True, then inputs and outputs are added
                to the list of variables to be differentiated.
        """  # noqa: D205, D212, D415
        self.__output_names = output_names
        self.__mdo_formulation = mdo_formulation
        self.__discipline = discipline
        self.__top_level_disc = top_level_disc
        self.__x_names = x_names
        self.__all_data_names = all_data_names
        self.__differentiable = differentiable
        self.__x_mask = None

        if self.__discipline is None:
            self.__gen = self.__mdo_formulation._get_generator_from(
                self.__output_names, top_level_disc=self.__top_level_disc
            )
            self.__discipline = self.__gen.discipline
        else:
            self.__gen = MDOFunctionGenerator(self.__discipline)

        if self.__x_names is None:
            self.__x_names = self.__mdo_formulation.get_x_names_of_disc(
                self.__discipline
            )

        self.__out_x_func = self.__gen.get_function(
            self.__x_names, self.__output_names, differentiable=self.__differentiable
        )

        super().__init__(
            self._func_to_wrap,
            jac=self._jac_to_wrap,
            name=self.__out_x_func.name,
            f_type=MDOFunction.TYPE_OBJ,
            args=self.__x_names,
            expr=self.__out_x_func.expr,
            dim=self.__out_x_func.dim,
            outvars=self.__out_x_func.outvars,
        )

    def _func_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the outputs.

        Args:
            x_vect: The design variable vector.

        Returns:
            The value of the outputs.
        """
        if self.__x_mask is None:
            self.__x_mask = self.__mdo_formulation.get_x_mask_x_swap_order(
                self.__x_names, self.__all_data_names
            )
        return self.__out_x_func(x_vect[self.__x_mask])

    def _jac_to_wrap(self, x_vect: ArrayType) -> ArrayType:
        """Compute the gradient of the outputs.

        Args:
            x_vect: The design variable vector.

        Returns:
            The value of the gradient of the outputs.
        """
        if self.__x_mask is None:
            self.__x_mask = self.__mdo_formulation.get_x_mask_x_swap_order(
                self.__x_names, self.__all_data_names
            )
        x_of_disc = x_vect[self.__x_mask]

        loc_jac = self.__out_x_func.jac(x_of_disc)  # pylint: disable=E1102

        if len(loc_jac.shape) == 1:
            # This is surprising but there is a duality between the
            # masking operation in the function inputs and the
            # unmasking of its outputs
            jac = self.__mdo_formulation.unmask_x_swap_order(
                self.__x_names, loc_jac, self.__all_data_names
            )
        else:
            n_outs = loc_jac.shape[0]
            jac = empty((n_outs, x_vect.size), dtype=x_vect.dtype)
            for func_ind in range(n_outs):
                gr_u = self.__mdo_formulation.unmask_x_swap_order(
                    self.__x_names, loc_jac[func_ind, :], self.__all_data_names
                )
                jac[func_ind, :] = gr_u
        return jac
