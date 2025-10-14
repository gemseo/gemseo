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

from typing import TYPE_CHECKING
from typing import Final

from numpy import eye
from numpy import newaxis
from numpy import ones
from numpy import zeros

from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.formulations.idf import IDF
    from gemseo.typing import NumberArray


class ConsistencyConstraint(MDOFunction):
    """An :class:`.MDOFunction` object to compute the consistency constraints."""

    __CONSISTENCY_CONSTRAINT_NAME: Final[str] = "consistency_{}"
    """The name template for consistency constraints."""

    def __init__(
        self,
        output_couplings: Sequence[str],
        formulation: IDF,
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

        self.__dv_names_of_disc = self.__coupl_func.input_names

        if self.__formulation.normalize_constraints:
            self.__norm_fact = self.__formulation._get_normalization_factor(
                output_couplings
            )
        else:
            self.__norm_fact = 1.0

        self.__input_names_to_sizes = self.__formulation.design_space.variable_sizes

        expr = ""
        for out_c in self.__output_couplings:
            expr += f"{out_c}({', '.join(self.__dv_names_of_disc)}) - {out_c}\n"

        super().__init__(
            self._func_to_wrap,
            self.__CONSISTENCY_CONSTRAINT_NAME.format(self.__coupl_func.name),
            input_names=self.__dv_names_of_disc,
            expr=expr,
            jac=self._jac_to_wrap,
            output_names=self.__coupl_func.output_names,
            f_type=MDOFunction.ConstraintType.EQ,
        )

    @property
    def coupling_function(self) -> FunctionFromDiscipline:
        """The coupling function."""
        return self.__coupl_func

    def _func_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the consistency constraints y(x, yt) - yt.

        Args:
            x_vect: The optimization vector
                including both design variables x and the target coupling variables yt.

        Returns:
            The value of the consistency constraints.
            Equal to zero if the disciplines are at equilibrium.
        """
        x_sw = self.__formulation.mask_x_swap_order(self.__output_couplings, x_vect)
        coupl = self.__coupl_func.evaluate(x_vect)
        if self.__formulation.normalize_constraints:
            return (coupl - x_sw) / self.__norm_fact
        return coupl - x_sw

    def _jac_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the gradient of the consistency constraints y(x, yt) - yt.

        Args:
            x_vect: The optimization vector
                including both design variables x and the target coupling variables yt.

        Returns:
            The value of the gradient of the consistency constraints.
        """
        y_jac = self.__coupl_func.jac(x_vect)

        if len(y_jac.shape) > 1:
            # In this case it is harder since a block diagonal
            # matrix with -Id should be placed for each output
            # coupling, at the right place.
            output_size = y_jac.shape[0]
            yt_jac = zeros((output_size, len(x_vect)), dtype=x_vect.dtype)
            input_names = (
                self.__formulation.optimization_problem.design_space.variable_names
            )
            o_min = 0
            o_max = 0
            for output_name in self.__output_couplings:
                i_min = 0
                i_max = 0
                o_max += self.__input_names_to_sizes[output_name]
                for input_name in input_names:
                    i_max += (input_size := self.__input_names_to_sizes[input_name])
                    if input_name == output_name:
                        yt_jac[o_min:o_max, i_min:i_max] = eye(input_size)
                    i_min = i_max

                o_min = o_max
        else:
            # We create a (..., input_dimension) matrix of ones
            # and replace the ones by zeros for all input variables but yt.
            shape = (
                *x_vect.shape[:-1],
                sum(
                    self.__input_names_to_sizes[name]
                    for name in self.__output_couplings
                ),
            )
            yt_jac = self.__formulation.unmask_x_swap_order(
                self.__output_couplings, ones(shape)
            )

        if self.__formulation.normalize_constraints:
            return (y_jac - yt_jac) / self.__norm_fact[:, newaxis]

        return y_jac - yt_jac
