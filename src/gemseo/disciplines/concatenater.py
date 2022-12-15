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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The concatenation of several input variables into a single one."""
from __future__ import annotations

from typing import Sequence

from numpy import concatenate
from numpy import eye
from numpy import zeros

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.python_compatibility import accumulate


class Concatenater(MDODiscipline):
    """Concatenate input variables into a single output variable.

    These input variables can be scaled before concatenation.

    Example:
        >>> from gemseo.api import create_discipline
        >>> sellar_system_disc = create_discipline('SellarSystem')
        >>> constraints_names = ['c1', 'c2']
        >>> output_name = ['c']
        >>> concatenation_disc = create_discipline(
        ...     'Concatenater', constraints_names, output_name
        ... )
        >>> disciplines = [sellar_system_disc, concatenation_disc]
        >>> chain = create_discipline('MDOChain', disciplines=disciplines)
        >>> print(chain.execute())
        >>> print(chain.linearize(force_all=True))
    """

    def __init__(
        self,
        input_variables: Sequence[str],
        output_variable: str,
        input_coefficients: dict[str, float] = None,
    ) -> None:
        """
        Args:
            input_variables: The input variables to concatenate.
            output_variable: The output variable name.
            input_coefficients: The coefficients
                related to the different input variables.
        """  # noqa: D205 D212 D415
        super().__init__()
        self.input_grammar.update(input_variables)
        self.output_grammar.update([output_variable])
        self.__output_variable = output_variable
        self.__coefficients = dict.fromkeys(input_variables, 1.0)
        if input_coefficients:
            self.__coefficients.update(input_coefficients)

    def _run(self) -> None:
        """Run the discipline."""
        input_data = self.get_input_data()
        self.local_data[self.__output_variable] = concatenate(
            [
                self.__coefficients[input_name] * input_data[input_name]
                for input_name in self.get_input_data_names()
            ]
        )

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        """Compute the jacobian matrix.

        Args:
            inputs: The linearization should be performed with respect
                to inputs list. If None, linearization should
                be performed wrt all inputs (Default value = None)
            outputs: The linearization should be performed on outputs list.
                If None, linearization should be performed
                on all outputs (Default value = None)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)

        names = self.get_input_data_names()
        sizes = [input.size for input in self.get_all_inputs()]
        total_size = self.get_inputs_asarray().size

        # Instead of manually accumulating, we use the accumulate() iterator.
        jac = self.jac[self.__output_variable]
        for name, size, start in zip(names, sizes, accumulate(sizes, initial=0)):
            val = zeros([total_size, size])
            val[start : (start + size), :] = self.__coefficients[name] * eye(size)
            jac[name] = val
