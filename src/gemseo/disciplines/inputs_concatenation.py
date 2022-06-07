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

from numpy import diag
from numpy import ones
from numpy import zeros

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.python_compatibility import accumulate


class ConcatenationDiscipline(MDODiscipline):
    """Concatenate input variables into a single output variable.

    Example:
        >>> from gemseo.api import create_discipline
        >>> sellar_system_disc = create_discipline('SellarSystem')
        >>> constraints_names = ['c1', 'c2']
        >>> output_name = ['c']
        >>> concatenation_disc = create_discipline(
        ...     'ConcatenationDiscipline', constraints_names, output_name
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
    ) -> None:
        """# noqa: D205 D212 D415
        Args:
            input_variables: The input variables to concatenate.
            output_variable: The output variable name.
        """
        super().__init__()
        self.input_grammar.update(input_variables)
        self.output_grammar.update([output_variable])
        self.__output_variable = output_variable

    def _run(self) -> None:
        """Run the discipline."""
        self.local_data[self.__output_variable] = self.get_inputs_asarray()

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

        inputs_names = self.get_input_data_names()
        inputs_sizes = [inp.size for inp in self.get_all_inputs()]
        inputs_total_size = self.get_inputs_asarray().size

        # Instead of manually accumulating, we use the accumulate() iterator.
        for name, size, start in zip(
            inputs_names, inputs_sizes, accumulate(inputs_sizes, initial=0)
        ):
            val = zeros([inputs_total_size, size])
            val[start : (start + size), :] = diag(ones(size))
            self.jac[self.__output_variable][name] = val
