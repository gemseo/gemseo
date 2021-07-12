# -*- coding: utf-8 -*-
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
"""Concatenation of several input variables into a single one.

The :class:`.ConcatenationDiscipline` enables to concatenate inputs variables into a
single output variable.

Example:

    This example demonstrates the use of the :class:`.ConcatenationDiscipline`
    instances.
    The contraints variables :math:`c_1` and :math:`c_2` are concatenated into a
    single variable :math:`c`.

        >>> from gemseo.api import create_discipline
        >>> sellar_system_disc = create_discipline('SellarSystem')
        >>> input_vars = ['c1', 'c2']
        >>> output_var = ['c']
        >>> concatenation_disc = create_discipline('ConcatenationDiscipline',
        ...                                         input_vars,
        ...                                         output_var)
        >>> disciplines = [sellar_system_disc, concatenation_disc]
        >>> chain = create_discipline('MDOChain', disciplines=disciplines)
        >>> print(chain.execute())
        >>> print(chain.linearize(force_all=True))
"""
from typing import Optional, Sequence

from numpy import diag, ones, zeros

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.py23_compat import accumulate


class ConcatenationDiscipline(MDODiscipline):
    """Concatenate input variables into a single output variable."""

    def __init__(
        self,
        input_variables,  # type: Sequence[str]
        output_variable,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            input_variables: The input variables to concatenate.
            output_variable: The output variable name.
        """
        super(ConcatenationDiscipline, self).__init__()
        self.input_grammar.initialize_from_data_names(input_variables)
        self.output_grammar.initialize_from_data_names([output_variable])
        self.__output_variable = output_variable

    def _run(self):  # type: (...) -> None
        """Run the discipline."""
        self.local_data[self.__output_variable] = self.get_inputs_asarray()

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Sequence[str]]
        outputs=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
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
