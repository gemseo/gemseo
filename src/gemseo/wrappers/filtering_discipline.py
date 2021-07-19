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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from gemseo.core.discipline import MDODiscipline


class FilteringDiscipline(MDODiscipline):
    """The FilteringDiscipline is a MDODiscipline wrapping another MDODiscipline, for a
    subset of inputs and outputs."""

    def __init__(
        self,
        discipline,
        inputs_names=None,
        outputs_names=None,
        keep_in=True,
        keep_out=True,
    ):
        """Constructor.

        :param MDODiscipline discipline: discipline.
        :param list(str) inputs_names: list of inputs names. If None, use all inputs.
            Default: None.
        :param list(str) outputs_names: list of outputs names. If None, use all outputs.
            Default: None.
        :param bool keep_in: if True, keep the list of inputs names.
            Otherwise, remove them.
        :param bool keep_out: if True, keep the list of outputs names.
            Otherwise, remove them.
        """
        self.discipline = discipline
        super(FilteringDiscipline, self).__init__(name=discipline.name)
        original_inputs_names = discipline.get_input_data_names()
        original_outputs_names = discipline.get_output_data_names()
        if inputs_names is not None:
            if not keep_in:
                inputs_names = list(set(original_inputs_names) - set(inputs_names))
        else:
            inputs_names = original_inputs_names
        if outputs_names is not None:
            if not keep_out:
                outputs_names = list(set(original_outputs_names) - set(outputs_names))
        else:
            outputs_names = original_outputs_names
        self.input_grammar.initialize_from_data_names(inputs_names)
        self.output_grammar.initialize_from_data_names(outputs_names)
        self.default_inputs = self.__filter_inputs(self.discipline.default_inputs)
        removed_inputs = set(original_inputs_names) - set(inputs_names)
        diff_inputs = set(self.discipline._differentiated_inputs) - removed_inputs
        self.add_differentiated_inputs(list(diff_inputs))
        removed_outputs = set(original_outputs_names) - set(outputs_names)
        diff_outputs = set(self.discipline._differentiated_outputs) - removed_outputs
        self.add_differentiated_outputs(list(diff_outputs))

    def _run(self):
        self.discipline.execute(self.get_input_data())
        self.store_local_data(**self.__filter_inputs(self.discipline.local_data))
        self.store_local_data(**self.__filter_outputs(self.discipline.local_data))

    def _compute_jacobian(self, inputs=None, outputs=None):
        self.discipline._compute_jacobian(inputs, outputs)
        self._init_jacobian(inputs, outputs, with_zeros=True)
        jac = self.discipline.jac
        for output_name in self.get_output_data_names():
            for input_name in self.get_input_data_names():
                self.jac[output_name][input_name] = jac[output_name][input_name]

    @staticmethod
    def __filter(data, keys):
        """Filter a data dictionary by names.

        :param dict data: data dictionary.
        :param list(str) keys: list of dictionary keys.
        """
        return {key: data[key] for key in keys}

    def __filter_inputs(self, data):
        """Filter a data dictionary by inputs names.

        :param dict data: data dictionary.
        """
        return self.__filter(data, self.get_input_data_names())

    def __filter_outputs(self, data):
        """Filter a data dictionary by outputs names.

        :param dict data: data dictionary.
        """
        return self.__filter(data, self.get_output_data_names())
