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
#    INITIAL AUTHORS - initial API and implementation and/or
#                  initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalable discipline
===================

The :mod:`~gemseo.problems.scalable.discipline`
implements the concept of scalable discipline.
This is a particular discipline
built from a input-output learning dataset associated with a function
and generalizing its behavior to a new user-defined problem dimension,
that is to say new used-defined input and output dimensions.

Alone or in interaction with other objects of the same type,
a scalable discipline can be used to compare the efficiency of an algorithm
applying to disciplines with respect to the problem dimension,
e.g. optimization algorithm, surrogate model, MDO formulation, MDA, ...

The :class:`.ScalableDiscipline` class implements this concept.
It inherits from the :class:`.MDODiscipline` class
in such a way that it can easily be used in a :class:`.Scenario`.
It is composed of a :class:`.ScalableModel`.


The user only needs to provide:

- the name of a class overloading :class:`.ScalableModel`,
- a dataset as an :class:`.AbstractFullCache`
- variables sizes as a dictionary
  whose keys are the names of inputs and outputs
  and values are their new sizes.
  If a variable is missing, its original size is considered.

The :class:`.ScalableModel` parameters can also be filled in,
otherwise the model uses default values.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.factory import ScalableModelFactory
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()

from gemseo import LOGGER


class ScalableDiscipline(MDODiscipline):
    """ Scalable discipline """

    def __init__(self, name, data, sizes=None, **parameters):
        """Constructor.

        :param str name: scalable model class name.
        :param AbstractFullCache data: learning dataset.
        :param dict sizes: sizes of input and output variables.
            If None, use the original sizes.
            Default: None.
        :param parameters: model parameters
        """
        create = ScalableModelFactory().create
        self.scalable_model = create(name, data=data, sizes=sizes, **parameters)
        super(ScalableDiscipline, self).__init__(self.scalable_model.name)
        self.initialize_grammars(data)
        self._default_inputs = self.scalable_model.default_inputs
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def initialize_grammars(self, data):
        """Initialize input and output grammars from data names.

        :param AbstractFullCache data: learning dataset.
        """
        self.input_grammar.initialize_from_data_names(data.inputs_names)
        self.output_grammar.initialize_from_data_names(data.outputs_names)

    def _run(self):
        """ Runs the scalable discipline and stores the output values. """
        output_value = self.scalable_model.scalable_function(self.local_data)
        self.local_data.update(output_value)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Compute the Jacobian of outputs wrt inputs and store the values.

        :param inputs: list of input variables. Default value: None.
        :type inputs: list(str)
        :param outputs: list of output functions.  Default value: None.
        :type outputs: list(str)
        """
        self._init_jacobian(inputs, outputs, with_zeros=True)
        jac = self.scalable_model.scalable_derivatives(self.local_data)
        inputs_names = self.scalable_model.inputs_names
        jac = {
            fname: DataConversion.array_to_dict(
                jac[fname], inputs_names, self.scalable_model.sizes
            )
            for fname in self.get_output_data_names()
        }
        self.jac = jac
