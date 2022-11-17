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
Scalable model
==============

This module implements the abstract concept of scalable model
which is used by scalable disciplines. A scalable model is built
from an input-output learning dataset associated with a function
and generalizing its behavior to a new user-defined problem dimension,
that is to say new user-defined input and output dimensions.

The concept of scalable model is implemented
through :class:`.ScalableModel`, an abstract class which is instantiated from:

- data provided as a :class:`.Dataset`
- variables sizes provided as a dictionary
  whose keys are the names of inputs and outputs
  and values are their new sizes.
  If a variable is missing, its original size is considered.

Scalable model parameters can also be filled in.
Otherwise, the model uses default values.

.. seealso::

   The :class:`.ScalableDiagonalModel` class overloads :class:`.ScalableModel`.
"""
from __future__ import annotations

from numpy import full
from numpy import where
from numpy import zeros

from gemseo.core.dataset import Dataset


class ScalableModel:
    """Scalable model."""

    ABBR = "sm"

    def __init__(self, data, sizes=None, **parameters):
        """Constructor.

        :param Dataset data: learning dataset.
        :param dict sizes: sizes of input and output variables.
            If None, use the original sizes.
            Default: None.
        :param parameters: model parameters
        """
        sizes = sizes or {}
        self.name = self.ABBR + "_" + data.name
        self.data = data
        self.sizes = self._set_sizes(sizes)
        self.parameters = parameters
        self.lower_bounds, self.upper_bounds = self.compute_bounds()
        self.normalize_data()
        self.lower_bounds, self.upper_bounds = self.compute_bounds()
        self.default_inputs = self._set_default_inputs()
        self.model = self.build_model()

    def _set_default_inputs(self):
        """Sets the default values of inputs from the model.

        :return: default inputs.
        :rtype: dict
        """
        return {name: full(self.sizes[name], 0.5) for name in self.inputs_names}

    def scalable_function(self, input_value=None):
        """Evaluate the scalable function.

        :param dict input_value: input values.
            If None, use default inputs. Default: None.
        :return: evaluation of the scalable function.
        :rtype: dict
        """
        raise NotImplementedError

    def scalable_derivatives(self, input_value=None):
        """Evaluate the scalable derivatives.

        :param dict input_value: input values.
            If None, use default inputs. Default: None
        :return: evaluation of the scalable derivatives.
        :rtype: dict
        """
        raise NotImplementedError

    def compute_bounds(self):
        """Compute lower and upper bounds of both input and output variables.

        :return: lower bounds, upper bounds.
        :rtype: dict, dict
        """
        inputs = self.data.get_data_by_group(self.data.INPUT_GROUP, True)
        outputs = self.data.get_data_by_group(self.data.OUTPUT_GROUP, True)
        lower_bounds = {name: value.min(0) for name, value in inputs.items()}
        lower_bounds.update({name: value.min(0) for name, value in outputs.items()})
        upper_bounds = {name: value.max(0) for name, value in inputs.items()}
        upper_bounds.update({name: value.max(0) for name, value in outputs.items()})

        return lower_bounds, upper_bounds

    def normalize_data(self):
        """Normalize dataset from lower and upper bounds."""
        normalized_data = Dataset()
        inputs = self.data.get_data_by_group(self.data.INPUT_GROUP, True)
        for name in self.data.get_names(self.data.INPUT_GROUP):
            data = inputs[name] - self.lower_bounds[name]
            data /= self.upper_bounds[name] - self.lower_bounds[name]
            normalized_data.add_variable(name, data, self.data.INPUT_GROUP)
        outputs = self.data.get_data_by_group(self.data.OUTPUT_GROUP, True)
        for name in self.data.get_names(self.data.OUTPUT_GROUP):
            indices = where(self.lower_bounds[name] == self.upper_bounds[name])[0]
            data = zeros(outputs[name].shape)
            if len(indices) != 0:
                data[:, indices] = zeros((data.shape[0], len(indices))) + 0.5
                self.lower_bounds[name][indices] = zeros(len(indices)) + 0.5
                self.upper_bounds[name][indices] = zeros(len(indices)) + 0.5
            indices = where(self.lower_bounds[name] != self.upper_bounds[name])[0]
            value = outputs[name][:, indices]
            lower_bound = self.lower_bounds[name][indices]
            upper_bound = self.upper_bounds[name][indices]
            data[:, indices] = (value - lower_bound) / (upper_bound - lower_bound)
            normalized_data.add_variable(name, data, self.data.OUTPUT_GROUP)
        self.data = normalized_data

    def build_model(self):
        """Build model with original sizes for input and output variables."""
        raise NotImplementedError

    @property
    def original_sizes(self):
        """Original sizes of variables.

        :return: original sizes of variables.
        :rtype: dict
        """
        return self.data.sizes

    @property
    def outputs_names(self):
        """Outputs names.

        :return: names of the outputs.
        :rtype: list(str)
        """
        return sorted(self.data.get_names(self.data.OUTPUT_GROUP))

    @property
    def inputs_names(self):
        """Inputs names.

        :return: names of the inputs.
        :rtype: list(str)
        """
        return sorted(self.data.get_names(self.data.INPUT_GROUP))

    def _set_sizes(self, sizes):
        """Set the new sizes of input and output variables.

        :param sizes: new sizes of some variables.
        :return: new sizes of all variables.
        :rtype: dict
        """
        for group in [self.data.INPUT_GROUP, self.data.OUTPUT_GROUP]:
            for name in self.data.get_names(group):
                original_size = self.original_sizes.get(name)
                sizes[name] = sizes.get(name, original_size)
        return sizes
