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

from numpy import array
from numpy import full

from gemseo.datasets.io_dataset import IODataset


class ScalableModel:
    """Scalable model."""

    ABBR = "sm"

    data: IODataset
    """The learning dataset."""

    def __init__(self, data: IODataset, sizes=None, **parameters) -> None:
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
        return {name: full(self.sizes[name], 0.5) for name in self.input_names}

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
        inputs = self.data.get_view(group_names=self.data.INPUT_GROUP).to_dict("list")
        outputs = self.data.get_view(group_names=self.data.OUTPUT_GROUP).to_dict("list")
        lower_bounds = {
            column[1]: array(value).min(0) for column, value in inputs.items()
        }
        lower_bounds.update(
            {column[1]: array(value).min(0) for column, value in outputs.items()}
        )
        upper_bounds = {
            column[1]: array(value).max(0) for column, value in inputs.items()
        }
        upper_bounds.update(
            {column[1]: array(value).max(0) for column, value in outputs.items()}
        )

        return lower_bounds, upper_bounds

    def normalize_data(self) -> None:
        """Normalize dataset from lower and upper bounds."""
        self.data = self.data.get_normalized()

    def build_model(self):
        """Build model with original sizes for input and output variables."""
        raise NotImplementedError

    @property
    def original_sizes(self):
        """Original sizes of variables.

        :return: original sizes of variables.
        :rtype: dict
        """
        return self.data.variable_names_to_n_components

    @property
    def output_names(self):
        """Outputs names.

        :return: names of the outputs.
        :rtype: list(str)
        """
        return self.data.get_variable_names(self.data.OUTPUT_GROUP)

    @property
    def input_names(self):
        """Inputs names.

        :return: names of the inputs.
        :rtype: list(str)
        """
        return self.data.get_variable_names(self.data.INPUT_GROUP)

    def _set_sizes(self, sizes):
        """Set the new sizes of input and output variables.

        :param sizes: new sizes of some variables.
        :return: new sizes of all variables.
        :rtype: dict
        """
        for group in [self.data.INPUT_GROUP, self.data.OUTPUT_GROUP]:
            for name in self.data.get_variable_names(group):
                original_size = self.original_sizes.get(name)
                sizes[name] = sizes.get(name, original_size)
        return sizes
