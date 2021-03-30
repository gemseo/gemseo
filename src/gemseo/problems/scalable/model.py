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
Scalable model
==============

This module implements the abstract concept of scalable model
which is used by scalable disciplines. A scalable model is built
from a input-output learning dataset associated with a function
and generalizing its behavior to a new user-defined problem dimension,
that is to say new used-defined input and output dimensions.

The concept of scalable model is implemented
through :class:`.ScalableModel`, an abstract class which is instantiated from:

- data provided as an :class:`.AbstractFullCache`
- variables sizes provided as a dictionary
  whose keys are the names of inputs and outputs
  and values are their new sizes.
  If a variable is missing, its original size is considered.

Scalable model parameters can also be filled in.
Otherwise the model uses default values.

.. seealso::

   The :class:`.ScalableDiagonalModel` class overloads :class:`.ScalableModel`.
"""
from __future__ import absolute_import, division, unicode_literals

from copy import deepcopy

from future import standard_library
from numpy import maximum as np_max
from numpy import minimum as np_min
from numpy import ones
from past.utils import old_div

from gemseo.caches.memory_full_cache import MemoryFullCache

standard_library.install_aliases()

from gemseo import LOGGER


class ScalableModel(object):
    """ Scalable model. """

    ABBR = "sm"

    def __init__(self, data, sizes=None, **parameters):
        """Constructor.

        :param AbstractFullCache data: learning dataset.
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
        default_inputs = {}
        for name in self.inputs_names:
            default_inputs[name] = 0.5 * ones(self.sizes[name])
        return default_inputs

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
        all_data = self.data.get_all_data(True)
        data = next(all_data)
        var_lb = deepcopy(data[self.data.INPUTS_GROUP])
        var_ub = deepcopy(data[self.data.INPUTS_GROUP])
        var_lb.update(data[self.data.OUTPUTS_GROUP])
        var_ub.update(data[self.data.OUTPUTS_GROUP])
        for data in all_data:
            input_data = data[self.data.INPUTS_GROUP]
            for varname, value in input_data.items():
                var_lb[varname] = np_min(var_lb[varname], value)
                var_ub[varname] = np_max(var_ub[varname], value)
            output_data = data[self.data.OUTPUTS_GROUP]
            for varname, value in output_data.items():
                var_lb[varname] = np_min(var_lb[varname], value)
                var_ub[varname] = np_max(var_ub[varname], value)
        all_data = self.data.get_all_data(True)
        return var_lb, var_ub

    def normalize_data(self):
        """ Normalize cache from lower and upper bounds. """

        def normalize_vector(value, name):
            """ Normalize vector. """
            lower = self.lower_bounds[name]
            upper = self.upper_bounds[name]
            return old_div((value - lower), (upper - lower))

        normalized_cache = MemoryFullCache()
        all_data = self.data.get_all_data(True)
        for data in all_data:
            input_data = data[self.data.INPUTS_GROUP]
            input_data = {
                name: normalize_vector(value, name)
                for name, value in input_data.items()
            }
            output_data = data[self.data.OUTPUTS_GROUP]
            output_data = {
                name: normalize_vector(value, name)
                for name, value in output_data.items()
            }
            normalized_cache.cache_outputs(
                input_data, self.inputs_names, output_data, self.outputs_names
            )
        self.data = normalized_cache

    def build_model(self):
        """ Build model with original sizes for input and output variables."""
        raise NotImplementedError

    @property
    def original_sizes(self):
        """Original sizes of variables.

        :return: original sizes of variables.
        :rtype: dict
        """
        return self.data.varsizes

    @property
    def outputs_names(self):
        """Outputs names.

        :return: names of the outputs.
        :rtype: list(str)
        """
        return sorted(self.data.outputs_names)

    @property
    def inputs_names(self):
        """Inputs names.

        :return: names of the inputs.
        :rtype: list(str)
        """
        return sorted(self.data.inputs_names)

    def _set_sizes(self, sizes):
        """Set the new sizes of input and output variables.

        :param dict: new sizes of some variables.
        :return: new sizes of all variables.
        :rtype: dict
        """
        for name in self.data.inputs_names + self.data.outputs_names:
            original_size = self.original_sizes.get(name)
            sizes[name] = sizes.get(name, original_size)
        return sizes
