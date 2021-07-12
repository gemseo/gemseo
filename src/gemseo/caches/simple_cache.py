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
#                         documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Caching module to avoid multiple evaluations of a discipline
************************************************************
"""
from __future__ import division, unicode_literals

import logging
from copy import deepcopy

from gemseo.core.cache import AbstractCache, check_cache_approx, check_cache_equal
from gemseo.utils.data_conversion import DataConversion

LOGGER = logging.getLogger(__name__)


class SimpleCache(AbstractCache):
    """Simple discipline cache based on a dictionary.

    Only caches the last execution.
    """

    def __init__(self, tolerance=0.0, name=None):
        """Initialize cache tolerance. By default, don't use approximate cache. It is up
        to the user to choose to optimize CPU time with this or not.

        could be something like 2 * finfo(float).eps

        Parameters
        ----------
        tolerance : float
            Tolerance that defines if two input vectors
            are equal and cached data shall be returned.
            If 0, no approximation is made. Default: 0.
        name : str
            Name of the cache.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> cache = SimpleCache()
        """
        super(SimpleCache, self).__init__(tolerance, name)
        self.__input_cache = None
        self.__output_cache = None
        self.__jacobian_input_cache = None
        self.__jacobian_cache = None

    def clear(self):
        """Clear the cache.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([.2])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_length()
        1
        >>> cache.clear()
        >>> cache.get_length()
        0
        """
        super(SimpleCache, self).clear()
        self.__input_cache = None
        self.__output_cache = None
        self.__jacobian_input_cache = None
        self.__jacobian_cache = None

    def get_length(self):
        """Get the length of the cache, ie the number of stored elements.

        Returns
        -------
        length : int
            Length of the cache.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_length()
        1
        """
        if not self.__input_cache:
            return 0
        return 1

    @property
    def max_length(self):
        """Get the maximal length of the cache (the maximal number of stored elements).

        Returns
        -------
        length : int
            Maximal length of the cache.
        """
        return 1

    def get_last_cached_inputs(self):
        """Retrieve the last execution inputs.

        Returns
        -------
        inputs : dict
            Last cached inputs.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_last_cached_inputs()
        {'X': array([1.])}
        """
        return self.__input_cache

    def get_last_cached_outputs(self):
        """Retrieve the last execution outputs.

        Returns
        -------
        outputs : dict
            Last cached outputs.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_last_cached_outputs()
        {'y': array([2.])}
        """
        return self.__output_cache

    def get_data(self, index, **options):
        """Returns an elementary sample.

        :param index: sample index.
        :type index: int
        :param options: getter options
        """
        self._check_index(index)
        return {
            self.INPUTS_GROUP: self.__input_cache,
            self.OUTPUTS_GROUP: self.__output_cache,
            self.JACOBIAN_GROUP: self.__jacobian_cache,
        }

    def get_all_data(self, as_iterator=False):
        """Read all the data in the cache.

        Parameters
        ----------
        as_iterator : bool
            If True, return an iterator. Otherwise a dictionary.
            Default: False.


        Returns
        -------
        all_data : dict
            A dictionary of dictionaries for inputs, outputs and jacobian
            where keys are data indices.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_all_data()
        {1: {'inputs': {'x': array([1.])}, 'jacobian': None,
        'outputs': {'y': array([0.2])}}}
        """
        return {1: self.get_data(1)}

    @staticmethod
    def _create_input_cache(input_data, input_names, output_names=None):
        """Create a cache dict for input data.

        :param input_data: the input data to cache
        :param input_names: list of input data names
        :param output_names: list of output data names
        """
        if output_names is None:
            cache_dict = DataConversion.deepcopy_datadict(input_data, input_names)
        else:
            cache_dict = {k: v for k, v in input_data.items() if k in input_names}
            # If also an output, keeps a copy of the original input value
            for key in set(output_names) & set(input_names):
                val = input_data.get(key)
                if val is not None:
                    cache_dict[key] = deepcopy(val)
        return cache_dict

    def cache_outputs(self, input_data, input_names, output_data, output_names=None):
        """Cache data to avoid re evaluation.

        Parameters
        ----------
        input_data : dict
            Input data to cache.
        input_names : list(str)
            List of input data names.
        output_data : dict
            Output data to cache.
        output_names : list(str)
            List of output data names. If None, use all output names.
            Default: None.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache[1]
        {'y': array([2.]), 'x': array([1.])}
        """
        self.__input_cache = self._create_input_cache(
            input_data, input_names, output_names
        )
        self.__output_cache = DataConversion.deepcopy_datadict(
            output_data, output_names
        )

    def get_outputs(self, input_data, input_names=None):
        """Check if the discipline has already been evaluated for the given input data
        dictionary. If True, return the associated cache, otherwise return None.

        Parameters
        ----------
        input_data : dict
            Input data dictionary to test for caching.
        input_names : list(str)
            List of input data names.
            If None, uses them all

        Returns
        -------
        output_data : dict
            Output data if there is no need to evaluate the discipline.
            None otherwise.
        jacobian : dict
            Jacobian if there is no need to evaluate the discipline.
            None otherwise.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_outputs({'x': array([1.])}, ['x'])
        ({'y': array([2.])}, None)
        >>> cache.get_outputs({'x': array([2.])}, ['x'])
        (None, None)
        """
        cached_outs, cached_jac = None, None
        if input_names is None:
            input_names = input_data.keys()
        in_are_cached = self._is_cached(self.__input_cache, input_names, input_data)

        jac_in_are_cached = self._is_cached(
            self.__jacobian_input_cache, input_names, input_data
        )
        if in_are_cached:
            cached_outs = self.__output_cache
        if jac_in_are_cached:
            cached_jac = self.__jacobian_cache

        return cached_outs, cached_jac

    def _is_cached(self, in_cache, input_names, input_data):
        """Check if the input_data dictionary is cached.

        :param in_cache: cached input dictionary
        :param input_names: list of input names
        :param input_data: input dict of data
        :return: True if the data is cached
        """
        if not in_cache or len(in_cache) != len(input_names):
            return False

        if self.tolerance == 0.0:
            if check_cache_equal(input_data, in_cache):
                return True

        else:
            if check_cache_approx(input_data, in_cache, self.tolerance):
                return True
        return False

    def cache_jacobian(self, input_data, input_names, jacobian):
        """Cache jacobian data to avoid re evaluation.

        Parameters
        ----------
        input_data : dict
            Input data to cache.
        input_names : list(str)
            List of input data names.
        jacobian : dict
            Jacobian to cache.

        Examples
        --------
        >>> from gemseo.caches.simple_cache import SimpleCache
        >>> from numpy import array
        >>> cache = SimpleCache()
        >>> data = {'x': array([1.]), 'y': array([2.])}
        >>> jacobian = {'y': {'x': array([3.])}}
        >>> cache.cache_jacobian(data, ['x'], jacobian)
        (None, {'y': {'x': array([3.])}})
        """
        self.__jacobian_input_cache = self._create_input_cache(
            input_data, input_names, jacobian.keys()
        )
        self.__jacobian_cache = jacobian
