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

from gemseo.core.cache import AbstractFullCache
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.locks import synchronized
from gemseo.utils.multi_processing import RLock

LOGGER = logging.getLogger(__name__)


class MemoryFullCache(AbstractFullCache):
    """Cache using memory to cache all data."""

    def __init__(self, tolerance=0.0, name=None, is_memory_shared=True):
        """Initialize a dictionary to cache data.

        Initialize cache tolerance.
        By default, don't use approximate cache.
        It is up to the user to choose to optimize CPU time with this or not
        could be something like 2 * finfo(float).eps

        Parameters
        ----------
        tolerance : float
            Tolerance that defines if two input vectors
            are equal and cached data shall be returned.
            If 0, no approximation is made. Default: 0.
        name : str
            Name of the cache.
        is_memory_shared : bool
            If True, a shared memory dict is used to store the data,
            which makes the cache compatible with multiprocessing.
            WARNING: if set to False, and multiple disciplines point to
            the same cache or the process is multiprocessed, there may
            be duplicate computations because the cache will not be
            shared among the processes.

        Examples
        --------
        >>> from gemseo.caches.memory_full_cache import MemoryFullCache
        >>> cache = MemoryFullCache()
        """
        self.__is_memory_shared = is_memory_shared
        super(MemoryFullCache, self).__init__(tolerance, name)
        self.__init_data()

    def __init_data(self):
        """Initializes the local dict that stores the data.

        Either a shared memory dict or a basic dict.
        """
        if self.__is_memory_shared:
            self._data = self._manager.dict()
        else:
            self._data = {}

    def _duplicate_from_scratch(self):
        return MemoryFullCache(self.tolerance, self.name, self.__is_memory_shared)

    def _initialize_entry(self, sample_id):
        """Initialize an entry of the dataset if needed.

        :param int sample_id: sample ID.
        """
        template = {}
        self._data[sample_id] = template

    def _set_lock(self):
        """Sets a lock for multithreading, either from an external object or internally
        by using RLock()."""
        return RLock()

    def _has_group(self, sample_id, var_group):
        """Checks if the dataset has the particular variables group filled in.

        :param int sample_id: sample ID.
        :param str var_group: name of the variables group.
        :return: True if the variables group is filled in.
        :rtype: bool
        """
        return var_group in self._data.get(sample_id)

    @synchronized
    def clear(self):
        """Clear the cache.

        Examples
        --------
        >>> from gemseo.caches.memory_full_cache import MemoryFullCache
        >>> from numpy import array
        >>> cache = MemoryFullCache()
        >>> for index in range(5):
        >>>     data = {'x': array([1.])*index, 'y': array([.2])*index}
        >>>     cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_length()
        5
        >>> cache.clear()
        >>> cache.get_length()
        0
        """
        super(MemoryFullCache, self).clear()
        self.__init_data()

    def _read_data(self, group_number, group_name):
        """Read a data dict in the hdf.

        :param group_name: name of the group where data is written
        :param group_number: number of the group
        :returns: data dict and jacobian
        """
        result = self._data[group_number].get(group_name)
        if group_name == self.JACOBIAN_GROUP and result is not None:
            result = DataConversion.dict_to_jac_dict(result)
        return result

    def _write_data(self, values, names, var_group, sample_id):
        """Writes data associated with a variables group and a sample ID into the
        dataset.

        :param dict values: data dictionary where keys are variables names
            and values are variables values (numpy arrays).
        :param list(str) names: list of input data names to write.
        :param str var_group: name of the variables group,
            either AbstractCache.INPUTS_GROUP, AbstractCache.OUTPUTS_GROUP or
            AbstractCache.JACOBIAN_GROUP.
        :param int sample_id: sample ID.
        """
        data = self._data[sample_id]
        data[var_group] = {name: values[name] for name in names}
        self._data[sample_id] = data

    @property
    def copy(self):
        """Copy cache."""
        cache = self._duplicate_from_scratch()
        cache.merge(self)
        return cache
