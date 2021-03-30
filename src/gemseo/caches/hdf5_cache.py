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
from __future__ import absolute_import, division, unicode_literals

from logging import getLogger
from os.path import exists

import h5py
from future.utils import with_metaclass
from numpy import append, array, bytes_, unicode_

from gemseo.core.cache import AbstractCache, AbstractFullCache, hash_data_dict, to_real
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.locks import synchronized
from gemseo.utils.multi_processing import RLock
from gemseo.utils.py23_compat import is_py2
from gemseo.utils.py23_compat import long as _long
from gemseo.utils.py23_compat import string_array, string_dtype
from gemseo.utils.singleton import SingleInstancePerFileAttribute

PY2 = is_py2()

from gemseo import LOGGER


class HDF5Cache(AbstractFullCache):
    """
    Cache using disk HDF5 file to store the data
    """

    def __init__(self, hdf_file_path, hdf_node_path, tolerance=0.0, name=None):
        """Initialize a singleton to access a HDF file.
        This singleton is used for multithreaded/multiprocessing access
        with a Lock.

        Initialize cache tolerance.
        By default, don't use approximate cache.
        It is up to the user to choose to optimize CPU time with this or not
        could be something like 2 * finfo(float).eps

        Parameters
        ----------
        hdf_file_path : str
            Path of the HDF file.
        hdf_node_path : str
            Node of the HDF file.
        tolerance : float
            Tolerance that defines if two input vectors
            are equal and cached data shall be returned.
            If 0, no approximation is made. Default: 0.
        name : str
            Name of the cache.

        Examples
        --------
        >>> from gemseo.caches.hdf5_cache import HDF5Cache
        >>> cache = HDF5Cache('my_cache.hdf5', 'my_node')
        """
        if not name:
            name = hdf_node_path
        self.__file_path = hdf_file_path
        self._hdf_file = HDF5FileSingleton(hdf_file_path)
        self.__hdf_node_path = hdf_node_path
        super(HDF5Cache, self).__init__(tolerance, name)
        self._read_hashes()

    def __str__(self):
        msg = super(HDF5Cache, self).__str__()
        msg += "\n" + "HDF file path " + str(self.__file_path) + "\n"
        msg += "HDF node path " + str(self.__hdf_node_path)
        return msg

    def _duplicate_from_scratch(self):
        return HDF5Cache(
            "inc_" + self.__file_path, self.__hdf_node_path, self.tolerance, self.name
        )

    def _set_lock(self):
        """Set a lock for multithreading,
        either from an external object or internally by using RLock()."""
        return self._hdf_file.lock

    @synchronized
    def _read_hashes(self):
        """
        Read the hashes dict in the HDF file
        """
        max_grp = self._hdf_file.read_hashes(self._hashes, self.__hdf_node_path)
        self._last_accessed_group.value = max_grp
        self._max_group.value = max_grp
        n_hash = len(self._hashes)
        if n_hash > 0:
            msg = "Found %s entries in the cache file : %s node : %s"
            LOGGER.info(msg, n_hash, self.__file_path, self.__hdf_node_path)

    def _has_group(self, sample_id, var_group):
        """Check if the dataset has the particular variables group filled in.

        :param int sample_id: sample ID.
        :param str var_group: name of the variables group.
        :return: True if the variables group is filled in.
        :rtype: bool
        """
        return self._hdf_file.has_group(sample_id, var_group, self.__hdf_node_path)

    @synchronized
    def clear(self):
        """
        Clear the cache

        Examples
        --------
        >>> from gemseo.caches.hdf5_cache import HDF5Cache
        >>> from numpy import array
        >>> cache = HDF5Cache('my_cache.hdf5', 'my_node')
        >>> for index in range(5):
        >>>     data = {'x': array([1.])*index, 'y': array([.2])*index}
        >>>     cache.cache_outputs(data, ['x'], data, ['y'])
        >>> cache.get_length()
        5
        >>> cache.clear()
        >>> cache.get_length()
        0
        """
        super(HDF5Cache, self).clear()
        self._hdf_file.clear(self.__hdf_node_path)

    def _read_data(self, group_number, group_name, h5_open_file=None):
        """Read data from an HDF file.

        :param str group_name: name of the group where data is written.
        :param int group_number: number of the group.
        :param h5_open_file: eventually the already opened file.
            this improves performance but is incompatible with
            multiprocess/treading
        :returns: data dict and jacobian
        """
        result = self._hdf_file.read_data(
            group_number, group_name, self.__hdf_node_path, h5_open_file=h5_open_file
        )[0]
        if group_name == self.JACOBIAN_GROUP and result is not None:
            result = DataConversion.dict_to_jac_dict(result)
        return result

    def _write_data(self, values, names, var_group, sample_id):
        """Write data associated with a variables group
        and a sample ID into the dataset.

        :param dict values: data dictionary where keys are variables names
            and values are variables values (numpy arrays).
        :param list(str) names: list of input data names to write.
        :param str var_group: name of the variables group,
            either AbstractCache.INPUTS_GROUP, AbstractCache.OUTPUTS_GROUP or
            AbstractCache.JACOBIAN_GROUP.
        :param int sample_id: sample ID.
        """
        self._hdf_file.write_data(
            values,
            names,
            var_group,
            group_num=sample_id,
            hdf_node_path=self.__hdf_node_path,
        )

    @synchronized
    def get_data(self, index, **options):
        """Gets the data associated to a sample ID.

        :param str index: sample ID.
        :param options: options passed to the _read_data() method.
        :return: input data, output data and jacobian.
        :rtype: dict
        """
        h5file = h5py.File(self._hdf_file.hdf_file_path, "a")
        datum = super(HDF5Cache, self).get_data(index, h5_open_file=h5file)
        h5file.close()
        return datum

    @synchronized
    def _get_all_data(self):
        """Same as _all_data() but first open a file,
        then pass the opened file to the generator and lastly,
        close the file.
        """
        h5file = h5py.File(self._hdf_file.hdf_file_path, "a")
        for data in self._all_data(h5_open_file=h5file):
            yield data
        h5file.close()


class HDF5FileSingleton(with_metaclass(SingleInstancePerFileAttribute, object)):
    """
    Singleton to access a HDF file
    Used for multithreaded/multiprocessing access
    with a Lock
    """

    # We create a single instance of cache per HDF5 file
    # to allow the multiprocessing lock to work
    # Ie we want a single lock even if we instantiate multiple
    # caches
    HASH_TAG = "hash"
    INPUTS_GROUP = AbstractCache.INPUTS_GROUP
    OUTPUTS_GROUP = AbstractCache.OUTPUTS_GROUP
    JACOBIAN_GROUP = AbstractCache.JACOBIAN_GROUP

    def __init__(self, hdf_file_path):
        """
        Constructor

        :param hdf_file_path: path to the HDF5 file
        """
        self.hdf_file_path = hdf_file_path
        # Attach the lock to the file and NOT the Cache
        # because it is a singletton
        self.lock = RLock()

    def write_data(
        self, data, data_names, group_name, group_num, hdf_node_path, h5_open_file=None
    ):
        """Cache input data to avoid re evaluation

        :param data: the data to cache
        :param data_names: list of data names
        :param group_name: inputs or outputs or jacobian group
        :param hdf_node_path: name of the main HDF group
        :param h5_open_file: eventually the already opened file.
            this improves performance but is incompatible with
            multiprocess/treading
        """
        if h5_open_file is None:
            h5file = h5py.File(self.hdf_file_path, "a")
        else:
            h5file = h5_open_file

        node_group = h5file.require_group(hdf_node_path)
        num_group = node_group.require_group(str(group_num))
        io_group = num_group.require_group(group_name)

        try:
            # Write hash if needed
            if num_group.get(self.HASH_TAG) is None:
                data_hash = string_array([hash_data_dict(data, data_names)])
                num_group.create_dataset(self.HASH_TAG, data=data_hash)

            for data_name in data_names:
                val = data.get(data_name)
                if val is not None:
                    if val.dtype.type is unicode_:
                        io_group.create_dataset(
                            data_name, data=val.astype(string_dtype)
                        )
                    else:
                        io_group.create_dataset(data_name, data=to_real(val))
        except (RuntimeError, OSError) as err:
            h5file.close()
            node_path = str(hdf_node_path) + "." + str(group_num)
            node_path += "." + str(group_name)
            LOGGER.error(err)
            raise RuntimeError(
                "Failed to cache dataset "
                + node_path
                + " in file : "
                + str(self.hdf_file_path)
            )
        if h5_open_file is None:
            h5file.close()

    def read_data(self, group_number, group_name, hdf_node_path, h5_open_file=None):
        """
        Read a data dict in the hdf

        :param group_name: name of the group where data is written
        :param group_number: number of the group
         :param hdf_node_path: name of the main HDF group
        :returns: data dict and jacobian
        :param h5_open_file: eventually the already opened file.
            this improves performance but is incompatible with
            multiprocess/treading
        """
        if h5_open_file is None:
            h5file = h5py.File(self.hdf_file_path, "r")
        else:
            h5file = h5_open_file

        node = h5file[hdf_node_path]
        if not self._has_group(group_number, group_name, hdf_node_path, h5file):
            return None, None

        number_dataset = node[str(group_number)]
        values_group = number_dataset[group_name]
        data = {key: array(val) for key, val in values_group.items()}
        if group_name == self.INPUTS_GROUP:
            read_hash = number_dataset[self.HASH_TAG]
            data_hash = _long(array(read_hash)[0])
        else:
            data_hash = None

        if h5_open_file is None:
            h5file.close()
        ##########################################
        # Python  key.encode("ascii")
        ##########################################
        if PY2:
            data = {key.encode("ascii"): val for key, val in data.items()}

        for key, val in data.items():
            if val.dtype.type is bytes_:
                data[key] = val.astype(unicode_)

        return data, data_hash

    @staticmethod
    def _has_group(group_number, group_name, hdf_node_path, h5file):
        """
        Check if a group is present in the HDF file

        :param group_name: name of the group where data is written
        :param group_number: number of the group
        :param hdf_node_path: name of the main HDF group
        :param h5file: the opened HDF File pointer
        :returns: True if the group exists
        """
        node = h5file[hdf_node_path]
        number_dataset = node.get(str(group_number))
        if number_dataset is None:
            return False
        values_group = number_dataset.get(group_name)
        if values_group is None:
            return False
        return True

    def has_group(self, group_number, group_name, hdf_node_path):
        """
        Check if a group is present in the HDF file

        :param group_name: name of the group where data is written
        :param group_number: number of the group
        :param hdf_node_path: name of the main HDF group
        :returns: True if the group exists
        """
        h5file = h5py.File(self.hdf_file_path, "r")
        has_grp = self._has_group(group_number, group_name, hdf_node_path, h5file)
        h5file.close()
        return has_grp

    def read_hashes(self, hashes_dict, hdf_node_path):
        """
        Read the hashes in the HDF file
        :param hashes_dict: dict of hashes to fill
        :param hdf_node_path: name of the main HDF group
        :return:  max_group
        """
        if not exists(self.hdf_file_path):
            return 0
        # We must lock so that no data is added to the cache meanwhile
        h5file = h5py.File(self.hdf_file_path, "r")
        node_group = h5file.get(hdf_node_path)
        if node_group is None:
            return 0
        max_group = 0
        for group_num, group in node_group.items():
            group_num = int(group_num)
            read_hash = group[self.HASH_TAG]
            hash_value = _long(array(read_hash)[0])
            get_hash = hashes_dict.get(hash_value)
            if get_hash is None:
                hashes_dict[hash_value] = array([group_num])
            else:
                hashes_dict[hash_value] = append(get_hash, array([group_num]))
            max_group = max(max_group, group_num)
        return max_group

    def clear(self, hdf_node_path):
        """
        Clear the data in the cache
        :param hdf_node_path: node path to clear
        """
        h5file = h5py.File(self.hdf_file_path, "a")
        del h5file[hdf_node_path]
        h5file.close()
