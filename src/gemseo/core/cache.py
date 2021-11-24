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
import sys
from hashlib import sha1
from numbers import Number

from numpy import (
    append,
    array,
    array_equiv,
    ascontiguousarray,
    complex128,
    concatenate,
    float64,
    uint8,
    vstack,
)
from numpy.linalg import norm

from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.locks import synchronized, synchronized_hashes
from gemseo.utils.multi_processing import Manager, RLock, Value

TYPE_ERR_MSG = "__getitem__ uses one of these argument types: "
TYPE_ERR_MSG += "int, str, "
TYPE_ERR_MSG += "list(int), list(str), "
TYPE_ERR_MSG += "(int, str), (int, list(str)), "
TYPE_ERR_MSG += "(list(int), str), (int, list(str)) "
TYPE_ERR_MSG += "or (list(int), list(str)). "

LOGGER = logging.getLogger(__name__)


class AbstractCache(object):
    """Abstract class for caches: Defines the common methods for caching inputs,
    outputs, and jacobians of a MDODiscipline.

    See also
    --------
    :class:`.SimpleCache`: store the last evaluation
    :class:`.MemoryFullCache`: store all data in memory
    :class:`.HDF5Cache`: store all data in an HDF5 file
    """

    SAMPLE_GROUP = "sample"
    INPUTS_GROUP = "inputs"
    OUTPUTS_GROUP = "outputs"
    JACOBIAN_GROUP = "jacobian"

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
        """
        self.tolerance = tolerance
        self.name = name if name is not None else self.__class__.__name__
        self._varsizes = None
        self.__inputs_names = None
        self.__outputs_names = None

    def __bool__(self):
        """returns True is the cache is not empty."""
        return len(self) > 0

    @property
    def inputs_names(self):
        """Return the inputs names."""
        if self.__inputs_names is None:
            if not self:
                return None
            data1 = self.get_data(1)[self.INPUTS_GROUP]
            self.__inputs_names = list(iter(data1.keys()))
        return self.__inputs_names

    @property
    def outputs_names(self):
        """Return the outputs names."""
        if self.__outputs_names is None:
            if not self:
                return None
            data1 = self.get_data(1)[self.OUTPUTS_GROUP]
            self.__outputs_names = list(iter(data1.keys()))
        return self.__outputs_names

    @property
    def varsizes(self):
        """Return the variables sizes."""

        def length(obj):
            """Length of an object if __len__ attribute exists, else 1."""
            return len(obj) if hasattr(obj, "__len__") else 1

        if self._varsizes is None:
            inputs = self.get_last_cached_inputs()
            outputs = self.get_last_cached_outputs()
            if inputs is not None and outputs is not None:
                self._varsizes = {key: length(val) for key, val in inputs.items()}
                self._varsizes.update(
                    {key: length(val) for key, val in outputs.items()}
                )
        return self._varsizes

    def __str__(self):
        msg = "Name: {}\n".format(self.name)
        msg += "Type: {}\n".format(self.__class__.__name__)
        msg += "Tolerance: {}\n".format(self.tolerance)
        msg += "Input names: {}\n".format(self.inputs_names)
        msg += "Output names: {}\n".format(self.outputs_names)
        msg += "Length: {}".format(len(self))
        return msg

    def __len__(self):
        return self.get_length()

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
        """
        raise NotImplementedError()

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
        """
        raise NotImplementedError()

    def get_outputs(self, input_data, input_names=None):
        """Check if the discipline has already been evaluated for the given input data
        dictionary. If True, return the associated cache, otherwise return None.

        Parameters
        ----------
        input_data : dict
            Input data dictionary to test for caching.
        input_names : list(str)
            List of input data names.
            If None, takes them all

        Returns
        -------
        output_data : dict
            Output data if there is no need to evaluate the discipline.
            None otherwise.
        jacobian : dict
            Jacobian if there is no need to evaluate the discipline.
            None otherwise.
        """
        raise NotImplementedError()

    def clear(self):
        """Clear the cache."""
        self.__inputs_names = None
        self.__outputs_names = None
        self._varsizes = None

    def get_last_cached_inputs(self):
        """Retrieve the last execution inputs.

        Returns
        -------
        inputs : dict
            Last cached inputs.
        """
        raise NotImplementedError()

    def get_last_cached_outputs(self):
        """Retrieve the last execution outputs.

        Returns
        -------
        outputs : dict
            Last cached outputs.
        """
        raise NotImplementedError()

    def get_all_data(self, **options):
        """Read all the data in the cache.

        Returns
        -------
        all_data : dict
            A dictionary of dictionaries for inputs, outputs and jacobian
            where keys are data indices.
        """
        raise NotImplementedError()

    def get_length(self):
        """Get the length of the cache, ie the number of stored elements.

        Returns
        -------
        length : int
            Length of the cache.
        """
        raise NotImplementedError()

    @property
    def max_length(self):
        """Get the maximal length of the cache (the maximal number of stored elements).

        Returns
        -------
        length : int
            Maximal length of the cache.
        """
        raise NotImplementedError()

    def _check_index(self, index):
        """Check if index is a sample index.

        :param int index: index.
        """
        is_integer = isinstance(index, Number)
        if not is_integer or index < 1 or index > self.get_length():
            raise ValueError(str(index) + " is not a sample index.")

    def get_data(self, index, **options):
        """Returns an elementary sample.

        :param index: sample index.
        :type index: int
        :param options: getter options
        """
        raise NotImplementedError()

    @property
    def samples_indices(self):
        """List of samples indices."""
        return range(1, self.get_length() + 1)


class AbstractFullCache(AbstractCache):
    """Abstract cache to store all data, either in memory or on the disk.

    See also
    --------
    :class:`.MemoryFullCache`: store all data in memory
    :class:`.HDF5Cache`: store all data in an HDF5 file
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
        """
        super(AbstractFullCache, self).__init__(tolerance, name)
        self.lock_hashes = RLock()
        self._manager = Manager()
        self._hashes = self._manager.dict()
        # Maximum index of the data stored in the subgroup of the node
        self._max_group = Value("i", 0)
        self._last_accessed_group = Value("i", 0)
        self.lock = self._set_lock()

    def _set_lock(self):
        """Sets a lock for multithreading, either from an external object or internally
        by using RLock()."""
        raise NotImplementedError

    def __get_or_increment_group_num(self, input_data, data_hash):
        """This method is the second step of caching new inputs. Either gets the right
        cache group number if the inputs are already cached or creates a new one if
        needed.

        :param dict input_data: the input data to cache
        :param int data_hash: the hash of the data
        :return: True if a group was created
        :rtype: bool
        """

        hash_groups = self._hashes.get(data_hash)
        if hash_groups is None:  # Checks if this hash is already stored
            self._max_group.value += 1
            self._last_accessed_group.value = self._max_group.value
            self._hashes[data_hash] = array([self._max_group.value])
            self._initialize_entry(self._max_group.value)
            return True

        # If yes, append to the group
        for group in hash_groups:
            read_in_data = self._read_data(group, self.INPUTS_GROUP)
            if check_cache_equal(input_data, read_in_data):
                # Data is already cached !
                # We dont store it again
                self._last_accessed_group.value = group
                return False

        # Inputs not yet cached, but hashes are equal
        # we need to update the group
        self._max_group.value += 1
        self._last_accessed_group.value = self._max_group.value
        self._hashes[data_hash] = append(hash_groups, self._max_group.value)
        self._initialize_entry(self._max_group.value)
        return True

    def _initialize_entry(self, sample_id):
        """Initialize an entry of the dataset if needed.

        :param int sample_id: sample ID.
        """

    def _has_group(self, sample_id, var_group):
        """Checks if the dataset has the particular variables group filled in.

        :param int sample_id: sample ID.
        :param str var_group: name of the variables group.
        :return: True if the variables group is filled in.
        :rtype: bool
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

    def _cache_inputs(self, input_data, input_names, out_group_to_check):
        """This method is the first step step of caching new inputs or Jacobian. It
        caches inputs and increments group if needed. Checks if the out_group_to_check
        exists for these inputs.

        :param dict input_data: data dictionary where keys are variables names
            and values are variables values (numpy arrays).
        :param list(str) input_names: list of input data names
        :param out_group_to_check: name of the variables group to check
            existence, either AbstractCache.OUTPUTS_GROUP or
            AbstractCache.JACOBIAN_GROUP.
        :return: True if the data group exists, avoids duplicate storage.
        :rtype: bool
        """
        data_hash = hash_data_dict(input_data, input_names)
        new_grp = self.__get_or_increment_group_num(input_data, data_hash)

        write_inputs = True
        if not new_grp:
            # The Jacobian may have been stored before but not the outputs yet
            # If the data is already stored, returns
            # Otherwise, store outputs but dont write inputs again
            group_number = self._last_accessed_group.value
            if self._has_group(group_number, out_group_to_check):
                return True
            # Inputs are cached
            write_inputs = False

        if write_inputs:
            self._write_data(
                input_data, input_names, self.INPUTS_GROUP, self._max_group.value
            )

        return False

    @synchronized
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
        """
        data_exists = self._cache_inputs(input_data, input_names, self.OUTPUTS_GROUP)

        if data_exists:
            return

        data_to_cache = output_names or output_data.keys()

        self._write_data(
            output_data, data_to_cache, self.OUTPUTS_GROUP, self._max_group.value
        )

    @synchronized
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
        """
        data_exists = self._cache_inputs(input_data, input_names, self.JACOBIAN_GROUP)
        if data_exists:
            return

        flat_jac = DataConversion.dict_jac_to_dict(jacobian)

        self._write_data(
            flat_jac,
            flat_jac.keys(),
            self.JACOBIAN_GROUP,
            self._last_accessed_group.value,
        )

        return

    @synchronized
    def clear(self):
        """Clears the cache."""
        super(AbstractFullCache, self).clear()
        self._hashes = self._manager.dict()
        self._max_group.value = 0
        self._last_accessed_group.value = 0

    @synchronized
    def get_last_cached_inputs(self):
        """Retrieve the last execution inputs.

        Returns
        -------
        inputs : dict
            Last cached inputs.
        """
        if self._max_group.value == 0:
            return None
        return self._read_data(self._last_accessed_group.value, self.INPUTS_GROUP)

    @synchronized
    def get_last_cached_outputs(self):
        """Retrieve the last execution outputs.

        Returns
        -------
        outputs : dict
            Last cached outputs.
        """
        if self._max_group.value == 0:
            return None
        return self._read_data(self._last_accessed_group.value, self.OUTPUTS_GROUP)

    @synchronized
    def get_length(self):
        """Get the length of the cache, ie the number of stored elements.

        Returns
        -------
        length : int
            Length of the cache.
        """
        return self._max_group.value

    @property
    def max_length(self):
        """Get the maximal length of the cache (the maximal number of stored elements).

        Returns
        -------
        length : int
            Maximal length of the cache.
        """
        return sys.maxsize

    def _read_data(self, group_number, group_name, **options):
        """Read data from a data provider defined in the overloaded classes.

        :param str group_name: name of the group where data is written.
        :param int group_number: number of the group.
        :param options: options passed to the overloaded methods.
        :returns: data dict and jacobian
        """
        raise NotImplementedError()

    @synchronized_hashes
    def __has_hash(self, data_hash):
        """Get data hash.

        :param int data_hash: the hash of the data.
        :return: the data hash.
        """
        out = self._hashes.get(data_hash)
        return out

    def _read_group(self, group_nums, input_data):
        """Read output data and jacobian associated to given input data inside given
        groups.

        :param list(int) group_nums: group numbers.
        :param dict input_data: input data dictionary.
        :returns: The output data and jacobian
            if there is no need to evaluate the
            discipline, None otherwise.
        :rtype: dict, dict
        """
        for group in group_nums:
            read_in_data = self._read_data(group, self.INPUTS_GROUP)
            if check_cache_equal(input_data, read_in_data):
                out_data = self._read_data(group, self.OUTPUTS_GROUP)
                jac = self._read_data(group, self.JACOBIAN_GROUP)
                # self._last_accessed_group.value = group
                return out_data, jac
        return None, None

    @synchronized
    def get_outputs(self, input_data, input_names=None):
        """Check if the discipline has already been evaluated for the given input data
        dictionary. If True, return the associated cache, otherwise return None.

        Parameters
        ----------
        input_data : dict
            Input data dictionary to test for caching.
        input_names : list(str)
            List of input data names.

        Returns
        -------
        output_data : dict
            Output data if there is no need to evaluate the discipline.
            None otherwise.
        jacobian : dict
            Jacobian if there is no need to evaluate the discipline.
            None otherwise.
        """
        if self.tolerance == 0.0:
            data_hash = hash_data_dict(input_data, input_names)
            group_nums = self.__has_hash(data_hash)
            if group_nums is None:
                return None, None
            return self._read_group(group_nums, input_data)
        for group_nums in self._hashes.values():
            for group in group_nums:
                read_in_data = self._read_data(group, self.INPUTS_GROUP)
                if check_cache_approx(input_data, read_in_data, self.tolerance):
                    out_data = self._read_data(group, self.OUTPUTS_GROUP)
                    jac = self._read_data(group, self.JACOBIAN_GROUP)
                    # self._last_accessed_group.value = group
                    return out_data, jac
        return None, None

    @property
    def _all_groups(self):
        """Sorted group numbers."""
        tmp = []
        for group_nums in self._hashes.values():
            tmp += list(group_nums)
        return sorted(tmp)

    @synchronized
    def _get_all_data(self):
        """Same as _all_data() but with pre- and post- treatment."""
        for data in self._all_data():
            yield data

    @synchronized
    def _all_data(self, **options):
        """Iterator of all data in the cache.

        :returns: sample ID, input data, output data and jacobian.
        :rtype: dict
        """
        for group in self._all_groups:
            read_in_data = self._read_data(group, self.INPUTS_GROUP, **options)
            out_data = self._read_data(group, self.OUTPUTS_GROUP, **options)
            jac = self._read_data(group, self.JACOBIAN_GROUP, **options)
            yield {
                self.SAMPLE_GROUP: group,
                self.INPUTS_GROUP: read_in_data,
                self.OUTPUTS_GROUP: out_data,
                self.JACOBIAN_GROUP: jac,
            }

    @synchronized
    def get_data(self, index, **options):
        """Gets the data associated to a sample ID.

        :param str index: sample ID.
        :param options: options passed to the _read_data() method.
        :return: input data, output data and jacobian.
        :rtype: dict
        """
        self._check_index(index)
        input_data = self._read_data(index, self.INPUTS_GROUP, **options)
        output_data = self._read_data(index, self.OUTPUTS_GROUP, **options)
        jacobian = self._read_data(index, self.JACOBIAN_GROUP, **options)
        return {
            self.INPUTS_GROUP: input_data,
            self.OUTPUTS_GROUP: output_data,
            self.JACOBIAN_GROUP: jacobian,
        }

    @synchronized
    def get_all_data(self, as_iterator=False):  # pylint: disable=W0221
        """Return all the data in the cache.

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
        """
        if as_iterator:
            return self._get_all_data()

        all_data = {}
        for data in self._get_all_data():
            sample = data[self.SAMPLE_GROUP]
            inputs = data[self.INPUTS_GROUP]
            outputs = data[self.OUTPUTS_GROUP]
            jac = data[self.JACOBIAN_GROUP]
            all_data[sample] = {
                self.INPUTS_GROUP: inputs,
                self.OUTPUTS_GROUP: outputs,
                self.JACOBIAN_GROUP: jac,
            }
        return all_data

    def export_to_ggobi(self, file_path, inputs_names=None, outputs_names=None):
        """Export history to xml file format for ggobi tool.

        Parameters
        ----------
        file_path : str
            Path to export the file.
        inputs_names : list(str)
            List of inputs to include in the export.
            By default, take all of them.
        outputs_names : list(str)
            Names of outputs to export.
            By default, take all of them.
        """
        n_groups = len(self._hashes)
        if n_groups == 0:
            raise ValueError("Empty cache!")
        in_n = None
        out_n = None
        in_d = []
        out_d = []
        len_dict = {}

        for data in self.get_all_data(True):
            in_dict = data[self.INPUTS_GROUP] or {}
            out_dict = data[self.OUTPUTS_GROUP] or {}
            try:
                if inputs_names is not None:
                    in_dict = {key: in_dict[key] for key in inputs_names}
                if outputs_names is not None:
                    out_dict = {key: out_dict[key] for key in outputs_names}
            except KeyError:
                # The data is not in this execution
                continue
            # Compute the size of the data
            len_dict.update({key: val.size for key, val in in_dict.items()})
            len_dict.update({key: val.size for key, val in out_dict.items()})
            in_keys = set(in_dict.keys())
            out_keys = set(out_dict.keys())
            in_n = (in_n or in_keys) & in_keys
            out_n = (out_n or out_keys) & out_keys
            in_d.append(in_dict)
            out_d.append(out_dict)

        if not out_d:
            raise ValueError("Failed to find outputs in the cache")

        variables_names = []
        for data_name in list(in_n) + list(out_n):
            len_data = len_dict[data_name]
            if len_data == 1:
                variables_names.append(data_name)
            else:
                variables_names += [
                    data_name + "_" + str(i + 1) for i in range(len_data)
                ]
        values_list = (
            concatenate(
                [in_d[g][i].flatten() for i in in_n]
                + [out_d[g][o].flatten() for o in out_n]
            )
            for g in range(len(in_d))
        )
        values_array = vstack(values_list)
        save_data_arrays_to_xml(variables_names, values_array, file_path)

    def merge(self, other_cache):
        """Merges an other cache with self.

        Parameters
        ----------
        other_cache : AbstractFullCache
            Cache to merge with the current one.
        """
        for data in other_cache.get_all_data(True):
            input_data = data[self.INPUTS_GROUP]
            output_data = data[self.OUTPUTS_GROUP]
            jacobian = data[self.JACOBIAN_GROUP]
            if other_cache.inputs_names is not None:
                inputs_names = other_cache.inputs_names
            else:
                inputs_names = self.inputs_names
            outputs_names = other_cache.outputs_names or self.outputs_names
            self.cache_outputs(input_data, inputs_names, output_data, outputs_names)
            if jacobian is not None:
                self.cache_jacobian(input_data, inputs_names, jacobian)

    def _duplicate_from_scratch(self):
        """Duplicate a cache from scratch, ie.

        only duplicate the construction step.
        """
        raise NotImplementedError()

    def __add__(self, other_cache):
        """Add another cache to the current cache and returns the sum. dtype="bytes"))

            for data_name in data_names:
                val = data.get(data_name)
                if val is not None:
                    io_group.create_dataset(data_name, data=to_real(val))
        except RuntimeError as err:
            h5file.close()
            node_path = str(hdf_node_path) + "." + str(group_num)
            node_path += "." + str(group_name)
            raise RuntimeError("Failed to cache dataset " + node_path +
                               " in file : " + str(self.hdf_file_path) +
                               " h5py error : " + str(err.args[0]))
        data = {key: array(val) for key, val in values_group.items()}
            data_hash = int(array(read_hash)[0])

        :param AbstractCache other_cache: other cache
        :return: cache with the same class as the first cache in the sum.
        """
        new_cache = self._duplicate_from_scratch()
        new_cache.merge(self)
        new_cache.merge(other_cache)
        return new_cache

    def export_to_dataset(
        self,
        name=None,
        by_group=True,
        categorize=True,
        inputs_names=None,
        outputs_names=None,
    ):
        """Set Dataset from a cache.

        :param str name: dataset name.
        :param bool by_group: if True, store the data by group. Otherwise,
            store them by variables. Default: True
        :param bool categorize: distinguish between the different groups of
            variables. Default: True.
        :param list(str) inputs_names: list of inputs names. If None, use all
            inputs. Default: None.
        :param list(str) outputs_names: list of outputs names. If None, use all
            outputs. Default: None.
        """
        from gemseo.core.dataset import Dataset

        dataset = Dataset(name or self.name, by_group)

        to_array = DataConversion.list_of_dict_to_array

        inputs_names = inputs_names or self.inputs_names
        outputs_names = outputs_names or self.outputs_names

        # Set the different groups
        in_grp = out_grp = dataset.DEFAULT_GROUP
        cache_output_as_input = True
        if categorize:
            in_grp = dataset.INPUT_GROUP
            out_grp = dataset.OUTPUT_GROUP
            cache_output_as_input = False

        # Add cache inputs and outputs
        data = list(self.get_all_data(True))
        inputs = to_array(data, inputs_names, self.INPUTS_GROUP)
        data = DataConversion.array_to_dict(inputs, inputs_names, self.varsizes)
        for input_name, value in data.items():
            dataset.add_variable(input_name, value, in_grp)
        data = list(self.get_all_data(True))
        outputs = to_array(data, outputs_names, self.OUTPUTS_GROUP)
        data = DataConversion.array_to_dict(outputs, outputs_names, self.varsizes)
        for output_name, value in data.items():
            dataset.add_variable(
                output_name, value, out_grp, cache_as_input=cache_output_as_input
            )
        return dataset


def hash_data_dict(data, names_tokeep=None):
    """Hash a data dict using sha1.
        for group_num, group in node_group.items():
            hash_value = int(array(read_hash)[0])

    Parameters
    ----------
    data : dict
        The data dictionary.
    names_tokeep : list(str)
        Names of the data to keep for hashing.
        If None, use sorted(data.keys()).

    Returns
    -------
    hash : int
        Hash value of the data dictionary.

    Examples
    --------
    >>> from gemseo.core.cache import hash_data_dict
    >>> from numpy import array
    >>> data = {'x':array([1.,2.]),'y':array([3.])}
    >>> hash_data_dict(data)
    1871784392126344814771968055738742895695521374568L
    >>> hash_data_dict(data,'x')
    756520441774735697349528776513537427923146459919L
    """
    names_with_hashed_values = []

    if names_tokeep is None:
        names = iter(data.keys())
    else:
        names = names_tokeep

    for name in sorted(names):
        value = data.get(name)
        if value is None:
            continue

        try:
            value = value.view(uint8)
        except (ValueError, AttributeError):
            # View may not support discontiguous arrays
            value = ascontiguousarray(value).view(uint8)

        hashed_value = int(sha1(value).hexdigest(), 16)
        names_with_hashed_values.append((name, hashed_value))

    return hash(tuple(names_with_hashed_values))


def check_cache_approx(data_dict, cache_dict, cache_tol=0.0):
    """Checks if the data_dict is approximately equal to the cache dict at
    self.tolerance (absolute + relative)

    :param data_dict: data dict to check
    :returns: True if the dict are approximately equal
    """
    for key, val in cache_dict.items():
        norm_err = norm((data_dict[key] - val))
        if norm_err > cache_tol * (1.0 + norm(val)):
            return False

    return True


def check_cache_equal(data_dict, cache_dict):
    """Check if the data dictionary is equal to the cache data dictionary.

    Parameters
    ----------
    data_dict : dict
        Data dictionary to check.
    cache_dict : dict
        Cache data dictionary to check.

    Returns
    -------
    is_equal : bool
        True if the dictionaries are equal.

    Examples
    --------
    >>> from numpy import array
    >>> data_1 = {'x': array([1.]), 'y': array([2.])}
    >>> data_2 = {'x': array([1.]), 'y': array([3.])}
    >>> check_cache_equal(data_1, data_1)
    True
    >>> check_cache_equal(data_1, data_2)
    False
    """
    for key, buff_val in cache_dict.items():
        in_val = data_dict.get(key)
        if in_val is None or not array_equiv(in_val, buff_val):
            return False
    return True


def to_real(data):
    """Convert complex to real numpy array."""
    if data.dtype == complex128:
        return array(array(data, copy=False).real, dtype=float64)
    return data
