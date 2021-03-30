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
"""
Dataset
=======

The :mod:`~gemseo.core.dataset` module implements the concept of dataset
which is a key element for machine learning, post-processing,
data analysis, ...

A :class:`.Dataset` is an object
defined by data stored as a dictionary of 2D numpy arrays,
whose rows are samples, a.k.a. realizations, and columns are features,
a.k.a. parameters or variables. The indices of this dictionary are either
names of groups of variables or names of variables.
A :class:`.Dataset` is also defined by
a list of variables names, a dictionary of variables sizes
and a dictionary of variables groups.

A :class:`.Dataset` can be set either from a numpy array or a file.
An :class:`.AbstractFullCache` or an :class:`.OptimizationProblem`
can also be exported to a :class:`.Dataset`
using :meth:`.AbstractFullCache.export_to_dataset`
and :meth:`.OptimizationProblem.export_to_dataset` respectively.

From a :class:`.Dataset`, we can easily access its length and get the data,
either as 2D array or as dictionaries indexed by the variables names.
We can get either the whole data,
data associated to a group or data associated to a list of variables.
It is also possible to export the :class:`.Dataset`
to an :class:`.AbstractFullCache` or a pandas DataFrame.

"""
from __future__ import absolute_import, division, unicode_literals

from numbers import Number

from future import standard_library
from numpy import array, concatenate, genfromtxt, hstack, ndarray, unique
from six import string_types

from gemseo.caches.cache_factory import CacheFactory
from gemseo.post.dataset.factory import DatasetPlotFactory
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import _long

standard_library.install_aliases()

from gemseo import LOGGER


class Dataset(object):
    """A Dataset is an object defined by data stored as a 2D numpy array,
    whose rows are samples, a.k.a. realizations, and columns are properties,
    a.k.a. parameters, variables or features. A dataset is also defined by
    a list of variables names, a dictionary of variables sizes
    and a dictionary of variables types. We can easily access its length
    and get the data, either as a 2D array or as a list of dictionaries
    indexed by the variables names. It is also possible to export the dataset
    to a :class:`.AbstractFullCache` or a pandas DataFrame."""

    PARAMETER_GROUP = "parameters"
    DESIGN_GROUP = "design_parameters"
    FUNCTION_GROUP = "functions"
    INPUT_GROUP = "inputs"
    OUTPUT_GROUP = "outputs"
    DEFAULT_GROUP = PARAMETER_GROUP
    DEFAULT_NAMES = {
        PARAMETER_GROUP: "x",
        DESIGN_GROUP: "dp",
        FUNCTION_GROUP: "func",
        INPUT_GROUP: "in",
        OUTPUT_GROUP: "out",
    }

    HDF5_CACHE = "HDF5Cache"
    MEMORY_FULL_CACHE = "MemoryFullCache"

    def __init__(self, name=None, by_group=True):
        """Constructor.

        :param str name: dataset name.
        :param bool group: if True, store the data by group. Otherwise,
            store them by variables. Default: True
        """
        self.name = name or self.__class__.__name__
        self._names = {}  # key = group, value = varnames
        self._groups = {}  # key = varname, value = group
        self.sizes = {}  # key = varname, value = varsize
        self._positions = {}
        self.dimension = {}  # key = group, value = groupsize
        self.length = 0
        self.data = {}
        self._group = by_group
        self.strings_encoding = None
        self._cached_inputs = []
        self._cached_outputs = []
        self._plot_factory = DatasetPlotFactory()
        self.metadata = {}

    def _clean(self):
        """ Clean dataset. """
        self._names = {}
        self._groups = {}
        self.sizes = {}
        self._positions = {}
        self.dimension = {}
        self.length = 0
        self.data = {}
        self.strings_encoding = None
        self._cached_inputs = []
        self._cached_outputs = []
        self.metadata = {}

    def is_group(self, group_name):
        """Returns True is group_name is a group.

        :param str group_name: group_name
        """
        return group_name in self._names

    def is_variable(self, variable_name):
        """Returns True is variable_name is a group.

        :param str variable_name: variable_name
        """
        return variable_name in self._groups

    def is_empty(self):
        """ Returns True if the dataset is empty. """
        return self.n_samples == 0

    def get_names(self, group_name):
        """Returns names for a given group.

        :param str group_name: group_name
        """
        return self._names.get(group_name)

    def get_group(self, variable_name):
        """Returns group for a given variable name.

        :param str variable_name: variable_name
        """
        return self._groups.get(variable_name)

    @property
    def variables(self):
        """ Names of variables. """
        return sorted(self._groups.keys())

    @property
    def groups(self):
        """ Names of the groups of variables. """
        return sorted(self._names.keys())

    def __str__(self):
        """ String representation. """
        string = self.name + "\n"
        string += "| Number of samples: " + str(self.n_samples) + "\n"
        string += "| Number of variables: " + str(self.n_variables) + "\n"
        string += "| Variables names and sizes by group: \n"
        for group, varnames in self._names.items():
            varnames = [name + " (" + str(self.sizes[name]) + ")" for name in varnames]
            varnames = ", ".join(varnames) + "\n"
            if varnames != "\n":
                string += "| - " + group + ": " + varnames
        total = str(sum(self.dimension.values()))
        string += "| Number of dimensions (total = " + total + ") by group:\n"
        for group, size in self.dimension.items():
            string += "| - " + group + ": " + str(size) + "\n"
        return string

    def __check_new_variable(self, variable):
        """Raise ValueError if variable is already defined.

        :param str variable: variable name.
        """
        if self.is_variable(variable):
            raise ValueError(variable + " is already defined.")
        if not isinstance(variable, string_types):
            raise TypeError("variable name must be a string.")

    def __check_new_group(self, group):
        """Raise ValueError if group is already defined.

        :param str group: group name.
        """
        if self.is_group(group):
            raise ValueError(group + " is already defined.")
        if not isinstance(group, string_types):
            raise TypeError("Group name must be a string.")

    def __check_length_consistency(self, length):
        """Raises ValueError if the length is different from the length of
        the dataset and if the latter is different from zero.

        :param int length: length.
        """
        if self.length != 0 and length != self.length:
            raise ValueError(
                "The number of rows of data must be equal to the"
                " length of the dataset."
            )
        self.length = length

    def __check_data_consistency(self, data):
        """Raises ValueError if the data is not a 2D numpy array
        or if its length is different from the length of the dataset.

        :param array data: data.
        """
        if not isinstance(data, ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy array.")
        self.__check_length_consistency(data.shape[0])

    @staticmethod
    def __check_variables_format(variables):
        """Raises TypeError if the format of variables is wrong.

        :param list(str) variables: list of variables names.
        """
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        if any([not isinstance(name, string_types) for name in variables]):
            raise TypeError("variables must be a list of strings.")

    @staticmethod
    def __check_sizes_format(sizes, variables, dimension):
        """Raises TypeError if the format of sizes is wrong.

        :param dict(int) sizes: dictionary of variables sizes.
        :param list(str) variables: list of variables names.
        :param int dimension: data dimension.
        """

        def is_size(size):
            return isinstance(size, (int, _long)) and size > 0

        if not isinstance(sizes, dict):
            raise TypeError("sizes must be a dictionary of positive integers.")
        if any([not is_size(sizes.get(name)) for name in variables]):
            raise TypeError("sizes must be a dictionary of positive integers.")
        total = sum([sizes[name] for name in variables])
        if total != dimension:
            raise ValueError(
                "The sum of variables sizes ({}) must be equal "
                "to the data dimension ({}).".format(total, dimension)
            )

    def __check_variables_sizes(self, variables, sizes, dimension):
        """Raises ValueError if the sum of the variables sizes is different
        from the number of data columns.

        :param list(str) variables: list of variables names.
        :param dict sizes: variables sizes.
        :param int dimension: data dimension.
        """
        if variables is not None:
            self.__check_variables_format(variables)
            if sizes is not None:
                self.__check_sizes_format(sizes, variables, dimension)

    def __get_default_group_variables(self, group, dimension, varname=None):
        """ Returns default variables names for a given group."""
        default_name = varname or self.DEFAULT_NAMES.get(group) or group
        variables = [default_name + "_" + str(index) for index in range(dimension)]
        sizes = {name: 1 for name in variables}
        groups = {name: group for name in variables}
        return variables, sizes, groups

    def __set_group_data(self, data, group, variables, sizes):
        """ Set data for a given group. """
        if self._group:
            self.data[group] = data
        else:
            array_to_dict = DataConversion.array_to_dict
            self.data.update(array_to_dict(data, variables, sizes))

    def __set_variable_data(self, name, data, group):
        """ Set data for a given group. """
        if self._group:
            if not self.is_group(group):
                self.data[group] = data
            else:
                self.data[group] = hstack((self.data[group], data))
        else:
            self.data[name] = data

    def __set_group_properties(self, group, variables, sizes, cache_as_input):
        """ Set properties for a given group. """
        self.sizes.update(sizes)
        self._groups.update({name: group for name in variables})
        self._names[group] = variables
        self.dimension[group] = sum([sizes[name] for name in variables])
        start = 0
        for name in variables:
            self._positions[name] = [start, start + self.sizes[name] - 1]
            if self._group:
                start += self.sizes[name]
            if cache_as_input:
                self._cached_inputs.append(name)
            else:
                self._cached_outputs.append(name)

    def __set_variable_properties(self, variable, group, size, cache_as_input):
        """ Set properties for a given variable. """
        self.sizes[variable] = size
        self._groups[variable] = group
        if not self.is_group(group):
            self._names[group] = [variable]
            self.dimension[group] = self.sizes[variable]
            self._positions[variable] = [0, self.sizes[variable] - 1]
        else:
            self._names[group].append(variable)
            if self._group:
                ncols = self.dimension[group]
                self._positions[variable] = [ncols, ncols + self.sizes[variable] - 1]
            else:
                self._positions[variable] = [0, self.sizes[variable] - 1]
            self.dimension[group] += self.sizes[variable]
        if cache_as_input:
            self._cached_inputs.append(variable)
        else:
            self._cached_outputs.append(variable)

    def add_group(
        self, group, data, variables=None, sizes=None, varname=None, cache_as_input=True
    ):
        """Add variable.

        :param str group: group name.
        :param ndarray data: data.
        :param list(str) variables: list of variables names.
        :param dict sizes: dictionary of variables sizes.
        :param str varname: variable name used if variables is None.
            If None, use the default variable name for group if it exists;
            otherwise, use the group name. Default: None.
        :param bool cache_as_input: cache as input when export to cache.
            Otherwise, as output. Default: True
        """
        self.__check_new_group(group)
        self.__check_data_consistency(data)
        self.__check_variables_sizes(variables, sizes, data.shape[1])
        if variables is None or sizes is None:
            get = self.__get_default_group_variables
            variables, sizes, _ = get(group, data.shape[1], varname)
        self.__set_group_data(data, group, variables, sizes)
        self.__set_group_properties(group, variables, sizes, cache_as_input)

    def add_variable(self, name, data, group=DEFAULT_GROUP, cache_as_input=True):
        """Add variable.

        :param str name: variable name.
        :param ndarray data: data.
        :param str group: group name. Default: DEFAULT_GROUP.
        :param bool cache_as_input: cache as input when export to cache.
            Otherwise, as output. Default: True
        """
        self.__check_new_variable(name)
        self.__check_data_consistency(data)
        self.__set_variable_data(name, data, group)
        self.__set_variable_properties(name, group, data.shape[1], cache_as_input)

    def __get_strings_encoding(self, data):
        """ Returns strings encoding. """
        self.strings_encoding = {name: {} for name in self._groups}
        encoding = {}
        if str(data.dtype).startswith(("|S", "<U")):
            data, encoding = self.__force_array_to_float(data)
        return data, encoding

    def set_from_array(
        self, data, variables=None, sizes=None, groups=None, default_name=None
    ):
        """Set Dataset from a numpy array or a dictionary of arrays

        :param array data: dataset.
        :param list(str) variables: list of variables names.
        :param dict(int) sizes: list of variables sizes.
        :param dict(str) groups: list of variables groups.
        :param str default_name: default variable name.
        """
        self._clean()
        self.__check_data_consistency(data)
        if variables is None:
            group = self.DEFAULT_GROUP
            get = self.__get_default_group_variables
            variables, sizes, groups = get(group, data.shape[1], default_name)
        else:
            self.__check_variables_format(variables)
        if sizes is None:
            sizes = {name: 1 for name in variables}
        self.__check_sizes_format(sizes, variables, data.shape[1])
        if groups is None:
            groups = {name: self.DEFAULT_GROUP for name in variables}
        self.__check_groups_format(groups, variables)
        self.__set_data_properties(variables, sizes, groups)
        data, encoding = self.__get_strings_encoding(data)
        self.__set_data(data, variables, encoding)

    def __check_groups_format(self, groups, variables):
        """ Check groups format and update it if necessary"""
        if not isinstance(groups, dict):
            raise TypeError(
                "groups must be a dictionary indexed by variables"
                "names whose values are strings."
            )
        for name in variables:
            if groups.get(name) is None:
                groups.update({name: self.DEFAULT_GROUP})
            elif not isinstance(groups[name], string_types):
                raise TypeError(
                    "groups must be a dictionary indexed "
                    "by variables names whose values are strings."
                )

    def __set_data(self, data, variables, encoding):
        """ Set data. """
        indices = {group: [] for group in self._names}
        start = 0
        end = 0
        for name in variables:
            end = start + self.sizes[name] - 1
            name_indices = list(range(start, end + 1))
            start = end + 1
            indices[self._groups[name]] += name_indices
            for key in encoding:
                if key in name_indices:
                    index = name_indices.index(key)
                    self.strings_encoding[name][index] = encoding[key]
            if not self._group:
                self.data[name] = data[:, name_indices]
        if self._group:
            for group in self._names:
                self.data[group] = data[:, indices[group]]

    def _set_variables_positions(self):
        """ Set variables positions. """
        for varnames in self._names.values():
            start = 0
            for varname in varnames:
                self._positions[varname] = [start, start + self.sizes[varname] - 1]
                if self._group:
                    start += self.sizes[varname]
                else:
                    start = 0

    def __set_data_properties(self, variables, sizes, groups):
        """Set properties for the whole data.

        :param list(str) variables: list of variables names.
        :param dict(int) sizes: dictionary of variables sizes.
        :param dict(str) groups: dictionary of variables groups.
        """
        for name in variables:
            if not self.is_group(groups[name]):
                self._names[groups[name]] = [name]
            else:
                self._names[groups[name]].append(name)
            self.sizes[name] = sizes[name]
            self._groups[name] = groups[name]
        for group, names in self._names.items():
            self.dimension[group] = sum([self.sizes[name] for name in names])
            if group == self.OUTPUT_GROUP:
                self._cached_outputs += names
            else:
                self._cached_inputs += names
        self._set_variables_positions()

    @staticmethod
    def __force_array_to_float(data):
        """Force a ndarray type to float.

        :param ndarray data: dataset.
        :return: data with float type.
        :rtype: ndarray
        """

        def __is_not_float(obj):
            """Return True if an object cannot be cast to float.

            :param obj: object to test.
            """
            try:
                float(obj)
                return False
            except ValueError:
                return True

        first_row = data[0, :]
        str_indices = [
            index for index, element in enumerate(first_row) if __is_not_float(element)
        ]

        strings_encoding = {}
        for index in range(data.shape[1]):
            if index in str_indices:
                column = list(data[:, index])
                encoding = dict(enumerate(unique(column)))
                strings_encoding[index] = encoding
                data[:, index] = unique(column, return_inverse=True)[1]

        data = data.astype("float")
        return data, strings_encoding

    def set_from_file(
        self,
        filename,
        variables=None,
        sizes=None,
        groups=None,
        delimiter=",",
        header=True,
    ):
        """Set Dataset from a file.

        :param str filename: file name.
        :param list(str) variables: list of variables names.
        :param dict(int) sizes: list of variables sizes.
        :param dict(str) groups: list of variables groups.
        :param str delimiter: field delimiter.
        :param bool header: if True, read the variables names
            on the first line of the file. Default: True.
        """
        self._clean()
        data = genfromtxt(filename, delimiter=delimiter, dtype="str")
        if header:
            if variables is None:
                variables = data[0, :].tolist()
            start_read = 1
        else:
            start_read = 0
        self.set_from_array(data[start_read:, :], variables, sizes, groups)

    def set_metadata(self, name, value):
        """Set metadata attribute.
        :param string name: Metadata attribute name.
        :param value: Metadata attribute value.
        """
        self.metadata[name] = value

    def _get_columns_names(self, as_list=False, start=0):
        """Return the names of the data columns. If dim(x)=1,
        its column name is 'x' while if dim(y)=2, its columns names
        are either 'x_0' and 'x_1' or ['x',0] and ['x',1].

        :param bool as_tuple: if True, return the name as a tuple.
            if False, return the name as a string. Default: False.
        :param int start: first index for components of a variable.
            Default: 0 ('x_0', 'x_1', ...).
        """
        tmp = []
        for _, names in self._names.items():
            for name in names:
                if as_list:
                    tmp += [
                        [self._groups[name], name, str(index + start)]
                        for index in range(self.sizes[name])
                    ]
                else:
                    if self.sizes[name] == 1:
                        tmp += [name]
                    else:
                        tmp += [
                            name + "_" + str(index + start)
                            for index in range(self.sizes[name])
                        ]
        return tmp

    def get_data_by_group(self, group, as_dict=False):
        """Returns data associated with a group.

        :param str group: variable group.
        :param bool as_dict: if True, return outputs values as dictionary.
            Default: False.
        """
        if not self.is_group(group):
            raise ValueError("{} is not an available group.".format(group))
        if group in self.data:
            data = self.data[group]
            if as_dict:
                data = DataConversion.array_to_dict(
                    self.data[group], self._names[group], self.sizes
                )
        else:
            data = {name: self.data[name] for name in self._names[group]}
            if not as_dict:
                data = DataConversion.dict_to_array(data, self._names[group])
        return data

    def get_data_by_names(self, names, as_dict=True):
        """Get data by variables names.

        :param list(str): names.
        :param bool as_dict: if True, return values as dictionary.
        """
        if isinstance(names, string_types):
            names = [names]
        dict_to_array = DataConversion.dict_to_array
        if not self._group:
            data = {name: self.data.get(name) for name in names}
        else:
            data = {}

            for name in names:
                indices = list(
                    range(self._positions[name][0], self._positions[name][1] + 1)
                )
                data[name] = self.data[self._groups[name]][:, indices]
        if not as_dict:
            data = dict_to_array(data, names)
        return data

    def get_all_data(self, by_group=True, as_dict=False):
        """Returns all data.

        :param str group: variable group.
        :param bool as_dict: if True, return outputs values as dictionary.
            Default: False.
        """
        if by_group:
            data = {
                group: self.get_data_by_group(group, as_dict) for group in self._names
            }
            if not as_dict:
                data = (data, self._names, self.sizes)
        else:
            if not as_dict:
                data = concatenate(
                    tuple([self.get_data_by_group(group) for group in self._names]), 1
                )
                names = [
                    item for sublist in list(self._names.values()) for item in sublist
                ]
                data = (data, names, self.sizes)
            else:
                data = {}
                for group in self._names:
                    data.update(self.get_data_by_group(group, as_dict))
        return data

    @property
    def n_variables(self):
        """ Return the number of variables. """
        return len(self._groups)

    def n_variables_by_group(self, group):
        """Return the number of variables for a group.

        :param str group: group name.
        """
        return len(self._names[group])

    @property
    def n_samples(self):
        """ Return the number of samples. """
        return self.length

    def __len__(self):
        """ Length of the object """
        return self.length

    def __bool__(self):
        """ returns True is the dataset is not empty. """
        return not self.is_empty()

    def export_to_dataframe(self, copy=True):
        """Export dataset to Dataframe.

        :param bool copy: If True, copy data. Otherwise, use reference.
            Default: True.
        """
        from pandas import DataFrame

        row1 = []
        row2 = []
        row3 = []
        for column in self._get_columns_names(True):
            row1.append(column[0])
            row2.append(column[1])
            row3.append(column[2])
        columns = [array(row1), array(row2), array(row3)]
        data = self.get_all_data(False, False)
        return DataFrame(data[0], columns=columns, copy=copy)

    def export_to_cache(
        self,
        inputs=None,
        outputs=None,
        cache_type=MEMORY_FULL_CACHE,
        cache_hdf_file=None,
        cache_hdf_node_name=None,
        **options
    ):
        """Export dataset to cache.

        :param list(str) inputs: names of the inputs to cache. If None,
            use all inputs. Default: None.
        :param list(str) outputs: names of the outputs to cache. If None,
            use all outputs. Default: None.
        :param str cache_type: type of cache to use.
        :param str cache_hdf_file: the file to store the data,
            mandatory when HDF caching is used
        :param str cache_hdf_node_name: name of the HDF
            dataset to store the discipline
            data. If None, self.name is used
        """
        if inputs is None:
            inputs = self._cached_inputs
        if outputs is None:
            outputs = self._cached_outputs
        create_cache = CacheFactory().create
        cache_hdf_node_name = cache_hdf_node_name or self.name
        if cache_type == self.HDF5_CACHE:
            cache = create_cache(
                cache_type,
                hdf_file_path=cache_hdf_file,
                hdf_node_path=cache_hdf_node_name,
                **options
            )
        else:
            cache = create_cache(cache_type, **options)
        for sample in range(len(self)):
            in_values = {name: self[(sample, name)][name][0] for name in inputs}
            out_values = {name: self[(sample, name)][name][0] for name in outputs}
            cache.cache_outputs(in_values, inputs, out_values, outputs)
        return cache

    def plot(self, name, show=True, save=False, **options):
        """Finds the appropriate library and executes
        the post processing on the problem

        :param str name: the post processing name
        :param show: show the figure (default: True)
        :param save: save the figure (default: False)
        :param options: options for the post method, see its package
        """
        options.update({"show": show, "save": save})
        post = self._plot_factory.create(name, dataset=self)
        post.execute(**options)
        return post

    def __getitem(self, indices, names):
        """Get items associated with sample indices and variables names.

        :param list(int) indices: samples indices.
        :param list(str) names: variables names.
        """
        for index in indices:
            is_lower = index < -1
            is_greater = index >= self.length
            is_int = isinstance(index, int)
            if index == self.n_samples:
                raise IndexError
            if is_lower or is_greater or not is_int:
                raise ValueError("{} is not a sample index.".format(index))
        for name in names:
            if name not in self._groups:
                raise ValueError("{} is not a variable name.".format(name))
        item = self.get_data_by_names(names)
        if len(indices) == 1:
            indices = indices[0]
        item = {name: value[indices, :] for name, value in item.items()}
        return item

    def __getitem__(self, key):
        """Gets item, where key is either int, str, list(int), list(str),
        (int, str), (int, list(str)), (list(int), str), (int, list(str)),
        or (list(int), list(str))."""
        indices = list(range(0, self.length))
        type_error_msg = "___getitem__ uses one of these argument types: "
        type_error_msg += "int, str, "
        type_error_msg += "list(int), list(str), "
        type_error_msg += "(int, str), (int, list(str)), "
        type_error_msg += "(list(int), str), (int, list(str)) "
        type_error_msg += "or (list(int), list(str)). "

        def getitem_list(index):
            """ Gets item when index is a list."""
            if all(isinstance(elem, string_types) for elem in index):
                item = self.__getitem(indices, index)
            elif all(isinstance(elem, Number) for elem in index):
                item = self.__getitem(index, self.variables)
            else:
                raise TypeError(type_error_msg)
            return item

        def getitem_tuple(tpl):
            """ Gets item when index is a tuple."""
            if isinstance(tpl[0], Number):
                indices = [tpl[0]]
            elif isinstance(tpl[0], slice):
                indices = list(range(tpl[0].start, tpl[0].stop, tpl[0].step or 1))
            elif isinstance(tpl[0], list):
                if all(isinstance(elem, Number) for elem in tpl[0]):
                    indices = tpl[0]
                else:
                    raise TypeError(type_error_msg)
            else:
                raise TypeError(type_error_msg)
            if isinstance(tpl[1], string_types):
                names = [tpl[1]]
            elif isinstance(tpl[1], list):
                if all(isinstance(elem, string_types) for elem in tpl[1]):
                    names = tpl[1]
                else:
                    raise TypeError(type_error_msg)
            else:
                raise TypeError(type_error_msg)
            return self.__getitem(indices, names)

        if isinstance(key, Number):
            item = self.__getitem([key], self.variables)
        elif isinstance(key, string_types):
            item = self.__getitem(indices, [key])
        elif isinstance(key, list):
            item = getitem_list(key)
        elif isinstance(key, slice):
            item = getitem_list(list(range(key.start, key.stop, key.step or 1)))
        elif isinstance(key, tuple) and len(key) == 2:
            item = getitem_tuple(key)
        else:
            raise TypeError(type_error_msg)
        return item
