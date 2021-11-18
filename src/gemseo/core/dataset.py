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
"""A generic dataset to store data in memory.

This module implements the concept of dataset
which is a key element for machine learning, post-processing, data analysis, ...

A :class:`.Dataset` uses its attribute :attr:`.Dataset.data`
to store :math:`N` series of data
representing the values of :math:`p` multidimensional features
belonging to different groups of features.

This attribute :attr:`.Dataset.data` is a dictionary of 2D numpy arrays,
whose rows are the samples, a.k.a. series, realizations or entries,
and columns are the variables, a.k.a. parameters or features.
The keys of this dictionary are
either the names of the groups of variables
or the names of the variables.
Thus, a :class:`.Dataset` is not only defined by the raw data stored
but also by the names, the sizes and the groups of the different variables.

A :class:`.Dataset` can be set
either from a file (:meth:`.Dataset.set_from_file`)
or from a numpy arrays (:meth:`.Dataset.set_from_array`),
and can be enriched from a group of variables (:meth:`.Dataset.add_group`)
or from a single variable (:meth:`.Dataset.add_variable`).

An :class:`.AbstractFullCache` or an :class:`.OptimizationProblem`
can also be exported to a :class:`.Dataset`
using :meth:`.AbstractFullCache.export_to_dataset`
and :meth:`.OptimizationProblem.export_to_dataset` respectively.

From a :class:`.Dataset`,
we can easily access its length and data,
either as 2D array or as dictionaries indexed by the variables names.
We can get either the whole data,
or the data associated to a group or the data associated to a list of variables.
It is also possible to export the :class:`.Dataset`
to an :class:`.AbstractFullCache` or a pandas DataFrame.
"""
from __future__ import division, unicode_literals

import logging
import operator
from numbers import Number
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from numpy import (
    array,
    concatenate,
    delete,
    genfromtxt,
    hstack,
    isnan,
    ndarray,
    unique,
    where,
)
from six import string_types

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.cache import AbstractFullCache
from gemseo.post.dataset.factory import DatasetPlotFactory
from gemseo.utils.data_conversion import DataConversion
from gemseo.utils.py23_compat import long
from gemseo.utils.string_tools import MultiLineString, pretty_repr

LOGGER = logging.getLogger(__name__)

LOGICAL_OPERATORS = {
    "<=": operator.le,
    "<": operator.lt,
    ">=": operator.ge,
    ">": operator.gt,
    "==": operator.eq,
    "!=": operator.ne,
}

ItemType = Union[
    int, str, List[int], List[str], Tuple[Union[int, List[int]], Union[str, List[str]]]
]
AllDataType = Union[
    Dict[str, Union[Dict[str, ndarray], ndarray]],
    Tuple[Union[ndarray, Dict[str, ndarray]], List[str], Dict[str, int]],
]


class Dataset(object):
    """A generic class to store data.

    Attributes:
        name (str): The name of the dataset.
        sizes (Dict[str,int]): The sizes of the variables.
        dimension (Dict[str,int]): The dimensions of the groups of variables.
        length (int): The length of the dataset.
        strings_encoding (Dict): The encoding structure,
            mapping the values of the string variables with integers;
            the keys are the names of the variables
            and the values are dictionaries
            whose keys are the components of the variables
            and the values are the integer values.
        metadata (Dict[str,Any]): The metadata
            used to store any kind of information that are not variables,
            e.g. the mesh associated with a multi-dimensional variable.
    """

    PARAMETER_GROUP = "parameters"
    DESIGN_GROUP = "design_parameters"
    FUNCTION_GROUP = "functions"
    INPUT_GROUP = "inputs"
    OUTPUT_GROUP = "outputs"
    GRADIENT_GROUP = "gradients"
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

    def __init__(
        self,
        name=None,  # type: Optional[str]
        by_group=True,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            name: The name of the dataset.
                If None, use the name of the class.
            by_group: If True, store the data by group.
                Otherwise, store them by variables.
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
        self.metadata = {}
        self.__row_names = None

    def remove(
        self,
        entries,  # type: Union[List[int],ndarray]
    ):  # type: (...) -> None
        """Remove entries.

        Args:
            entries: The entries to be removed,
                either indices
                or a boolean 1D array
                whose length is equal to the length of the dataset
                and elements to delete are coded True.
        """
        if isinstance(entries, ndarray):
            entries = self.find(entries)
        self.length -= len(entries)
        for name, value in list(self.data.items()):
            self.data[name] = delete(value, entries, 0)

    @staticmethod
    def find(
        comparison,  # type: ndarray
    ):  # type: (...) -> List[int]
        """Find the entries for which a comparison is satisfied.

        This search uses a boolean 1D array
        whose length is equal to the length of the dataset.

        Args:
            comparison: A boolean vector whose length is equal to the number of samples.

        Returns:
            The indices of the entries for which the comparison is satisfied.
        """
        return where(comparison)[0].tolist()

    def is_nan(self):  # type: (...) -> ndarray
        """Check if an entry contains NaN.

        Returns:
             Whether any entries is NaN or not.
        """
        return isnan(self.get_all_data(False)[0]).any(1)

    def compare(
        self,
        value_1,  # type: Union[str,float]
        logical_operator,  # type: str
        value_2,  # type: Union[str,float]
        component_1=0,  # type: int
        component_2=0,  # type: int
    ):  # type: (...) -> ndarray
        """Compare either a variable and a value or a variable and another variable.

        Args:
            value_1: The first value,
                either a variable name or a numeric value.
            logical_operator: The logical operator,
                either "==", "<", "<=", ">" or ">=".
            value_2: The second value,
                either a variable name or a numeric value.
            component_1: If value_1 is a variable name,
                component_1 corresponds to its component used in the comparison.
            component_2: If value_2 is a variable name,
                component_2 corresponds to its component used in the comparison.

        Returns:
             Whether the comparison is valid for the different entries.
        """
        if value_1 not in self.variables and value_2 not in self.variables:
            raise ValueError(
                "Either value_1 ({}) or value_2 ({}) "
                "must be a variable name from the list: {}".format(
                    value_1, value_2, self.variables
                )
            )
        if value_1 in self.variables:
            value_1 = self[value_1][value_1][:, component_1]
        if value_2 in self.variables:
            value_2 = self[value_2][value_2][:, component_2]
        try:
            result = LOGICAL_OPERATORS[logical_operator](value_1, value_2)
        except KeyError:
            raise ValueError(
                "{} is not a logical operator: "
                "use either '==', '<', '<=', '>' or '>='".format(logical_operator)
            )
        return result

    def _clean(self):  # type: (...) -> None
        """Remove all data from the dataset."""
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

    def is_group(
        self,
        name,  # type: str
    ):  # type: (...) -> bool
        """Check if a name is a group name.

        Args:
            name: A name of a group.

        Returns:
            Whether the name is a group name.
        """
        return name in self._names

    def is_variable(
        self,
        name,  # type: str
    ):  # type: (...) -> bool
        """Check if a name is a variable name.

        Args:
            name: A name of a variable.

        Returns:
            Whether the name is a variable name.
        """
        return name in self._groups

    def is_empty(self):  # type: (...) -> bool
        """Check if the dataset is empty.

        Returns:
            Whether the dataset is empty.
        """
        return self.n_samples == 0

    def get_names(
        self,
        group_name,  # type: str
    ):  # type: (...) -> List[str]
        """Get the names of the variables of a group.

        Args:
            group_name: The name of the group.

        Returns:
            The names of the variables of the group.
        """
        return self._names.get(group_name)

    def get_group(
        self,
        variable_name,  # type: str
    ):  # type: (...) -> str
        """Get the name of the group that contains a variable.

        Args:
            variable_name: The name of the variable.

        Returns:
            The group to which the variable belongs.
        """
        return self._groups.get(variable_name)

    @property
    def variables(self):  # type: (...) -> List[str]
        """The sorted names of the variables."""
        return sorted(self._groups.keys())

    @property
    def groups(self):  # type: (...) -> List[str]
        """The sorted names of the groups of variables."""
        return sorted(self._names.keys())

    def __str__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add(self.name)
        msg.indent()
        msg.add("Number of samples: {}", self.n_samples)
        msg.add("Number of variables: {}", self.n_variables)
        msg.add("Variables names and sizes by group:")
        msg.indent()
        for group, varnames in sorted(self._names.items()):
            varnames = ["{} ({})".format(name, self.sizes[name]) for name in varnames]
            if varnames:
                msg.add("{}: {}", group, pretty_repr(varnames))
        total = sum(self.dimension.values())
        msg.dedent()
        msg.add("Number of dimensions (total = {}) by group:", total)
        msg.indent()
        for group, size in sorted(self.dimension.items()):
            msg.add("{}: {}", group, size)
        return str(msg)

    def __check_new_variable(
        self,
        variable,  # type: str
    ):  # type: (...) -> None
        """Check if a variable is defined.

        Args:
            variable: The name of the variable.
        """
        if self.is_variable(variable):
            raise ValueError("{} is already defined.".format(variable))
        if not isinstance(variable, string_types):
            raise TypeError("{} is not a string.".format(variable))

    def __check_new_group(self, group):
        """Check if a group is defined.

        Args:
            group: The name of the group.
        """
        if self.is_group(group):
            raise ValueError("{} is already defined.".format(group))
        if not isinstance(group, string_types):
            raise TypeError("{} is not a string.".format(group))

    def __check_length_consistency(
        self,
        length,  # type: int
    ):  # type: (...) -> None
        """Check if a length is consistent with the length of the dataset and set it.

        Args:
            length: A length to be tested.

        Raises:
            ValueError: If the tested length is different from the dataset one.
        """
        if self.length != 0 and length != self.length:
            raise ValueError(
                "The number of rows of data must be equal to the"
                " length of the dataset."
            )
        self.length = length

    def __check_data_consistency(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> None
        """Check that a data array is consistent.

        It must me a 2D numpy array with length equal to the dataset one.

        Raises:
            ValueError: If the data is not a 2D numpy array.
        """
        if not isinstance(data, ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy array.")
        self.__check_length_consistency(data.shape[0])

    @staticmethod
    def __check_variables_format(
        variables,  # type: List[str]
    ):  # type: (...) -> None
        """Check that the names of the variables are well formatted.

        Args:
            variables: The names of the variables.
        """
        if not isinstance(variables, list):
            raise TypeError("variables must be a list of strings.")
        if any([not isinstance(name, string_types) for name in variables]):
            raise TypeError("variables must be a list of strings.")

    @staticmethod
    def __check_sizes_format(
        sizes,  # type: Dict[str,int]
        variables,  # type: Iterable[int],
        dimension,  # type: int
    ):  # type:(...) -> None
        """Check that the sizes of the variables are well specified.

        Args:
            sizes: The sizes of the variables.
            variables: The names of the variables.
            dimension: The data dimension.
        """

        def is_size(size):
            return isinstance(size, (int, long)) and size > 0

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

    def __check_variables_sizes(
        self,
        variables,  # type: List[str]
        sizes,  # type: Dict[str,int]
        dimension,  # type: int
    ):  # type: (...) -> None
        """Check that the variables are well formatted.

        Args:
            sizes: The sizes of the variables.
            variables: The names of the variables.
            dimension: The data dimension.
        """
        if variables is not None:
            self.__check_variables_format(variables)
            if sizes is not None:
                self.__check_sizes_format(sizes, variables, dimension)

    def __get_default_group_variables(
        self,
        group,  # type: str
        dimension,  # type: int
        pattern=None,  # type: Optional[str]
    ):  # type: (...) -> Tuple[List[str], Dict[str,int],Dict[str,str]]
        """Create default names of the variables of a group.

        Args:
            group: The name of the group.
            dimension: The dimension of the group.
            pattern: A pattern to be used for the default name,
                e.g. 'x' will lead to 'x_0', 'x_1', ...
                If None,
                use :attr:`.Dataset.DEFAULT_NAMES`
                or the name of the group.

        Returns:
            The names, the sizes and the groups of the variables.
        """
        pattern = pattern or self.DEFAULT_NAMES.get(group) or group
        variables = ["{}_{}".format(pattern, index) for index in range(dimension)]
        sizes = {name: 1 for name in variables}
        groups = {name: group for name in variables}
        return variables, sizes, groups

    def __set_group_data(
        self,
        data,  # type: ndarray
        group,  # type: str
        variables,  # type: Iterable[str]
        sizes,  # type: Dict[str,int]
    ):  # type: (...) -> None
        """Set the data related to a group.

        Args:
            data: The data.
            group: The name of the group.
            variables: The names of the variables.
            sizes: the sizes of the variables.
        """
        if self._group:
            self.data[group] = data
        else:
            array_to_dict = DataConversion.array_to_dict
            self.data.update(array_to_dict(data, variables, sizes))

    def __set_variable_data(
        self,
        name,  # type: str
        data,  # type: ndarray
        group,  # type: str
    ):  # type: (...) -> None
        """Set the data related to a variable.

        Args:
            name: The name of the variable.
            data: The data.
            group: The name of the group.
        """
        if self._group:
            if not self.is_group(group):
                self.data[group] = data
            else:
                self.data[group] = hstack((self.data[group], data))
        else:
            self.data[name] = data

    def __set_group_properties(
        self,
        group,  # type: str
        variables,  # type: List[str]
        sizes,  # type: Dict[str,int],
        cache_as_input,  # type: bool
    ):  # type: (...) -> None
        """Set the properties related to a group.

        Args:
            group: The name of the group.
            variables: The names of the variables.
            sizes: The sizes of the variables.
            cache_as_input: If True,
                cache these data as inputs
                when the cache is exported to a cache.
        """
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

    def __set_variable_properties(
        self,
        variable,  # type: str
        group,  # type: str
        size,  # type:int
        cache_as_input,  # type: bool
    ):  # type: (...) -> None
        """Set the properties related to a variable.

        Args:
            variable: The name of the variable.
            group: The name of the group.
            size: The size of the variable.
            cache_as_input: If True,
                cache these data as inputs
                when the cache is exported to a cache.
        """
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
        self,
        group,  # type: str
        data,  # type: ndarray
        variables=None,  # type: Optional[List[str]]
        sizes=None,  # type: Optional[Dict[str,int]]
        pattern=None,  # type: Optional[str]
        cache_as_input=True,  # type: bool
    ):  # type: (...) -> str
        """Add data related to a group.

        Args:
            group: The name of the group of data to be added.
            data: The data to be added.
            variables: The names of the variables.
                If None, use default names based on a pattern.
            sizes: The sizes of the variables.
                If None,
                assume that all the variables have a size equal to 1.
            pattern: The name of the variable to be used as a pattern
                when variables is None.
                If None,
                use the :attr:`.Dataset.DEFAULT_NAMES` for this group if it exists.
                Otherwise, use the group name.
            cache_as_input: If True,
                cache these data as inputs
                when the cache is exported to a cache.
        """
        self.__check_new_group(group)
        self.__check_data_consistency(data)
        self.__check_variables_sizes(variables, sizes, data.shape[1])
        if variables is None or sizes is None:
            variables, sizes, _ = self.__get_default_group_variables(
                group, data.shape[1], pattern
            )
        self.__set_group_data(data, group, variables, sizes)
        self.__set_group_properties(group, variables, sizes, cache_as_input)

    def add_variable(
        self,
        name,  # type: str
        data,  # type: ndarray
        group=DEFAULT_GROUP,  # type: str
        cache_as_input=True,  # type: bool
    ):  # type:(...) -> None
        """Add data related to a variable.

        Args:
            name: The name of the variable to be stored.
            data: The data to be stored.
            group: The name of the group related to this variable.
            cache_as_input: If True,
                cache these data as inputs
                when the cache is exported to a cache.
        """
        self.__check_new_variable(name)
        self.__check_data_consistency(data)
        self.__set_variable_data(name, data, group)
        self.__set_variable_properties(name, group, data.shape[1], cache_as_input)

    def __get_strings_encoding(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> Tuple[ndarray,Dict[str,Dict[int,int]]]
        """Encode string data.

        Args:
            data: The data to be encoded.

        Returns:
            The encoded data and the encoding structure.
        """
        self.strings_encoding = {name: {} for name in self._groups}
        encoding = {}
        if str(data.dtype).startswith(("|S", "<U")):
            data, encoding = self.__force_array_to_float(data)
        return data, encoding

    def set_from_array(
        self,
        data,  # type: ndarray
        variables=None,  # type: Optional[List[str]]
        sizes=None,  # type: Optional[Dict[str,int]]
        groups=None,  # type: Optional[Dict[str,str]]
        default_name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """Set the dataset from an array.

        Args:
            data: The data to be stored.
            variables: The names of the variables.
                If None,
                use one default name per column of the array
                based on the pattern 'default_name'.
            sizes: The sizes of the variables.
                If None,
                assume that all the variables have a size equal to 1.
            groups: The groups of the variables.
                If None,
                use :attr:`.Dataset.DEFAULT_GROUP` for all the variables.
            default_name: The name of the variable to be used as a pattern
                when variables is None.
                If None,
                use the :attr:`.Dataset.DEFAULT_NAMES` for this group if it exists.
                Otherwise, use the group name.
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

    def __check_groups_format(
        self,
        groups,  # type: Dict[str,str]
        variables,  # type: Iterable[str]
    ):  # type: (...) -> None
        """Check the format of groups and update it if necessary.

        Args:
            groups: The names of the groups of the variables.
            variables: The names of the variables.
        """
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

    def __set_data(
        self,
        data,  # type: ndarray
        variables,  # type: Iterable[str]
        encoding,  # type: Dict[str,Dict[int,int]]
    ):  # type: (...) -> None
        """Set data.

        Args:
            data: The data to be stored.
            variables: The names of the variables.
            encoding: An encoding structure
                mapping the values of the string variables with integers;
                the keys are the names of the variables
                and the values are dictionaries
                whose keys are the components of the variables
                and the values are the integer values.
        """
        indices = {group: [] for group in self._names}
        start = 0
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
        """Set variables positions."""
        for varnames in self._names.values():
            start = 0
            for varname in varnames:
                self._positions[varname] = [start, start + self.sizes[varname] - 1]
                if self._group:
                    start += self.sizes[varname]
                else:
                    start = 0

    def __set_data_properties(
        self,
        variables,  # type: Iterable[str]
        sizes,  # type: Mapping[str,int]
        groups,  # type: Mapping[str,str]
    ):  # type: (...) -> None
        """Set the properties for the whole data.

        Args:
            variables: The names of the variables.
            sizes: The sizes of the variables.
            groups: The groups of the variables.
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
    def __force_array_to_float(
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Force a ndarray type to float.

        Args:
            data: The data to be casted.

        Returns:
            The data with float type.
        """

        def __is_not_float(
            obj,  # type: Any
        ):  # type: (...) -> bool
            """Return True if an object cannot be cast to float.

            Args:
                obj: The object to test.

            Returns:
                Whether the object can be converted to float.
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
        filename,  # type: str
        variables=None,  # type: Optional[List[str]]
        sizes=None,  # type: Optional[Dict[str,int]]
        groups=None,  # type: Optional[Dict[str,str]]
        delimiter=",",  # type:str
        header=True,  # type: bool
    ):  # type: (...) -> None
        """Set the dataset from a file.

        Args:
            filename: The name of the file containing the data.
            variables: The names of the variables.
                If None and `header` is True,
                read the names from the first line of the file.
                If None and `header` is False,
                use default names
                based on the patterns the :attr:`.Dataset.DEFAULT_NAMES`
                associated with the different groups.
            sizes: The sizes of the variables.
                If None,
                assume that all the variables have a size equal to 1.
            groups: The groups of the variables.
                If None,
                use :attr:`.Dataset.DEFAULT_GROUP` for all the variables.
            delimiter: The field delimiter.
            header: If True,
                read the names of the variables on the first line of the file.
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

    def set_metadata(
        self,
        name,  # type: str
        value,  # type: Any
    ):  # type: (...) -> None
        """Set a metadata attribute.

        Args:
            name: The name of the metadata attribute.
            value: The value of the metadata attribute.
        """
        self.metadata[name] = value

    @property
    def columns_names(self):  # type: (...) -> List[Union[str,Tuple[str,str,str]]]
        """The names of the columns of the dataset."""
        return self._get_columns_names()

    def _get_columns_names(
        self,
        as_list=False,  # type: bool
        start=0,  # type: int
    ):  # type: (...) -> List[Union[str,Tuple[str,str,str]]]
        """Return the names of the columns of the dataset.

        If dim(x)=1,
        its column name is 'x'
        while if dim(y)=2,
        its columns names are either 'x_0' and 'x_1'
        or [group_name, 'x', '0'] and [group_name, 'x', '1'].

        Args:
            as_list: If True, return the name as a tuple.
                otherwise, return the name as a string.
            start: The first index for the components of a variable.
                E.g. with '0': 'x_0', 'x_1', ...

        Returns:
            The names of the columns of the data.
        """
        column_names = []
        for group, names in self._names.items():
            for name in names:
                if as_list:
                    column_names += [
                        [group, name, str(index + start)]
                        for index in range(self.sizes[name])
                    ]
                else:
                    if self.sizes[name] == 1:
                        column_names += [name]
                    else:
                        column_names += [
                            "{}_{}".format(name, index + start)
                            for index in range(self.sizes[name])
                        ]
        return column_names

    def get_data_by_group(
        self,
        group,  # type: str
        as_dict=False,  # type: bool
    ):  # type: (...) -> Union[ndarray, Dict[str,ndarray]]
        """Get the data for a specific group name.

        Args:
            group: The name of the group.
            as_dict: If True, return values as dictionary.

        Returns:
            The data related to the group.
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

    def get_data_by_names(
        self,
        names,  # type: Union[str,Iterable[str]]
        as_dict=True,  # type: bool
    ):  # type: (...) -> Union[ndarray, Dict[str,ndarray]]
        """Get the data for specific names of variables.

        Args:
            names: The names of the variables.
            as_dict: If True, return values as dictionary.

        Returns:
            The data related to the variables.
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

    def get_all_data(self, by_group=True, as_dict=False):  # type: (...) -> AllDataType
        """Get all the data stored in the dataset.

        The data can be returned
        either as a dictionary indexed by the names of the variables,
        or as an array concatenating them,
        accompanied with the names and sizes of the variables.

        The data can also classified by groups of variables.

        Args:
            by_group: If True, sort the data by group.
            as_dict: If True, return the data as a dictionary.

        Returns:
            All the data stored in the dataset.
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
    def n_variables(self):  # type: (...) -> int
        """The number of variables."""
        return len(self._groups)

    def n_variables_by_group(
        self,
        group,  # type: str
    ):  # type: (...) -> int
        """The number of variables for a group.

        Args:
            group: The name of a group.

        Returns:
            The group dimension.
        """
        return len(self._names[group])

    @property
    def n_samples(self):  # type: (...) -> int
        """The number of samples."""
        return self.length

    def __len__(self):  # type: (...) -> int
        """The length of the dataset."""
        return self.length

    def __bool__(self):  # type: (...) -> bool
        """True is the dataset is not empty."""
        return not self.is_empty()

    def export_to_dataframe(
        self,
        copy=True,  # type: bool
    ):  # type: (...) -> DataFrame
        """Export the dataset to a pandas Dataframe.

        Args:
            copy: If True, copy data.
                Otherwise, use reference.

        Returns:
            A pandas DataFrame containing the dataset.
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
        dataframe = DataFrame(data[0], columns=columns, copy=copy)
        dataframe.index = self.row_names
        return dataframe

    def export_to_cache(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
        outputs=None,  # type: Optional[Iterable[str]]
        cache_type=MEMORY_FULL_CACHE,  # type: str
        cache_hdf_file=None,  # type: Optional[str]
        cache_hdf_node_name=None,  # type: Optional[str]
        **options
    ):  # type: (...) -> AbstractFullCache
        """Export the dataset to a cache.

        Args:
            inputs: The names of the inputs to cache.
                If None, use all inputs.
            outputs: The names of the outputs to cache.
                If None, use all outputs.
            cache_type: The type of cache to use.
            cache_hdf_file: The name of the HDF file to store the data.
                Required if the type of the cache is 'HDF5Cache'.
            cache_hdf_node_name: The name of the HDF node to store the discipline.
                If None, use the name of the dataset.

        Returns:
            A cache containing the dataset.
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
                name=self.name,
                **options
            )
        else:
            cache = create_cache(cache_type, name=self.name, **options)
        for sample in range(len(self)):
            in_values = {name: self[(sample, name)][name] for name in inputs}
            out_values = {name: self[(sample, name)][name] for name in outputs}
            cache.cache_outputs(in_values, inputs, out_values, outputs)
        return cache

    def get_available_plots(self):  # type: (...) -> List[str]
        """Return the available plot methods."""
        return DatasetPlotFactory().plots

    def plot(
        self,
        name,  # type:str
        show=True,  # type:bool
        save=False,  # type:bool
        **options
    ):  # type: (...) -> None
        """Plot the dataset from a :class:`DatasetPlot`.

        See :meth:`.Dataset.get_available_plots`

        Args:
            name: The name of the post-processing,
                which is the name of a class inheriting from :class:`DatasetPlot`.
            show: If True, display the figure.
            save: If True, save the figure.
            options: The options for the post-processing.
        """
        options.update({"show": show, "save": save})
        post = DatasetPlotFactory().create(name, dataset=self)
        post.execute(**options)
        return post

    def __getitem(
        self,
        indices,  # type: Sequence[int],
        names,  # type: Iterable[str],
    ):  # type: (...) -> Dict[str,ndarray]
        """Get the items associated with sample indices and variables names.

        Args:
            indices: The samples indices.

        Returns:
            The data related to the samples and variables names.
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
                raise ValueError("'{}' is not a variable name.".format(name))
        item = self.get_data_by_names(names)
        if len(indices) == 1:
            indices = indices[0]
        item = {name: value[indices, :] for name, value in list(item.items())}
        return item

    def __getitem__(
        self,
        key,  # type: ItemType
    ):  # type: (...) -> Dict[str,ndarray]
        indices = list(range(0, self.length))
        type_error_msg = (
            "You can get items from a dataset in one of the following ways: "
            "dataset[3] for the 4th sample, "
            "dataset['x'] for the variable 'x', "
            "dataset[['x','y']] for the variables 'x' and 'y', "
            "dataset[[0,3]] for the 1st and 4th samples, "
            "dataset[(1,'x')] for the variable 'x' of the 2nd sample, "
            "dataset[(1,['x','y'])] for the variables 'x' and 'y' of the 2nd sample, "
            "dataset[([0,3],'x')] for the variable 'x' of the 1st and 4th samples, "
            "dataset[([0,3],['x','y'])] for the variables 'x' and 'y' "
            "of the 1st and 4th samples, "
        )

        def getitem_list(index):
            """Get the item when the index is a list."""
            if all(isinstance(elem, string_types) for elem in index):
                item = self.__getitem(indices, index)
            elif all(isinstance(elem, Number) for elem in index):
                item = self.__getitem(index, self.variables)
            else:
                raise TypeError(type_error_msg)
            return item

        def getitem_tuple(tpl):
            """Get the item when the index is a tuple."""
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

    @property
    def row_names(self):  # type: (...) -> List[str]
        """The names of the rows."""

        return self.__row_names or [str(val) for val in range(len(self))]

    @row_names.setter
    def row_names(
        self,
        names,  # type: List[str]
    ):  # type: (...) -> None
        self.__row_names = names

    def get_normalized_dataset(
        self,
        excluded_variables=None,  # type: Optional[Sequence[str]]
        excluded_groups=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> Dataset
        """Get a normalized copy of the dataset.

        Args:
            excluded_variables: The names of the variables not to be normalized.
                If None, normalize all the variables.
            excluded_groups: The names of the groups not to be normalized.
                If None, normalize all the groups.

        Returns:
            A normalized dataset.
        """
        excluded_groups = excluded_groups or []
        excluded_variables = excluded_variables or []

        dataset = Dataset(self.name, self._group)
        for group, names in self._names.items():
            normalize_group = group not in excluded_groups

            for name in names:
                normalize_name = name not in excluded_variables
                data = self.get_data_by_names(name, False)
                if normalize_group and normalize_name:
                    data = (data - np.min(data, 0)) / (
                        np.max(data, 0) - np.min(data, 0)
                    )

                dataset.add_variable(name, data, group, name in self._cached_inputs)

        return dataset
