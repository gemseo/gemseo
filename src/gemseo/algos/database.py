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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)

"""
A database of function calls and design variables
*************************************************
"""

from __future__ import division, unicode_literals

import logging
from ast import literal_eval
from hashlib import sha1
from itertools import chain, islice
from typing import (
    Any,
    ItemsView,
    Iterable,
    KeysView,
    List,
    Mapping,
    Optional,
    Set,
    Tuple,
    Union,
    ValuesView,
)
from xml.etree.ElementTree import parse as parse_element

import h5py
from numpy import (
    array,
    atleast_2d,
    concatenate,
    float64,
    isclose,
    ndarray,
    string_,
    uint8,
)
from numpy.linalg import norm
from six import string_types

from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.py23_compat import OrderedDict

LOGGER = logging.getLogger(__name__)

# Type of the values associated to the keys (values of input variables) in the database
DatabaseValueType = Mapping[str, Union[float, ndarray, List[int]]]


class Database(object):
    """Class to store evaluations of functions, such as DOE or optimization histories.

    Avoid multiple calls of the same functions,
    useful when simulations are costly.

    It is also used to store inputs and retrieve them
    for optimization graphical post processing and plots
    generation.

    Can be serialized to HDF5 for portability and cold post processing.

    The database is based on a two-levels dictionary-like mapping such as
    ``{key_level_1: {key_level_2: value_level_2} }`` with:
        * ``key_level_1``: the values of the input design variables that have been used
          during the evaluations;
        * ``key_level_2``: the name of the output functions that have been returned,
          the name of the gradient
          (the gradient of a function called ``func`` is typically denoted as ``@func``),
          the iteration numbers which correspond to the evaluation
          (this key is typically denoted ``Iter``)
          and any additional information related to the methods which use the database
          (penalization parameter, trust region radii...);
        * ``value_level_2``: depending on the ``key_level_2``, but is typically a float
          for an output function, an array for a gradient or a list for
          the iterations (when several iterations are stored for a same point,
          this means that the point has been replicated during the process at
          these iterations).

    Attributes:
        name (str): The name of the database.
    """

    missing_value_tag = "NA"
    KEYSSEPARATOR = "__KEYSSEPARATOR__"
    GRAD_TAG = "@"
    ITER_TAG = "Iter"

    def __init__(
        self,
        input_hdf_file=None,  # type: Optional[str]
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            input_hdf_file: The path to a HDF5 file from which the database is created.
                If None, do not import a database.
            name: The name to be given to the database.
                If None, use the class name.
        """
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.__dict = OrderedDict()
        self.__max_iteration = 0
        # Call functions when store is called

        if input_hdf_file is not None:
            self.import_hdf(input_hdf_file)

    def __setitem__(
        self,
        key,  # type: Union[ndarray, HashableNdarray]
        value,  # type: DatabaseValueType
    ):  # type: (...) -> None
        """Set an item to the database.

        Args:
            key: The key of the item (values of input variables).
            value: The value of the item (output functions).

        Raises:
            TypeError:
                * If the key is neither an array, nor a hashable array.
                * If the value is not a dictionary.
        """
        if not isinstance(key, (ndarray, HashableNdarray)):
            raise TypeError(
                "Optimization history keys must be design variables numpy arrays."
            )
        if not isinstance(value, dict):
            raise TypeError("Optimization history values must be data dictionary.")
        if isinstance(key, HashableNdarray):
            self.__dict[key] = value
        else:
            self.__dict[HashableNdarray(key, True)] = value

    @staticmethod
    def __get_hashed_key(
        x_vect,  # type: Union[ndarray, HashableNdarray]
    ):  # type: (...) -> HashableNdarray
        """Convert an array to a hashable array.

        This array basically represent a key of the first level of the database.

        Args:
            x_vect: An array.

        Returns:
            The input array converted to a hashable array.

        Raises:
            TypeError: If the input is not an array.
        """
        if not isinstance(x_vect, (ndarray, HashableNdarray)):
            raise TypeError("Database keys must have ndarray type.")
        if isinstance(x_vect, ndarray):
            return HashableNdarray(x_vect)
        return x_vect

    def __getitem__(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> DatabaseValueType
        """Get an item value from the database.

        Args:
            x_vect: The key of the item.

        Returns:
            The value of the item.
        """
        hashed = self.__get_hashed_key(x_vect)
        return self.__dict[hashed]

    def __delitem__(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> None
        """Delete an item from the database.

        Args:
            x_vect: The key of the item that must be deleted.
        """
        hashed = self.__get_hashed_key(x_vect)
        del self.__dict[hashed]

    def setdefault(
        self,
        key,  # type: ndarray
        default,  # type: DatabaseValueType
    ):  # type: (...) -> DatabaseValueType
        """Set a default database entry.

        Args:
            key: The key of the default item.
            default: The value of the item.

        Returns:
            The value which is set.

        Raises:
            TypeError:
                * If the key is not ndarray type.
                * If the value is not dictionary type.
        """
        if not isinstance(key, (ndarray, HashableNdarray)):
            raise TypeError("Database keys must have ndarray type.")
        if not isinstance(default, dict):
            raise TypeError("Database values must have dictionary type")

        if isinstance(key, ndarray):
            return self.__dict.setdefault(HashableNdarray(key), default)

        return self.__dict.setdefault(key, default)

    def __len__(self):  # type: (...) -> int
        """Get the length of the database.

        Returns:
            The length.
        """
        return len(self.__dict)

    def keys(self):  # type: (...) -> KeysView[ndarray]
        """Database keys generator.

        Yields:
            The next key in the database.
        """
        return self.__dict.keys()

    def values(self):  # type: (...) -> ValuesView[DatabaseValueType]
        """Database values generator.

        Yields:
            The next value in the database.
        """
        return self.__dict.values()

    def items(self):  # type: (...) -> ItemsView[ndarray, DatabaseValueType]
        """Database items generator.

        Yields:
            The next key and value in the database.
        """
        return self.__dict.items()

    def get_value(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> DatabaseValueType
        """Return a value in the database.

        Args:
            x_vect: The key associated to the value.

        Returns:
            The values that correspond to the required key.

        Raises:
            KeyError: If the key does not exist.
        """
        return self[x_vect]

    def get_max_iteration(self):  # type: (...) -> int
        """Return the maximum number of iterations.

        Returns:
            The max number of iterations.
        """
        return self.__max_iteration

    def get_x_history(self):  # type: (...) -> List[ndarray]
        """Return the history of the input design variables ordered by calls.

        Returns:
            This values of input design variables.
        """
        return [x_vect.unwrap() for x_vect in self.__dict.keys()]

    def get_last_n_x(
        self, n  # type: int
    ):  # type: (...) -> List[ndarray]
        """Return the last n ordered calls of the input design variables.

        Args:
            n: The number of last returned calls.

        Returns:
            The values of the input design variables for the last n calls.

        Raises:
            ValueError: If the number n is higher than the size of the database.
        """
        n_max = len(self)
        if n > n_max:
            raise ValueError(
                "The number n = {} must be lower than "
                "the database size = {}".format(n, n_max)
            )
        return [
            x_vect.unwrap() for x_vect in islice(self.__dict.keys(), n_max - n, n_max)
        ]

    def get_index_of(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> int
        """Return the index of an input values in the database.

        Args:
            x_vect: The input values.

        Returns:
            The index of the input values in the database.

        Raises:
            KeyError: If the required key is not found.
        """
        hashed = HashableNdarray(x_vect)
        for i, key in enumerate(self.__dict.keys()):
            if key == hashed:
                return i
        raise KeyError(x_vect)

    def get_x_by_iter(
        self, iteration  # type: int
    ):  # type: (...) -> ndarray
        """Return the values of the input design variables at a specified iteration.

        Args:
            iteration: The iteration required.

        Returns:
            The values of the input design variables.

        Raises:
            ValueError:
                * If the database is empty.
                * If the required iteration is higher than the maximum
                number of iterations in the database.
        """
        nkeys = len(self.__dict)
        if nkeys == 0:
            raise ValueError("The database is empty!")
        if iteration < 0:
            iteration = nkeys + iteration
        if iteration >= nkeys or (iteration < 0 and -iteration > nkeys):
            raise ValueError(
                "Iteration required should be lower "
                "than the maximum number of iterations = {} "
                "got instead = {}".format(len(self) - 1, iteration)
            )
        for i, key in enumerate(self.__dict.keys()):
            if i == iteration:
                return key.unwrap()

    def clear(self):  # type: (...) -> None
        """Clear the database."""
        self.__dict.clear()

    def clean_from_iterate(
        self, iterate  # type: int
    ):  # type: (...) -> None
        """Delete the iterates after a given iteration number.

        Args:
            iterate: The iteration number.
        """

        def gen_todel():
            for iterate_number, x_vect in enumerate(self.__dict.keys()):
                # remove iterations beyond limit iterate number
                if iterate < iterate_number:
                    yield x_vect

        # Copies only the keys after iterate
        to_del = list(gen_todel())
        for key in to_del:
            del self.__dict[key]
        self.__max_iteration = len(self)

    def remove_empty_entries(self):  # type: (...) -> None
        """Remove items when the key is associated to an empty value."""
        empt = [
            k
            for k, v in self.items()
            if len(v) == 0 or (len(v) == 1 and list(v.keys())[0] == self.ITER_TAG)
        ]
        for k in empt:
            del self[k]

    def filter(
        self, data_list_to_keep  # type: Iterable[str]
    ):  # type: (...) -> None
        """Filter the database so that only the required output functions are kept.

        Args:
            data_list_to_keep: The name of output functions that must be kept.
        """
        data_list_to_keep = set(data_list_to_keep)
        for val in self.values():
            keys_to_del = set(val.keys()) - data_list_to_keep
            for key in keys_to_del:
                del val[key]

    def get_func_history(
        self,
        funcname,  # type: str
        x_hist=False,  # type: bool
    ):  # type: (...) -> Union[ndarray, Tuple[ndarray, List[ndarray]]]
        """Return the history of the output function values.

        This function can also return the history of the input values.

        Args:
            funcname: The name of the function.
            x_hist: Whether the input values history is also returned.

        Returns:
            The function values history.
            The input values history if required.
        """
        outf_l = []
        x_history = []
        for x_vect, out_val in self.items():
            val = out_val.get(funcname)
            if val is not None:
                if isinstance(val, ndarray) and val.size == 1:
                    val = val[0]
                outf_l.append(val)
                if x_hist:
                    x_history.append(x_vect.unwrap())
        outf = array(outf_l)
        if x_hist:
            return outf, x_history

        return outf

    def get_func_grad_history(
        self,
        funcname,  # type: str
        x_hist=False,  # type: bool
    ):  # type: (...) -> Union[ndarray, Tuple[ndarray, List[ndarray]]]
        """Return the history of the gradient values of any function.

        The function can also return the history of the input values.

        Args:
            funcname: The name of the function.
            x_hist: Whether the input values history is also returned.

        Returns:
            The gradient values history of the function.
            The input values history if required.
        """
        gradient_name = self.get_gradient_name(funcname)
        return self.get_func_history(funcname=gradient_name, x_hist=x_hist)

    def is_func_grad_history_empty(
        self, funcname  # type: str
    ):  # type: (...) -> bool
        """Check if the history of the gradient of any function is empty.

        Args:
            funcname: The name of the function.

        Returns:
            True if the history of the gradient is empty, False otherwise.
        """
        return len(self.get_func_grad_history(funcname, x_hist=False)) == 0

    def contains_x(
        self, x_vect  # type: ndarray
    ):  # type: (...) -> bool
        """Check if the history contains a specific value of input design variables.

        Args:
            x_vect: The input values that is checked.

        Returns:
            True is the input values are in the history, False otherwise.
        """
        return HashableNdarray(x_vect) in self.__dict

    def get_f_of_x(
        self,
        fname,  # type: str
        x_vect,  # type: ndarray
        dist_tol=0.0,  # type: float
    ):  # type: (...) -> Optional[float, ndarray, List[int]]
        """Return the output function values of any input values in the database.

        Args:
            fname: The name of the required output function.
            x_vect: The required input values.
            dist_tol: In the N-dimensional space of input design variables,
                this value is the threshold of the normalized distance
                with respect to the required point (prescribed input values)
                below which any point in the database can be selected.

        Returns:
            The values of the output function corresponding to
            the required input values.
            ``None`` if no point matches.
        """
        if isclose(dist_tol, 0.0, rtol=1e-16):
            vals = self.get(x_vect)
            if vals is not None:
                return vals.get(fname)  # Returns None if not in self
        else:
            for x_key, vals in self.items():
                x_v = x_key.unwrap()
                if norm(x_v - x_vect) <= dist_tol * norm(x_v):
                    return vals.get(fname)
        return None

    def get(
        self,
        x_vect,  # type: ndarray
        default=None,  # type: Any
    ):  # type: (...) -> Any
        """Return the value of the required key if the key is in the dictionary.

        Args:
            x_vect: The required key.
            default: The values that is returned if the key is not found.

        Returns:
            The value corresponding to the key or the prescribed default value
            if the key is not found.

        Raises:
            TypeError: If the type of the required key is neither an array
                nor a hashable array.
        """
        if not isinstance(x_vect, (HashableNdarray, ndarray)):
            raise TypeError(
                "The key must be an ndarray with the values "
                "of the input design variables."
            )
        if isinstance(x_vect, ndarray):
            x_vect = HashableNdarray(x_vect)
        return self.__dict.get(x_vect, default)

    def pop(
        self, key  # type: ndarray
    ):  # type: (...) -> DatabaseValueType
        """Remove the required key from the database and return the corresponding value.

        Args:
            key: The required key.

        Returns:
            The values corresponding to the key.

        Raises:
            KeyError: If the key is not found.
        """
        return self.__dict.pop(key)

    def contains_dataname(
        self,
        data_name,  # type: str
        skip_grad=False,  # type: bool
    ):  # type: (...) -> bool
        """Check if the database has an output function with the required name.

        Args:
            data_name: The required name of the output function.
            skip_grad: True if the name of the gradients are skipped during the search,
                False otherwise.

        Returns:
            Whether the required output function name is in the database.
        """
        return data_name in self.get_all_data_names(skip_grad=skip_grad)

    def store(
        self,
        x_vect,  # type: ndarray
        values_dict,  # type: DatabaseValueType
        add_iter=True,  # type: bool
    ):  # type: (...) -> None
        """Store the output values associated to the input values.

        Args:
            x_vect: The input values.
            values_dict: The output values corresponding to input values.
            add_iter: True if iterations are added to the output values, False otherwise.
        """
        if self.contains_x(x_vect):
            curr_val = self.get_value(x_vect)
            # No new keys = already computed = new iteration
            # otherwise just calls to other functions
            curr_val.update(values_dict)
        elif add_iter:
            self.__max_iteration += 1
            # include the iteration index
            new_values_dict = dict(
                values_dict, **{self.ITER_TAG: [self.__max_iteration]}
            )
            self.__setitem__(x_vect, new_values_dict)
        else:
            self.__max_iteration += 1
            # do not include the iteration index but still update it
            self.__setitem__(x_vect, values_dict)

    def get_all_data_names(
        self,
        skip_grad=True,  # type: bool
        skip_iter=False,  # type: bool
    ):  # type: (...) -> Set[str]
        """Return all the names of the output functions contained in the database.

        Args:
            skip_grad: True if the names of the keys corresponding to the gradients
                are not returned, False otherwise.
            skip_iter: True if the names of the keys corresponding to the iteration
                numbers are not returned, False otherwise.

        Returns:
            The names of the output functions.
        """
        names = set()
        for value in self.__dict.values():
            for key in value.keys():
                if skip_grad and key.startswith(self.GRAD_TAG):
                    continue
                names.add(key)
        if skip_iter and self.ITER_TAG in names:
            names.remove(self.ITER_TAG)
        return sorted(names)

    def _format_history_names(
        self,
        functions,  # type: Iterable[str]
        stacked_data,  # type: Iterable[str]
    ):  # type: (...) -> Tuple[Iterable[str], List[str]]
        """Format the names of the output functions to be displayed in the history.

        Args:
            functions: The names of output functions.
            stacked_data: The names of outputs that are stored as Iterable
                in the database (e.g. iterations) and that must be unstacked.

        Returns:
            The names of the functions.
            The names of the stacked outputs.

        Raises:
            ValueError: If the names of stacked data are not a subset of the names
                of the functions.
        """
        if functions is None:
            functions = self.get_all_data_names()
        if stacked_data is None:
            if self.ITER_TAG in functions:
                stacked_data = [self.ITER_TAG]
            else:
                stacked_data = []
        elif not set(stacked_data).issubset(functions):
            raise ValueError(
                "The names of the data to be unstacked ({}) "
                "must be included in the names of the data "
                "to be returned ({}).".format(stacked_data, functions)
            )
        elif self.ITER_TAG in functions and self.ITER_TAG not in stacked_data:
            stacked_data.append(self.ITER_TAG)
        return functions, stacked_data

    def get_complete_history(
        self,
        functions=None,  # type: Optional[Iterable[str]]
        add_missing_tag=False,  # type: bool
        missing_tag="NA",  # type: str
        all_iterations=False,  # type: bool
        stacked_data=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> Tuple[List[List[Union[float, ndarray]]], List[ndarray]]
        """Return the complete history of the optimization: design variables, functions
        and gradients.

        Args:
            functions: The names of output functions.
            add_missing_tag: If True, the tag specified in ``missing_tag`` is added to
                iterations where data are not available.
            missing_tag: The tag that is written when data are missing.
            all_iterations: If True, the points which are called at several
                iterations will be duplicated in the history (each duplicate
                corresponding to a different calling index); otherwise each point
                will appear only once (with the latest calling index)
            stacked_data: The names of outputs corresponding to data stored as
                ``iterable`` (e.g. iterations, penalization parameters or
                trust region radii).

        Returns:
            The history of the output values.
            The history of the input values.
        """
        functions, stacked_data = self._format_history_names(functions, stacked_data)
        f_history = []
        x_history = []
        for x_vect, out_val in self.items():
            # If duplicates are not to be considered, or if no iteration index
            # is specified, then only one entry (the last one) will be written:
            if not all_iterations or self.ITER_TAG not in out_val:
                first_index = -1
                last_index = -1
            # Otherwise all the entries will be written:
            else:
                first_index = 0
                # N.B. if the list of indexes is empty, then no entry will be
                # written.
                last_index = len(out_val[self.ITER_TAG]) - 1
            # Add an element to the history for each duplicate required:
            for duplicate_ind in range(first_index, last_index + 1):
                out_vals = []
                for funcname in functions:
                    if funcname in out_val:
                        if funcname not in stacked_data:
                            out_vals.append(out_val[funcname])
                        # If the data 'funcname' is stacked and there remains
                        # entries to unstack, then unstack the next entry:
                        elif duplicate_ind < len(out_val[funcname]):
                            val = out_val[funcname][duplicate_ind]
                            out_vals.append(val)
                    elif add_missing_tag:
                        out_vals.append(missing_tag)
                if out_vals:
                    f_history.append(out_vals)
                    x_history.append(x_vect.unwrap())
        return f_history, x_history

    @staticmethod
    def __to_real(
        data,  # type: Union[ndarray, Iterable[complex]]
    ):  # type: (...) -> ndarray
        """Convert complex to real numpy array.

        Args:
            data: The input values.

        Returns:
            The real output values.
        """
        return array(array(data, copy=False).real, dtype=float64)

    def export_hdf(
        self,
        file_path="optimization_history.h5",  # type: str
        append=False,  # type: bool
    ):  # type: (...) -> None
        """Export the optimization database to a hdf file.

        Args:
            file_path: The name of the hdf file.
            append: If True, append the data to the file; False otherwise.
        """
        mode = "a" if append else "w"

        with h5py.File(file_path, mode) as h5file:
            design_vars_grp = h5file.require_group("x")
            keys_group = h5file.require_group("k")
            values_group = h5file.require_group("v")
            iterated = self.items()
            i = 0

            if append and design_vars_grp:
                iterated = islice(iterated, len(design_vars_grp), len(self.__dict))
                i = len(design_vars_grp)

            for key, val in iterated:
                design_vars_grp.create_dataset(str(i), data=key.unwrap())
                keys_data = array(list(val.keys()), dtype=string_)
                locvalues_scalars = []
                argrp = None

                for ind, locval in enumerate(val.values()):
                    if isinstance(locval, (ndarray, list)):
                        if argrp is None:
                            argrp = values_group.require_group("arr_" + str(i))
                        argrp.create_dataset(str(ind), data=self.__to_real(locval))
                    else:
                        locvalues_scalars.append(locval)

                keys_group.create_dataset(str(i), data=keys_data)
                values_group.create_dataset(
                    str(i), data=self.__to_real(locvalues_scalars)
                )
                i += 1

    def import_hdf(
        self, filename="optimization_history.h5"  # type: str
    ):  # type: (...) -> None
        """Import a database from a hdf file.

        Args:
            filename: The path to the HDF5 file.
        """
        with h5py.File(filename, "r") as h5file:
            design_vars_grp = h5file["x"]
            keys_group = h5file["k"]
            values_group = h5file["v"]

            for raw_index in range(len(design_vars_grp)):
                str_index = str(raw_index)
                keys = [k.decode() for k in get_hdf5_group(keys_group, str_index)]

                array_name = "arr_{}".format(str_index)

                if array_name in values_group:
                    argrp = values_group[array_name]
                    vec_dict = {keys[int(k)]: array(v) for k, v in argrp.items()}
                else:
                    vec_dict = {}

                locvalues_scalars = get_hdf5_group(values_group, str_index)
                scalar_keys = (k for k in keys if k not in vec_dict)

                scalar_dict = dict(
                    ((k, v) for k, v in zip(scalar_keys, locvalues_scalars))
                )
                scalar_dict.update(vec_dict)

                self.store(
                    array(design_vars_grp[str_index]), scalar_dict, add_iter=False
                )

    @staticmethod
    def set_dv_names(
        n_dv,  # type: int
    ):  # type: (...) -> List[str]
        """Return the default input variables names.

        Args:
            n_dv: The number of variables.

        Returns:
            The names of the variables.
        """
        return ["x_" + str(i) for i in range(1, n_dv + 1)]

    def _format_design_variables_names(
        self,
        design_variables_names,  # type: Optional[str, Iterable[str]]
        dimension,  # type: int
    ):  # type: (...) -> Union[List[str], Tuple[str]]
        """Format the design variables names to be displayed in the history.

        Args:
            design_variables_names: The names of design variables.
            dimension: The dimension for default names if ``design_variables_names``
                is ``None``.

        Returns:
            The formatted names of the design variables.

        Raises:
            TypeError: If the type of ``design_variables_names`` is finally not iterable.
        """
        if design_variables_names is None:
            design_variables_names = self.set_dv_names(dimension)
        elif isinstance(design_variables_names, string_types):
            design_variables_names = [design_variables_names]
        elif not isinstance(design_variables_names, list) and not isinstance(
            design_variables_names, tuple
        ):
            raise TypeError(
                "The argument design_variables_names must be a list or a tuple whereas "
                "a {} is provided".format(type(design_variables_names))
            )
        return design_variables_names

    def get_history_array(
        self,
        functions=None,  # type: Optional[Iterable[str]]
        design_variables_names=None,  # type: Optional[Union[str, Iterable[str]]]
        add_missing_tag=False,  # type: bool
        missing_tag="NA",  # type: str
        add_dv=True,  # type: bool
        all_iterations=False,  # type: bool
        stacked_data=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> Tuple[ndarray, List[str], Iterable[str]]
        """Return the history of the optimization process.

        Args:
            functions: The names of output functions that must be returned.
            design_variables_names: The names of the input design variables.
            add_missing_tag: If True, add the tag specified in ``missing_tag`` for
                data that are not available.
            missing_tag: The tag that is added for data that are not available.
            add_dv: If True, the input design variables are returned in the history.
            all_iterations: If True, the points which are called at several
                iterations will be duplicated in the history (each duplicate
                corresponding to a different calling index); otherwise each point
                will appear only once (with the latest calling index).
            stacked_data: The names of outputs corresponding to data stored as
                ``iterable`` (e.g. iterations, penalization parameters or
                trust region radii).

        Returns:
            The values of the history.
            The names of the columns corresponding to the values of the history.
            The names of the output functions.
        """
        if functions is None:
            functions = self.get_all_data_names()
        f_history, x_history = self.get_complete_history(
            functions, add_missing_tag, missing_tag, all_iterations, stacked_data
        )
        design_variables_names = self._format_design_variables_names(
            design_variables_names, len(x_history[0])
        )
        flat_vals = []
        fdict = OrderedDict()
        for f_val_i in f_history:
            flat_vals_i = []
            for f_val, f_name in zip(f_val_i, functions):
                if isinstance(f_val, list):
                    f_val = array(f_val)
                if isinstance(f_val, ndarray) and len(f_val) > 1:
                    flat_vals_i = flat_vals_i + f_val.tolist()
                    fdict[f_name] = [
                        f_name + "_" + str(i + 1) for i in range(len(f_val))
                    ]
                else:
                    flat_vals_i.append(f_val)
                    if f_name not in fdict:
                        fdict[f_name] = [f_name]
            flat_vals.append(flat_vals_i)
        flat_names = sorted(list(chain(*fdict.values())))

        x_flat_vals = []
        xdict = OrderedDict()
        for x_val_i in x_history:
            x_flat_vals_i = []
            for x_val, x_name in zip(x_val_i, design_variables_names):
                if isinstance(x_val, ndarray) and len(x_val) > 1:
                    x_flat_vals_i = x_flat_vals_i + x_val.tolist()
                    xdict[x_name] = [
                        x_name + "_" + str(i + 1) for i in range(len(x_val))
                    ]
                else:
                    x_flat_vals_i.append(x_val)
                    if x_name not in xdict:
                        xdict[x_name] = [x_name]
            x_flat_vals.append(x_flat_vals_i)

        x_flat_names = list(chain(*xdict.values()))
        if add_dv:
            variables_names = flat_names + x_flat_names
        else:
            variables_names = flat_names

        f_history = array(flat_vals).real
        x_history = array(x_flat_vals).real
        if add_dv:
            f2d = atleast_2d(f_history)
            x2d = atleast_2d(x_history)
            if f2d.shape[0] == 1:
                f2d = f2d.T
            if x2d.shape[0] == 1:
                x2d = x2d.T
            values_array = concatenate((f2d, x2d), axis=1)
        else:
            values_array = f_history
        return values_array, variables_names, functions

    def export_to_ggobi(
        self,
        functions=None,  # type: Optional[Iterable[str]]
        file_path="opt_hist.xml",  # type: str
        design_variables_names=None,  # type: Optional[Union[str, Iterable[str]]]
    ):  # type: (...) -> None
        """Export the database to a xml file for ggobi tool.

        Args:
            functions: The names of output functions.
            file_path: The path to the xml file.
            design_variables_names: The names of the input design variables.
        """
        values_array, variables_names, functions = self.get_history_array(
            functions, design_variables_names, add_missing_tag=True, missing_tag="NA"
        )
        LOGGER.info("Export to ggobi for functions: %s", str(functions))
        LOGGER.info("Export to ggobi file: %s", str(file_path))
        save_data_arrays_to_xml(
            variables_names=variables_names,
            values_array=values_array,
            file_path=file_path,
        )

    def import_from_opendace(
        self, database_file  # type: str
    ):  # type: (...) -> None
        """Load the current database from an opendace xml database.

        Args:
            database_file: The path to an opendace database.
        """
        tree = parse_element(database_file)
        for link in tree.getroot().iter("link"):
            data = {}
            for information in link:
                for x_ydyddy in information:
                    data[x_ydyddy.tag] = literal_eval(x_ydyddy.text)
            x_vect = array(data.pop("x"))
            data_reformat = data["y"]
            for key, value in data["dy"].items():
                data_reformat["@" + key[1:]] = array(value)
            self.store(x_vect, data_reformat)

    @classmethod
    def get_gradient_name(
        cls,
        name,  # type: str
    ):  # type: (...) -> str
        """Return the name of the gradient related to a function.

        This name is the concatenation of a GRAD_TAG, e.g. '@',
        and the name of the function, e.g. 'f'.
        With this example, the name of the gradient is '@f'.

        Args:
            name: The name of a function.

        Returns:
            The name of the gradient based on the name of the function.
        """
        return "{}{}".format(cls.GRAD_TAG, name)

    def __str__(self):  # type: (...) -> str
        """Return the string representation.

        The string representation of the database is based on the underlying dictionary
        string representation.
        """
        return str(self.__dict)


class HashableNdarray(object):
    """HashableNdarray wrapper for ndarray objects.

    Instances of ndarray are not HashableNdarray,
    meaning they cannot be added to sets,
    nor used as keys in dictionaries.
    This is by design, ndarray objects are mutable,
    and therefore cannot reliably implement the __hash__() method.

    The HashableNdarray class allows a way around this limitation.
    It implements the required methods for HashableNdarray
    objects in terms of an encapsulated ndarray object.
    This can be either a copied instance (which is safer)
    or the original object
    (which requires the user to be careful enough not to modify it).
    """

    def __init__(
        self,
        wrapped,  # type: ndarray
        tight=False,  # type: bool
    ):  # type: (...) -> None
        """
        Args:
            wrapped: The array that must be wrapped.
            tight: If True, the wrapped array is copied.
        """
        self.__tight = tight
        self.wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(sha1(wrapped.view(uint8)).hexdigest(), 16)

    def __eq__(
        self, other  # type: HashableNdarray
    ):  # type: (...) -> bool
        """Check equality with another HashableNdarray.

        Args:
            other: The other hashable array.

        Returns:
            True if the two arrays are equal, False otherwise.
        """
        return all(self.wrapped == other.wrapped)

    def __hash__(self):  # type: (...) -> int
        """Return the hash number of the current array.

        Returns:
            The hash number.
        """
        return self.__hash

    def unwrap(self):  # type: (...) -> ndarray
        """Return the encapsulated ndarray.

        Returns:
            The encapsulated ndarray, or a copy if the wrapper is ``tight``.
        """
        if self.__tight:
            return array(self.wrapped)

        return self.wrapped

    def __str__(self):  # type: (...) -> str
        """Return the informal string representation of the array.

        Returns:
            The string representation.
        """
        return str(array(self.wrapped))

    def __repr__(self):  # type: (...) -> str
        """Return the official string representation.

        Returns:
            The string representation.
        """
        return str(self)
