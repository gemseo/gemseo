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
"""A database of function calls and design variables."""
from __future__ import annotations

import logging
import sys
from ast import literal_eval
from itertools import chain
from itertools import islice
from numbers import Number
from pathlib import Path
from typing import Any
from typing import Callable
from typing import ItemsView
from typing import Iterable
from typing import KeysView
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Union
from typing import ValuesView
from xml.etree.ElementTree import parse as parse_element

import h5py
from numpy import array
from numpy import array_equal
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import float64
from numpy import hstack
from numpy import ndarray
from numpy import string_
from numpy import uint8
from numpy.linalg import norm
from numpy.typing import NDArray
from xxhash._xxhash import xxh3_64_hexdigest

from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.hdf5 import get_hdf5_group

LOGGER = logging.getLogger(__name__)

# Type of the values associated to the keys (values of input variables) in the database
DatabaseValueType = Mapping[str, Union[float, ndarray, List[int]]]
ReturnedHdfMissingOutputType = Tuple[
    Mapping[str, Union[float, ndarray, List[int]]], Union[None, Mapping[str, int]]
]


class Database:
    """Storage of :class:`.MDOFunction` evaluations.

    A :class:`.Database` is typically attached to an :class:`.OptimizationProblem`
    to store the evaluations of its objective, constraints and observables.

    Then,
    a :class:`.Database` can be an optimization history
    or a collection of samples in the case of a DOE.

    It is useful when simulations are costly
    because it avoids re-evaluating functions
    at points where they have already been evaluated

    .. seealso:: :class:`.NormDBFunction`

    It can also be post-processed by an :class:`.OptPostProcessor`
    to visualize its content,
    e.g. :class:`.OptHistoryView` generating a series of graphs
    to visualize the histories of the objective, constraints and design variables.

    A :class:`.Database` can be serialized to HDF5
    for portability and cold post-processing.

    .. note::
        Serializing an :class:`.OptimizationProblem`
        using its method :class:`~.OptimizationProblem.export_hdf`
        also serializes its :class:`.Database`.

    The database is based on a two-levels dictionary-like mapping such as
    ``{key_level_1: {key_level_2: value_level_2}}`` with:

        * ``key_level_1``: the values of the input design variables that have been used
          during the evaluations,
          if the types of the design variables are different,
          then they are promoted to the unique type that can represent all them,
          for instance integer would be promoted to float;
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
    """

    name: str
    """The name of the database."""

    missing_value_tag = "NA"
    KEYSSEPARATOR = "__KEYSSEPARATOR__"
    GRAD_TAG = "@"
    ITER_TAG = "Iter"

    __component_names_to_names: dict[str, str]
    """The variable names associated with variable component names."""

    def __init__(
        self,
        input_hdf_file: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            input_hdf_file: The path of an HDF file containing an initial database
                if any.
                It should have been generated with :meth:`.Database.export_hdf`.
            name: The name to be given to the database.
                If ``None``, use the class name.
        """  # noqa: D205, D212, D415
        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

        self.__dict = {}
        self.__max_iteration = 0
        self.__store_listeners = []
        self.__newiter_listeners = []

        # This list enables to temporary save the last inputs that have been
        # stored before any export_hdf is called.
        # It is used to append the exported hdf file.
        self.__hdf_export_buffer = []

        if input_hdf_file is not None:
            self.import_hdf(input_hdf_file)

        self.__component_names_to_names = {}

    @property
    def last_item(self) -> DatabaseValueType:
        """The last item stored in the database."""
        if sys.version_info < (3, 8, 0):
            return next(reversed(list(self.__dict.values())))
        else:
            return next(reversed(self.__dict.values()))

    def __setitem__(
        self,
        key: ndarray | HashableNdarray,
        value: DatabaseValueType,
    ) -> None:
        """Set an item to the database.

        Args:
            key: The key of the item (values of input variables).
            value: The value of the item (output functions).

        Raises:
            TypeError:

                * If the key is neither an array, nor a hashable array.
                * If the value is not a dictionary.
        """
        if isinstance(key, HashableNdarray):
            self.__dict[key] = dict(value)
        else:
            if not isinstance(key, ndarray):
                raise TypeError(
                    "Optimization history keys must be design variables numpy arrays."
                )
            self.__dict[HashableNdarray(key, True)] = dict(value)

    def is_new_eval(self, x_vect: ndarray) -> bool:
        """Whether storing the given values would generate a new iteration.

        Args:
            x_vect: The design variables values.

        Returns:
            Whether a new iteration would occur after storing the values.
        """
        curr_val = self.get(x_vect)

        if curr_val is None:
            return True

        n_cval = len(curr_val)
        return (n_cval == 1 and self.ITER_TAG in curr_val) or n_cval == 0

    @staticmethod
    def get_hashed_key(
        x_vect: ndarray | HashableNdarray,
        copy: bool = False,
    ) -> HashableNdarray:
        """Convert an array to a hashable array.

        This array basically represent a key of the first level of the database.

        Args:
            x_vect: An array.
            copy: Whether to copy the original array.

        Returns:
            The input array converted to a hashable array.

        Raises:
            TypeError: If the input is not an array or HashableNdarray.
        """
        if isinstance(x_vect, ndarray):
            return HashableNdarray(x_vect, tight=copy)

        if isinstance(x_vect, HashableNdarray):
            if copy and not x_vect.tight:
                x_vect.wrapped = array(x_vect.wrapped)

            return x_vect

        raise KeyError(f"Invalid key type {type(x_vect)}.")

    def __getitem__(self, x_vect: ndarray) -> DatabaseValueType | None:
        return self.__dict.get(self.get_hashed_key(x_vect))

    def __delitem__(self, x_vect: ndarray) -> None:
        del self.__dict[self.get_hashed_key(x_vect)]

    def setdefault(
        self,
        key: ndarray,
        default: DatabaseValueType,
    ) -> DatabaseValueType:
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

    def __len__(self) -> int:
        return len(self.__dict)

    def keys(self) -> KeysView[ndarray]:
        """Database keys generator.

        Yields:
            The next key in the database.
        """
        return self.__dict.keys()

    def values(self) -> ValuesView[DatabaseValueType]:
        """Database values generator.

        Yields:
            The next value in the database.
        """
        return self.__dict.values()

    def items(self) -> ItemsView[ndarray, DatabaseValueType]:
        """Database items generator.

        Yields:
            The next key and value in the database.
        """
        return self.__dict.items()

    def get_value(self, x_vect: ndarray) -> DatabaseValueType:
        """Return a value in the database.

        Args:
            x_vect: The key associated to the value.

        Returns:
            The values that correspond to the required key.

        Raises:
            KeyError: If the key does not exist.
        """
        return self[x_vect]

    def get_max_iteration(self) -> int:
        """Return the maximum number of iterations.

        Returns:
            The maximum number of iterations.
        """
        return self.__max_iteration

    def get_x_history(self) -> list[ndarray]:
        """Return the history of the input design variables ordered by calls.

        Returns:
            This values of input design variables.
        """
        return [x_vect.wrapped for x_vect in self.__dict.keys()]

    def get_last_n_x(self, n: int) -> list[ndarray]:
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
            x_vect.wrapped for x_vect in islice(self.__dict.keys(), n_max - n, n_max)
        ]

    def get_index_of(self, x_vect: ndarray) -> int:
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

    def get_x_by_iter(self, iteration: int) -> ndarray:
        """Return the values of the input design variables at a specified iteration.

        Args:
            iteration: The required iteration.

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
            raise ValueError("The database is empty.")
        if iteration < 0:
            iteration += nkeys
        if iteration >= nkeys or (iteration < 0 and -iteration > nkeys):
            raise ValueError(
                "Iteration required should be lower "
                "than the maximum number of iterations = {} "
                "got instead = {}".format(len(self) - 1, iteration)
            )

        # The database dictionary uses the input design variables as keys for the
        # function values. Here we convert it to an iterator that returns the
        # key located at the required iteration using the islice method from
        # itertools.
        key = next(islice(iter(self.__dict), iteration, iteration + 1))
        return key.wrapped

    def clear(self, reset_iteration_counter=False) -> None:
        """Clear the database.

        Args:
            reset_iteration_counter: Whether to reset the iteration counter.
        """
        self.__dict.clear()
        if reset_iteration_counter:
            self.__max_iteration = 0

    def clean_from_iterate(self, iterate: int) -> None:
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

    def remove_empty_entries(self) -> None:
        """Remove items when the key is associated to an empty value."""
        empt = [
            k
            for k, v in self.items()
            if len(v) == 0 or (len(v) == 1 and list(v.keys())[0] == self.ITER_TAG)
        ]
        for k in empt:
            del self[k]

    def filter(self, names: Iterable[str]) -> None:
        """Filter the database so that only the required output functions are kept.

        Args:
            names: The names of output functions that must be kept.
        """
        names = set(names)
        for value in self.values():
            for key in set(value.keys()) - names:
                del value[key]

    def get_func_history(
        self,
        funcname: str,
        x_hist: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
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
                    x_history.append(x_vect.wrapped)
        outf = array(outf_l)
        if x_hist:
            return outf, x_history

        return outf

    def get_func_grad_history(
        self,
        funcname: str,
        x_hist: bool = False,
    ) -> ndarray | tuple[ndarray, list[ndarray]]:
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

    def is_func_grad_history_empty(self, funcname: str) -> bool:
        """Check if the history of the gradient of any function is empty.

        Args:
            funcname: The name of the function.

        Returns:
            True if the history of the gradient is empty, False otherwise.
        """
        return len(self.get_func_grad_history(funcname)) == 0

    def contains_x(self, x_vect: ndarray) -> bool:
        """Check if the history contains a specific value of input design variables.

        Args:
            x_vect: The input values that is checked.

        Returns:
            True is the input values are in the history, False otherwise.
        """
        return HashableNdarray(x_vect) in self.__dict

    def get_f_of_x(
        self,
        fname: str,
        x_vect: ndarray,
        dist_tol: float = 0.0,
    ) -> None | float | ndarray | list[int]:
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
        if abs(dist_tol) < sys.float_info.epsilon:
            val = self.get(x_vect)
            if val is not None:
                return val.get(fname)
        else:
            if isinstance(x_vect, HashableNdarray):
                x_vect = x_vect.wrapped
            for x_key, vals in self.items():
                x_v = x_key.wrapped
                if norm(x_v - x_vect) <= dist_tol * norm(x_v):
                    return vals.get(fname)
        return None

    def get(
        self,
        x_vect: ndarray,
        default: Any = None,
    ) -> Any:
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
        if isinstance(x_vect, ndarray):
            x_vect = HashableNdarray(x_vect)
        return self.__dict.get(x_vect, default)

    def pop(self, key: ndarray) -> DatabaseValueType:
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
        data_name: str,
        skip_grad: bool = False,
    ) -> bool:
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
        x_vect: ndarray,
        values_dict: DatabaseValueType,
        add_iter: bool = True,
    ) -> None:
        """Store the output values associated to the input values.

        Args:
            x_vect: The input values.
            values_dict: The output values corresponding to the input values.
            add_iter: True if iterations are added to the output values, False otherwise.
        """
        x_vect_hash = self.get_hashed_key(x_vect, True)

        n_values = len(values_dict)
        values_not_empty = n_values > 1 or (
            n_values == 1 and self.ITER_TAG not in values_dict
        )
        curr_val = self.get(x_vect_hash)
        self.__hdf_export_buffer.append(x_vect_hash)
        if curr_val is None:
            self.__max_iteration += 1
            cval_ok = True
            curr_val = values_dict
            if add_iter:
                curr_val[self.ITER_TAG] = [self.__max_iteration]
            self[x_vect_hash] = curr_val
        else:
            # No new keys = already computed = new iteration
            # otherwise just calls to other functions
            cval_ok = (len(curr_val) == 1 and self.ITER_TAG in curr_val) or not curr_val
            curr_val.update(values_dict)

        if self.__store_listeners:
            self.notify_store_listeners(x_vect)
        # Notify the new iteration after storing x_vect
        # because callbacks may need an updated x_vect
        if self.__newiter_listeners:
            if cval_ok and values_not_empty:
                self.notify_newiter_listeners(x_vect)

    def add_store_listener(self, listener_func: Callable) -> None:
        """Add a listener to be called when an item is stored to the database.

        Args:
            listener_func: The function to be called.

        Raises:
            TypeError: If the argument is not a callable
        """
        if not callable(listener_func):
            raise TypeError("Listener function is not callable")
        self.__store_listeners.append(listener_func)

    def add_new_iter_listener(
        self,
        listener_func: Callable,
    ) -> None:
        """Add a listener to be called when a new iteration is stored to the database.

        Args:
            listener_func: The function to be called, it must have one argument that is
                the current x_vector.

        Raises:
            TypeError: If the argument is not a callable.
        """
        if not callable(listener_func):
            raise TypeError("Listener function is not callable.")
        self.__newiter_listeners.append(listener_func)

    def clear_listeners(self) -> None:
        """Clear all the listeners."""
        self.__store_listeners = []
        self.__newiter_listeners = []

    def notify_store_listeners(
        self,
        x_vect: ndarray | None = None,
    ) -> None:
        """Notify the listeners that a new entry was stored in the database.

        Args:
            x_vect: The values of the design variables. If None, use
                the values of the last iteration.
        """
        if isinstance(x_vect, HashableNdarray):
            x_vect = x_vect.wrapped
        elif x_vect is None:
            x_vect = self.get_x_by_iter(-1)
        for func in self.__store_listeners:
            func(x_vect)

    def notify_newiter_listeners(
        self,
        x_vect: ndarray | None = None,
    ) -> None:
        """Notify the listeners that a new iteration is ongoing.

        Args:
            x_vect: The values of the design variables. If None, use
                the values of the last iteration.
        """
        if not self.__newiter_listeners:
            return
        if isinstance(x_vect, HashableNdarray):
            x_vect = x_vect.wrapped
        elif x_vect is None:
            x_vect = self.get_x_by_iter(-1)

        for func in self.__newiter_listeners:
            func(x_vect)

    def get_all_data_names(
        self,
        skip_grad: bool = True,
        skip_iter: bool = False,
    ) -> set[str]:
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
        functions: Iterable[str],
        stacked_data: Iterable[str],
    ) -> tuple[Iterable[str], list[str]]:
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
        functions: Iterable[str] | None = None,
        add_missing_tag: bool = False,
        missing_tag: str | float = "NA",
        all_iterations: bool = False,
        stacked_data: Iterable[str] | None = None,
    ) -> tuple[list[list[float | ndarray]], list[ndarray]]:
        """Return the complete history of the optimization.

        This includes the design variables, functions and gradients.

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
            The history of the output values,
            then the history of the input values.
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
                    x_history.append(x_vect.wrapped)
        return f_history, x_history

    @staticmethod
    def __to_real(
        data: ndarray | Iterable[complex],
    ) -> ndarray:
        """Convert complex to real numpy array.

        Args:
            data: The input values.

        Returns:
            The real output values.
        """
        return array(array(data, copy=False).real, dtype=float64)

    @staticmethod
    def _add_hdf_input_dataset(
        index_dataset: int,
        design_vars_group: h5py.Group,
        design_vars_values: HashableNdarray,
    ) -> None:
        """Add a new input to the hdf group of input values.

        Args:
            index_dataset: The index of the new hdf5 entry.
            design_vars_group: The hdf5 group of the design variable values.
            design_vars_values: The values of the input.

        Raises:
            ValueError: If the dataset name ``index_dataset`` already exists
                in the group of design variables.
        """
        if str(index_dataset) in design_vars_group:
            raise ValueError(
                "Dataset name '{}' already exists in the group "
                "of design variables.".format(index_dataset)
            )

        design_vars_group.create_dataset(
            str(index_dataset), data=design_vars_values.wrapped
        )

    def _add_hdf_output_dataset(
        self,
        index_dataset: int,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        output_values: Mapping[str, float | ndarray | list],
        output_name_to_idx: Mapping[str, int] | None = None,
    ) -> None:
        """Add new outputs to the hdf group of output values.

        Args:
            index_dataset: The index of the new hdf5 entry.
            keys_group: The hdf5 group of the output names.
            values_group: The hdf5 group of the output values.
            output_values: The output values.
            output_name_to_idx: The indices of the output names in ``output_values``.
                If ``None``, these indices are automatically built using the
                order of the names in ``output_values``.
                These indices are used to build the dataset of output vectors.
        """
        self._add_hdf_name_output(index_dataset, keys_group, list(output_values.keys()))

        if not output_name_to_idx:
            output_name_to_idx = dict(zip(output_values, range(len(output_values))))

        # We separate scalar data from vector data in the hdf file.
        # Scalar data are first stored into a list (``values``),
        # then added to the hdf file.
        # Vector data are directly added to the hdf file.
        values = []
        for name, value in output_values.items():
            idx_value = output_name_to_idx[name]
            if isinstance(value, (ndarray, list)):
                self._add_hdf_vector_output(
                    index_dataset, idx_value, values_group, value
                )
            else:
                values.append(value)

        if values:
            self._add_hdf_scalar_output(index_dataset, values_group, values)

    @staticmethod
    def _get_missing_hdf_output_dataset(
        index_dataset: int,
        keys_group: h5py.Group,
        output_values: Mapping[str, float | ndarray | list[int]],
    ) -> ReturnedHdfMissingOutputType:
        """Return the missing values in the hdf group of the output names.

        Compare the keys of ``output_values`` with the existing names
        in the group of the output names ``keys_group`` in order to know which
        outputs are missing.

        Args:
            index_dataset: The index of the new hdf5 entry.
            keys_group: The hdf5 group of the output names.
            output_values: The output values to be compared with.

        Returns:
            The missing values.
            The indices of the missing outputs.

        Raises:
            ValueError: If the index of the dataset does not correspond to
                an existing dataset.
        """
        name = str(index_dataset)
        if name not in keys_group:
            raise ValueError(f"The dataset named '{name}' does not exist.")

        existing_output_names = {out.decode() for out in keys_group[name]}
        all_output_names = set(output_values)
        missing_names = all_output_names - existing_output_names

        if not missing_names:
            return {}, None

        missing_names_values = {name: output_values[name] for name in missing_names}
        all_output_idx_mapping = dict(zip(output_values, range(len(output_values))))
        missing_names_idx_mapping = {
            name: all_output_idx_mapping[name] for name in missing_names
        }

        return missing_names_values, missing_names_idx_mapping

    @staticmethod
    def _add_hdf_name_output(
        index_dataset: int,
        keys_group: h5py.Group,
        keys: list[str],
    ):
        """Add new output names to the hdf5 group of output names.

        Create a dataset in the group of output names
        if the dataset index is not found in the group.
        If the dataset already exists, the new names are appended
        to the existing dataset.

        Args:
            index_dataset: The index of the new hdf5 entry.
            keys_group: The hdf5 group of the output names.
            keys: The names that must be added.
        """
        name = str(index_dataset)
        keys = array(keys, dtype=string_)
        if name not in keys_group:
            keys_group.create_dataset(
                name, data=keys, maxshape=(None,), dtype=h5py.string_dtype()
            )
        else:
            offset = len(keys_group[name])
            keys_group[name].resize((offset + len(keys),))
            keys_group[name][offset:] = keys

    def _add_hdf_scalar_output(
        self,
        index_dataset: int,
        values_group: h5py.Group,
        values: list[float],
    ) -> None:
        """Add new scalar values to the hdf5 group of output values.

        Create a dataset in the group of output values
        if the dataset index is not found in the group.
        If the dataset already exists, the new values are appended to
        the existing dataset.

        Args:
            index_dataset: The index of the new hdf5 entry.
            values_group: The hdf5 group of the output values.
            values: The scalar values that must be added.
        """
        name = str(index_dataset)
        if name not in values_group:
            values_group.create_dataset(
                name, data=self.__to_real(values), maxshape=(None,), dtype=float64
            )
        else:
            offset = len(values_group[name])
            values_group[name].resize((offset + len(values),))
            values_group[name][offset:] = self.__to_real(values)

    def _add_hdf_vector_output(
        self,
        index_dataset: int,
        idx_sub_group: int,
        values_group: h5py.Group,
        value: ndarray | list[int],
    ) -> None:
        """Add a new vector of values to the hdf5 group of output values.

        Create a subgroup dedicated to vectors in the group of output
        values.
        Inside this subgroup, a new dataset is created for each vector.
        If the subgroup already exists, it is just appended.
        Otherwise, the sub-group is created.

        Args:
            index_dataset: The index of the hdf5 entry.
            idx_sub_group: The index of the dataset in the subgroup of vectors.
            values_group: The hdf5 group of the output values.
            value: The vector which is added to the group.

        Raises:
            ValueError: If the index of the dataset in the subgroup of vectors
                already exist.
        """
        sub_group_name = f"arr_{index_dataset}"

        if sub_group_name not in values_group:
            sub_group = values_group.require_group(sub_group_name)
        else:
            sub_group = values_group[sub_group_name]

        if str(idx_sub_group) in sub_group:
            raise ValueError(
                "Dataset name '{}' already exists in the sub-group of "
                "array output '{}'.".format(idx_sub_group, sub_group_name)
            )

        sub_group.create_dataset(
            str(idx_sub_group), data=self.__to_real(value), dtype=float64
        )

    def _append_hdf_output(
        self,
        index_dataset: int,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        output_values: Mapping[str, float | ndarray | list[int]],
    ) -> None:
        """Append the existing hdf5 datasets of the outputs with new values.

        Find the values among ``output_values`` that do not
        exist in the hdf5 datasets and append them to the datasets.

        Args:
            index_dataset: The index of the existing hdf5 entry.
            keys_group: The hdf5 group of the output names.
            values_group: The hdf5 group of the output values.
            output_values: The output values. Only the values which
                do not exist in the dataset will be appended.
        """
        added_values, mapping_to_idx = self._get_missing_hdf_output_dataset(
            index_dataset, keys_group, output_values
        )
        if added_values:
            self._add_hdf_output_dataset(
                index_dataset,
                keys_group,
                values_group,
                added_values,
                output_name_to_idx=mapping_to_idx,
            )

    def _create_hdf_input_output(
        self,
        index_dataset: int,
        design_vars_group: h5py.Group,
        keys_group: h5py.Group,
        values_group: h5py.Group,
        input_values: HashableNdarray,
        output_values: Mapping[str, float | ndarray | list[int]],
    ):
        """Create the new hdf5 datasets for the given inputs and outputs.

        Useful when exporting the database to a hdf5 file.

        Args:
            index_dataset: The index of the new hdf5 entry.
            design_vars_group: The hdf5 group of the design variable values.
            keys_group: The hdf5 group of the output names.
            values_group: The hdf5 group of the output values.
            input_values: The input values.
            output_values: The output values.
        """
        self._add_hdf_input_dataset(index_dataset, design_vars_group, input_values)
        self._add_hdf_output_dataset(
            index_dataset, keys_group, values_group, output_values
        )

    def export_hdf(
        self,
        file_path: str | Path = "optimization_history.h5",
        append: bool = False,
    ) -> None:
        """Export the optimization database to an HDF file.

        Args:
            file_path: The path of the HDF file.
            append: Whether to append the data to the file.
        """
        with h5py.File(file_path, "a" if append else "w") as h5file:
            design_vars_grp = h5file.require_group("x")
            keys_group = h5file.require_group("k")
            values_group = h5file.require_group("v")
            index_dataset = 0

            # The append mode loops over the last stored entries in order to
            # check whether some new outputs have been added.
            # However, if the hdf file has been re-written by a previous function
            # (such as OptimizationProblem.export_hdf),
            # there is no existing database inside the hdf file.
            # In such case, we have to check whether the design
            # variables group exists because otherwise the function tries to
            # append something empty.
            if append and len(design_vars_grp) != 0:
                input_values_to_idx = dict(zip(self.keys(), range(len(self.keys()))))

                for input_values in self.__hdf_export_buffer:
                    output_values = self[input_values]
                    index_dataset = input_values_to_idx[input_values]

                    if str(index_dataset) in design_vars_grp:
                        self._append_hdf_output(
                            index_dataset, keys_group, values_group, output_values
                        )
                    else:
                        self._create_hdf_input_output(
                            index_dataset,
                            design_vars_grp,
                            keys_group,
                            values_group,
                            input_values,
                            output_values,
                        )
            else:
                for input_values, output_values in self.items():
                    self._create_hdf_input_output(
                        index_dataset,
                        design_vars_grp,
                        keys_group,
                        values_group,
                        input_values,
                        output_values,
                    )
                    index_dataset += 1

        self.__hdf_export_buffer = []

    def import_hdf(self, filename: str | Path = "optimization_history.h5") -> None:
        """Import a database from an HDF file.

        Args:
            filename: The path of the HDF file.
        """
        with h5py.File(filename) as h5file:
            design_vars_grp = h5file["x"]
            keys_group = h5file["k"]
            values_group = h5file["v"]

            for raw_index in range(len(design_vars_grp)):
                str_index = str(raw_index)
                keys = [k.decode() for k in get_hdf5_group(keys_group, str_index)]

                array_name = f"arr_{str_index}"
                if array_name in values_group:
                    names_to_arrays = {
                        keys[int(k)]: array(v)
                        for k, v in values_group[array_name].items()
                    }
                else:
                    names_to_arrays = {}

                if str_index in values_group:
                    scalar_dict = dict(
                        zip(
                            (k for k in keys if k not in names_to_arrays),
                            get_hdf5_group(values_group, str_index),
                        )
                    )
                else:
                    scalar_dict = {}
                scalar_dict.update(names_to_arrays)

                self.store(
                    array(design_vars_grp[str_index]), scalar_dict, add_iter=False
                )

    @staticmethod
    def set_dv_names(
        n_dv: int,
    ) -> list[str]:
        """Return the default input variables names.

        Args:
            n_dv: The number of variables.

        Returns:
            The names of the variables.
        """
        return ["x_" + str(i) for i in range(1, n_dv + 1)]

    def _format_design_variables_names(
        self,
        design_variables_names: None | str | Iterable[str],
        dimension: int,
    ) -> list[str] | tuple[str]:
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
        elif isinstance(design_variables_names, str):
            design_variables_names = [design_variables_names]
        elif not isinstance(design_variables_names, list) and not isinstance(
            design_variables_names, tuple
        ):
            raise TypeError(
                "The argument design_variables_names must be a list or a tuple whereas "
                "a {} is provided".format(type(design_variables_names))
            )
        return design_variables_names

    def __set_variable_component_name(
        self, name: str, component: int, size: int
    ) -> str:
        """Define the name of the component of a variable.

        Args:
            name: The name of the variable.
            component: The component of the variable.
            size: The size of the variable.

        Returns:
            The name of the component of a variable.
        """
        component_name = name if size == 1 else f"{name} ({component})"
        self.__component_names_to_names[component_name] = name
        return component_name

    def retrieve_variable_name(self, name: str) -> str:
        """Retrieve a variable name from a name.

        Args:
            name: A name.

        Returns:
            The name of the variable for which ``name`` is the name of a component,
            or ``name`` otherwise.
        """
        return self.__component_names_to_names.get(name, name)

    def get_history_array(
        self,
        functions: Iterable[str] | None = None,
        design_variables_names: str | Iterable[str] | None = None,
        add_missing_tag: bool = False,
        missing_tag: str | float = "NA",
        add_dv: bool = True,
        all_iterations: bool = False,
        stacked_data: Iterable[str] | None = None,
    ) -> tuple[NDArray[Number | str], list[str], Iterable[str]]:
        """Return the database as a 2D array shaped as ``(n_iterations, n_features)``.

        The features are the outputs of interest and possibly the design variables.

        Args:
            functions: The names of the outputs that must be returned.
            design_variables_names: The names of the design variables.
            add_missing_tag: If ``True``,
                add the tag specified in ``missing_tag``
                for data that are not available.
            missing_tag: The tag that is added for data that are not available.
            add_dv: If ``True``,
                the input design variables are returned in the history.
            all_iterations: If ``True``,
                the points which are called at several iterations will be duplicated
                in the history
                (each duplicate corresponding to a different calling index);
                otherwise each point will appear only once
                (with the latest calling index).
            stacked_data: The names of outputs
                corresponding to data stored as ``Iterable``
                (e.g. iterations, penalization parameters or trust region radii).

        Returns:
            The history as an 2D array
            whose rows are observations and columns are the variables,
            the names of these columns
            and the names of the output functions.
        """
        f_names = functions
        if f_names is None:
            f_names = self.get_all_data_names()

        f_history, x_history = self.get_complete_history(
            f_names, add_missing_tag, missing_tag, all_iterations, stacked_data
        )
        f_flat_names, f_flat_values = self.__split_history(f_history, f_names)
        variables_flat_names = f_flat_names
        f_history = array(f_flat_values).real
        if add_dv:
            x_names = self._format_design_variables_names(
                design_variables_names, len(x_history[0])
            )
            x_flat_names, x_flat_values = self.__split_history(x_history, x_names)
            variables_flat_names = f_flat_names + x_flat_names
            x_history = array(x_flat_values).real
            variables_history = hstack((f_history, x_history))
        else:
            variables_history = f_history

        return atleast_2d(variables_history), variables_flat_names, f_names

    def __split_history(
        self, history: list[list[float | ndarray]] | list[ndarray], names: Iterable[str]
    ) -> tuple[list[str], list[float]]:
        """Split a history.

        Args:
            history: A history of values.
            names: The names of the variables.

        Returns:
            The history as an array whose lines are observations,
            the names of the columns of the array.
        """
        flat_values = []
        names_to_flat_names = {}
        for values in history:
            flat_value = []
            for value, name in zip(values, names):
                value = atleast_1d(value)
                size = value.size
                flat_value.extend(value)
                names_to_flat_names[name] = [
                    self.__set_variable_component_name(name, i, size)
                    for i in range(size)
                ]

            flat_values.append(flat_value)

        return list(chain(*names_to_flat_names.values())), flat_values

    def export_to_ggobi(
        self,
        functions: Iterable[str] | None = None,
        file_path: str | Path = "opt_hist.xml",
        design_variables_names: str | Iterable[str] | None = None,
    ) -> None:
        """Export the database to a xml file for ggobi tool.

        Args:
            functions: The names of output functions.
            file_path: The path to the xml file.
            design_variables_names: The names of the input design variables.
        """
        values_array, variables_names, functions = self.get_history_array(
            functions, design_variables_names, add_missing_tag=True
        )
        LOGGER.info("Export to ggobi for functions: %s", str(functions))
        LOGGER.info("Export to ggobi file: %s", file_path)
        save_data_arrays_to_xml(
            variables_names=variables_names,
            values_array=values_array,
            file_path=file_path,
        )

    def import_from_opendace(self, database_file: str | Path) -> None:
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
        name: str,
    ) -> str:
        """Return the name of the gradient related to a function.

        This name is the concatenation of a GRAD_TAG, e.g. '@',
        and the name of the function, e.g. 'f'.
        With this example, the name of the gradient is '@f'.

        Args:
            name: The name of a function.

        Returns:
            The name of the gradient based on the name of the function.
        """
        return f"{cls.GRAD_TAG}{name}"

    def __str__(self) -> str:
        return str(self.__dict)


class HashableNdarray:
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
        wrapped: ndarray,
        tight: bool = False,
    ) -> None:
        """
        Args:
            wrapped: The array that must be wrapped.
            tight: If True, the wrapped array is copied.
        """  # noqa: D205, D212, D415
        self.__tight = tight
        self.wrapped = array(wrapped) if tight else wrapped
        self.__hash = int(xxh3_64_hexdigest(wrapped.view(uint8)), 16)

    @property
    def tight(self) -> bool:
        """Whether the wrapped array is a copy of the original one.copied."""
        return self.__tight

    def __eq__(self, other: Any) -> bool:
        if hash(self) != hash(other):
            return False
        return array_equal(self.wrapped, other.wrapped)

    def __hash__(self) -> int:
        return self.__hash

    def __str__(self) -> str:
        return str(self.wrapped)

    def __repr__(self) -> str:
        return str(self)

    def unwrap(self) -> ndarray:
        """Return the encapsulated ndarray.

        Returns:
            The encapsulated ndarray, or a copy if the wrapper is ``tight``.
        """
        if self.__tight:
            return array(self.wrapped)

        return self.wrapped
