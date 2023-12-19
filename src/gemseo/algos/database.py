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
from collections.abc import Iterable
from collections.abc import Iterator
from collections.abc import Mapping
from itertools import chain
from itertools import islice
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Union
from xml.etree.ElementTree import parse as parse_element

from numpy import array
from numpy import atleast_1d
from numpy import atleast_2d
from numpy import hstack
from numpy import ndarray
from numpy.linalg import norm

from gemseo.algos._hdf_database import HDFDatabase
from gemseo.algos.hashable_ndarray import HashableNdarray
from gemseo.utils.ggobi_export import save_data_arrays_to_xml
from gemseo.utils.string_tools import pretty_repr
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from numbers import Number
    from pathlib import Path

    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

# Type of the values associated to the keys (values of input variables) in the database
DatabaseKeyType = Union[ndarray, HashableNdarray]
FunctionOutputValueType = Union[float, ndarray, list[int]]
DatabaseValueType = Mapping[str, FunctionOutputValueType]
ReturnedHdfMissingOutputType = tuple[
    Mapping[str, FunctionOutputValueType], Union[None, Mapping[str, int]]
]


class Database(Mapping):
    """Storage of :class:`.MDOFunction` evaluations.

    A :class:`.Database` is typically attached to an :class:`.OptimizationProblem`
    to store the evaluations of its objective, constraints and observables.

    Then,
    a :class:`.Database` can be an optimization history
    or a collection of samples in the case of a DOE.

    It is useful when simulations are costly
    because it avoids re-evaluating functions
    at points where they have already been evaluated.

    .. seealso:: :class:`.NormDBFunction`

    It can also be post-processed by an :class:`.OptPostProcessor`
    to visualize its content,
    e.g. :class:`.OptHistoryView` generating a series of graphs
    to visualize the histories of the objective, constraints and design variables.

    A :class:`.Database` can be saved to an HDF file
    for portability and cold post-processing
    with its method :meth:`.to_hdf`.
    A database can also be initialized from an HDF file
    as ``database = Database.from_hdf(file_path)``.

    .. note::
        Saving an :class:`.OptimizationProblem` to an HDF file
        using its method :class:`~.OptimizationProblem.to_hdf`
        also saves its :class:`.Database`.

    The database is based on a two-level dictionary-like mapping such as
    ``{x: {output_name: output_value, ...}, ...}`` with:

        * ``x``: the input value as an :class:`.HashableNdarray`
          wrapping a NumPy array that can be accessed as ``x.array``;
          if the types of the input variables are different,
          then they are promoted to the unique type that can represent all them,
          for instance integer would be promoted to float;
        * ``output_name``: either the name of the function
          that has been evaluated at ``x_vect``,
          the name of its gradient
          (the gradient of a function called ``"f"`` is typically denoted as ``"@f"``)
          and any additional information related to the methods which use the database;
        * ``outputs``: the output value,
          typically a float or a 1D-array for a function output,
          a 1D- or 2D-array for a gradient
          or a list for the iteration.
    """

    name: str
    """The name of the database."""

    MISSING_VALUE_TAG: ClassVar[str] = "NA"
    """The tag for a missing value."""

    GRAD_TAG: ClassVar[str] = "@"
    """The tag prefixing a function name to make it a gradient name.

    E.g. ``"@f"`` is the name of the gradient of ``"f"`` when ``GRAD_TAG == "@"``.
    """

    __data: dict[HashableNdarray, DatabaseValueType]
    """The input values bound to the output values."""

    __store_listeners: list[Callable]
    """The functions to be called when an item is stored to the database."""

    __new_iter_listeners: list[Callable]
    """The functions to be called when a new iteration is stored to the database."""

    __hdf_database: HDFDatabase
    """The handler to export the database to a HDF file."""

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name to be given to the database.
                If empty, use the class name.
        """  # noqa: D205, D212, D415
        self.name = name or self.__class__.__name__
        self.__data = {}
        self.__store_listeners = []
        self.__new_iter_listeners = []
        self.__hdf_database = HDFDatabase()

    @property
    def last_item(self) -> DatabaseValueType:
        """The last item of the database."""
        return next(reversed(self.__data.values()))

    @staticmethod
    def get_hashable_ndarray(
        original_array: DatabaseKeyType,
        copy: bool = False,
    ) -> HashableNdarray:
        """Convert an array to a hashable array.

        This hashable array basically represents a key of the database.

        Args:
            original_array: An array.
            copy: Whether to copy the original array.

        Returns:
            A hashable array wrapping the original array.

        Raises:
            KeyError: If the original array is
                neither an array nor a :class:`.HashableNdarray`.
        """
        if isinstance(original_array, ndarray):
            return HashableNdarray(original_array, copy=copy)

        if isinstance(original_array, HashableNdarray):
            if copy:
                original_array.copy_wrapped_array()
            return original_array

        raise KeyError(
            "A database key must be either a NumPy array of a HashableNdarray; "
            f"got {type(original_array)} instead."
        )

    def __getitem__(self, x_vect: DatabaseKeyType) -> DatabaseValueType | None:
        return self.__data[self.get_hashable_ndarray(x_vect)]

    def __iter__(self) -> Iterator[HashableNdarray]:
        return iter(self.__data)

    def __len__(self) -> int:
        return len(self.__data)

    def __delitem__(self, x_vect: DatabaseKeyType) -> None:
        del self.__data[self.get_hashable_ndarray(x_vect)]

    @property
    def n_iterations(self) -> int:
        """The number of iterations.

        This is the number of entries in the database.
        """
        return len(self)

    def clear(self) -> None:
        """Clear the database."""
        self.__data.clear()

    def clear_from_iteration(self, iteration: int) -> None:
        """Delete the items after a given iteration.

        Args:
            iteration: An iteration between 1 and the number of iterations;
                it can also be a negative integer if counting from the last iteration
                (e.g. -2 for the penultimate iteration).
        """
        iteration_index = self.__get_index(iteration)
        for index, x in enumerate(tuple(self.__data.keys())):
            if index > iteration_index:
                del self.__data[x]

    def remove_empty_entries(self) -> None:
        """Remove the entries that do not have output values."""
        for x, outputs in tuple(self.items()):
            if not outputs:
                del self.__data[x]

    def filter(self, output_names: Iterable[str]) -> None:  # noqa: A003
        """Keep only some outputs and remove the other ones.

        Args:
            output_names: The names of the outputs that must be kept.
        """
        output_names = set(output_names)
        for output_names_to_values in self.values():
            for function_name in output_names_to_values.keys() - output_names:
                del output_names_to_values[function_name]

    def get_last_n_x_vect(self, n: int) -> list[ndarray]:
        """Return the last ``n`` input values.

        Args:
            n: The number of last iterations to be considered.

        Returns:
            The last ``n`` input value.

        Raises:
            ValueError: If the number ``n`` is higher than the number of iterations.
        """
        n_iterations = len(self)
        if n > n_iterations:
            raise ValueError(
                f"The number of last iterations ({n}) is greater "
                f"than the number of iterations ({n_iterations})."
            )
        return [
            x.wrapped_array
            for x in islice(self.__data.keys(), n_iterations - n, n_iterations)
        ]

    def get_x_vect_history(self) -> list[ndarray]:
        """Return the history of the input vector.

        Returns:
            The history of the input vector.
        """
        return [x.wrapped_array for x in self.__data]

    def check_output_history_is_empty(self, output_name: str) -> bool:
        """Check if the history of an output is empty.

        Args:
            output_name: The name of the output.

        Returns:
            Whether the history of the output is empty.
        """
        return all(output_name not in outputs for outputs in self.values())

    def get_function_history(
        self,
        function_name: str,
        with_x_vect: bool = False,
    ) -> ndarray | tuple[ndarray, ndarray]:
        """Return the history of a function output.

        Args:
            function_name: The name of the function.
            with_x_vect: Whether to return also the input history.

        Returns:
            The history of the function output, and possibly the input history.
        """
        output_history = []
        input_history = []
        for x, outputs in self.items():
            function_value = outputs.get(function_name)
            if function_value is not None:
                if isinstance(function_value, ndarray) and function_value.size == 1:
                    function_value = function_value[0]
                output_history.append(function_value)

                if with_x_vect:
                    input_history.append(x.wrapped_array)

        try:
            output_history = array(output_history)
        except ValueError:
            # For Numpy > 1.24 that no longer automatically handle containers that
            # cannot produce an array with a consistent shape.
            output_history = array(output_history, dtype=object)

        if with_x_vect:
            return output_history, array(input_history)

        return output_history

    def get_gradient_history(
        self,
        function_name: str,
        with_x_vect: bool = False,
    ) -> ndarray | tuple[ndarray, ndarray]:
        """Return the history of the gradient of a function.

        Args:
            function_name: The name of the function
                for which we want the gradient history.
            with_x_vect: Whether the input history should be returned as well.

        Returns:
            The history of the gradient of the function output,
            and possibly the input history.
        """
        return self.get_function_history(
            function_name=self.get_gradient_name(function_name),
            with_x_vect=with_x_vect,
        )

    def get_iteration(self, x_vect: ndarray) -> int:
        """Return the iteration of an input value in the database.

        Args:
            x_vect: The input value.

        Returns:
            The iteration of the input values in the database.

        Raises:
            KeyError: If the required input value is not found.
        """
        hashed_input_value = HashableNdarray(x_vect)
        for index, key in enumerate(self.__data.keys()):
            if key == hashed_input_value:
                return index + 1

        raise KeyError(x_vect)

    def get_x_vect(self, iteration: int) -> ndarray:
        """Return the input value at a specified iteration.

        Args:
            iteration: An iteration between 1 and the number of iterations;
                it can also be a negative integer if counting from the last iteration
                (e.g. -2 for the penultimate iteration).

        Returns:
            The input value at this iteration.
        """
        iteration_index = self.__get_index(iteration)
        # The database dictionary uses the input design variables as keys for the
        # function values. Here we convert it to an iterator that returns the
        # key located at the required iteration using the islice method from
        # itertools.
        x = next(islice(iter(self.__data), iteration_index, iteration_index + 1))
        return x.wrapped_array

    def __get_output(
        self,
        x_vect_or_iteration: DatabaseKeyType | int,
        tolerance: float = 0.0,
    ) -> DatabaseValueType | None:
        r"""Return the output value corresponding to a given input value.

        Args:
            x_vect_or_iteration: An input value
                or an iteration between 1 and the number of iterations;
                it can also be a negative integer if counting from the last iteration
                (e.g. -2 for the penultimate iteration).
            tolerance: The relative tolerance :math:`\epsilon`
                such that the input value :math:`x` is considered as equal
                to the input value :math:`x_{\text{database}}` stored in the database
                if
                :math:`\|x-x_{\text{database}}\|/\|x_{\text{database}}\|\leq\epsilon`.

        Returns:
            The output value at the given input value if any, otherwise ``None``.
        """
        if isinstance(x_vect_or_iteration, int):
            return self.__get_output(self.get_x_vect(x_vect_or_iteration))

        x = x_vect_or_iteration

        if abs(tolerance) < sys.float_info.epsilon:
            return self.get(x)

        if isinstance(x, HashableNdarray):
            x = x.wrapped_array

        for db_input_value, db_output_names_to_values in self.items():
            _db_in_value = db_input_value.wrapped_array
            if norm(_db_in_value - x) <= tolerance * norm(_db_in_value):
                return db_output_names_to_values

        return None

    def get_function_value(
        self,
        function_name: str,
        x_vect_or_iteration: DatabaseKeyType | int,
        tolerance: float = 0.0,
    ) -> FunctionOutputValueType | None:
        r"""Return the output value of a function corresponding to a given input value.

        Args:
            function_name: The name of the required output function.
            x_vect_or_iteration: An input value
                or an iteration between 1 and the number of iterations;
                it can also be a negative integer if counting from the last iteration
                (e.g. -2 for the penultimate iteration).
            tolerance: The relative tolerance :math:`\epsilon`
                such that the input value :math:`x` is considered as equal
                to the input value :math:`x_{\text{database}}` stored in the database
                if
                :math:`\|x-x_{\text{database}}\|/\|x_{\text{database}}\|\leq\epsilon`.

        Returns:
            The output value of the function at the given input value if any,
            otherwise ``None``.
        """
        outputs = self.__get_output(x_vect_or_iteration, tolerance)
        if outputs:
            return outputs.get(function_name)
        return None

    def store(
        self,
        x_vect: ndarray,
        outputs: DatabaseValueType,
    ) -> None:
        """Store the output values associated to the input values.

        Args:
            x_vect: The input value.
            outputs: The output value corresponding to the input value.
        """
        hashed_input_value = self.get_hashable_ndarray(x_vect, True)
        self.__hdf_database.add_pending_array(hashed_input_value)

        stored_outputs = self.get(hashed_input_value)
        current_outputs_is_empty = not stored_outputs

        if stored_outputs is None:
            self.__data[hashed_input_value] = outputs
        else:
            # No new keys = already computed = new iteration
            # otherwise just calls to other functions
            stored_outputs.update(outputs)

        if self.__store_listeners:
            self.notify_store_listeners(x_vect)

        # Notify the new iteration after storing x
        # because callbacks may need an updated x
        if self.__new_iter_listeners and outputs and current_outputs_is_empty:
            self.notify_new_iter_listeners(x_vect)

    def add_store_listener(self, function: Callable) -> None:
        """Add a function to be called when an item is stored to the database.

        Args:
            function: The function to be called.

        Raises:
            TypeError: If the argument is not a callable.
        """
        if not callable(function):
            raise TypeError("Listener function is not callable")
        self.__store_listeners.append(function)

    def add_new_iter_listener(self, function: Callable) -> None:
        """Add a function to be called when a new iteration is stored to the database.

        Args:
            function: The function to be called, it must have one argument that is
                the current input value.

        Raises:
            TypeError: If the argument is not a callable.
        """
        if not callable(function):
            raise TypeError("Listener function is not callable.")
        self.__new_iter_listeners.append(function)

    def clear_listeners(self) -> None:
        """Clear all the listeners."""
        self.__store_listeners = []
        self.__new_iter_listeners = []

    def notify_store_listeners(self, x_vect: DatabaseKeyType | None = None) -> None:
        """Notify the listeners that a new entry was stored in the database.

        Args:
            x_vect: The input value.
                If ``None``, use the input value of the last iteration.
        """
        self.__notify_listeners(self.__store_listeners, x_vect)

    def notify_new_iter_listeners(self, x_vect: DatabaseKeyType | None = None) -> None:
        """Notify the listeners that a new iteration is ongoing.

        Args:
            x_vect: The input value.
                If ``None``, use the input value of the last iteration.
        """
        self.__notify_listeners(self.__new_iter_listeners, x_vect)

    def __notify_listeners(
        self,
        listeners: list[Callable],
        x_vect: DatabaseKeyType | None,
    ) -> None:
        """Notify the listeners.

        Args:
            listeners: The listeners.
            x_vect: The input value.
                If ``None``, use the input value of the last iteration.
        """
        if not listeners:
            return

        if isinstance(x_vect, HashableNdarray):
            x_vect = x_vect.wrapped_array
        elif x_vect is None:
            x_vect = self.get_x_vect(-1)

        for function in listeners:
            function(x_vect)

    def get_function_names(self, skip_grad: bool = True) -> list[str]:
        """Return the names of the outputs contained in the database.

        Args:
            skip_grad: Whether to skip the names of gradient functions.

        Returns:
            The names of the outputs in alphabetical order.
        """
        output_names = set()
        for output_names_to_values in self.__data.values():
            for outputs in output_names_to_values:
                if skip_grad and outputs.startswith(self.GRAD_TAG):
                    continue
                output_names.add(outputs)

        return sorted(output_names)

    def get_history(
        self,
        function_names: Iterable[str] = (),
        add_missing_tag: bool = False,
        missing_tag: str | float = MISSING_VALUE_TAG,
    ) -> tuple[list[list[float | ndarray]], list[ndarray]]:
        """Return the history of the inputs and outputs.

        This includes the inputs, functions and gradients.

        Args:
            function_names: The names of functions.
            add_missing_tag: Whether to add the tag ``missing_tag``
                to the iterations where data are missing.
            missing_tag: The tag to represent missing data.

        Returns:
            The history of the output values,
            then the history of the input values.

        Raises:
            ValueError: When a function has no values in the database.
        """
        if not function_names:
            function_names = self.get_function_names()
        else:
            all_function_names = set(self.get_function_names(skip_grad=False))
            not_function_names = set(function_names) - all_function_names
            if not_function_names:
                suffix = (
                    "is not an output name"
                    if len(not_function_names) == 1
                    else "are not output names"
                )
                raise ValueError(
                    f"{pretty_repr(not_function_names, use_and=True)} {suffix}; "
                    f"available ones are "
                    f"{pretty_repr(all_function_names, use_and=True)}."
                )

        output_history = []
        input_history = []
        for x, output_names_to_values in self.items():
            output_values = []
            for function_name in function_names:
                if function_name in output_names_to_values:
                    output_values.append(output_names_to_values[function_name])
                elif add_missing_tag:
                    output_values.append(missing_tag)

            if output_values:
                output_history.append(output_values)
                input_history.append(x.wrapped_array)

        return output_history, input_history

    def to_hdf(
        self,
        file_path: str | Path = "optimization_history.h5",
        append: bool = False,
    ) -> None:
        """Export the optimization database to an HDF file.

        Args:
            file_path: The path of the HDF file.
            append: Whether to append the data to the file.
        """
        self.__hdf_database.to_file(self, file_path, append)

    @classmethod
    def from_hdf(
        cls,
        file_path: str | Path = "optimization_history.h5",
        name: str = "",
    ) -> Database:
        """Create a database from an HDF file.

        Args:
            file_path: The path of the HDF file.
            name: The name of the database.

        Returns:
            The database defined in the file.
        """
        database = cls(name)
        database.update_from_hdf(file_path)
        return database

    def update_from_hdf(
        self,
        file_path: str | Path = "optimization_history.h5",
    ) -> None:
        """Update the current database from an HDF file.

        Args:
            file_path: The path of the HDF file.
        """
        self.__hdf_database.update_from_file(self, file_path)

    def get_history_array(
        self,
        function_names: Iterable[str] = (),
        add_missing_tag: bool = False,
        missing_tag: str | float = MISSING_VALUE_TAG,
        input_names: str | Iterable[str] = (),
        with_x_vect: bool = True,
    ) -> tuple[NDArray[Number | str], list[str], Iterable[str]]:
        """Return the database as a 2D array shaped as ``(n_iterations, n_features)``.

        The features are the outputs of interest and possibly the input variables.

        Args:
            function_names: The names of the functions
                whose output values must be returned.
                If empty, use all the functions.
            input_names: The names of the input variables.
                If empty, use :attr:`.input_names`.
            add_missing_tag: If ``True``,
                add the tag specified in ``missing_tag``
                for data that are not available.
            missing_tag: The tag that is added for data that are not available.
            with_x_vect: If ``True``,
                the input variables are returned in the history
                as ``np.hstack((get_output_history, x_vect_history))``.

        Returns:
            The history as an 2D array
            whose rows are observations and columns are the variables,
            the names of these columns
            and the names of the functions.
        """
        f_names = function_names
        if not f_names:
            f_names = self.get_function_names()

        f_history, x_history = self.get_history(f_names, add_missing_tag, missing_tag)
        f_flat_names, f_flat_values = self.__split_history(f_history, f_names)
        variables_flat_names = f_flat_names
        try:
            f_history = array(f_flat_values).real
        except ValueError:
            # For Numpy > 1.24 that no longer automatically handle containers that
            # cannot produce an array with a consistent shape.
            f_history = array(f_flat_values, dtype=object).real
        if with_x_vect:
            if not input_names:
                x_names = [f"x_{i + 1}" for i in range(len(self))]
            elif isinstance(input_names, str):
                x_names = [input_names]
            else:
                x_names = input_names

            x_flat_names, x_flat_values = self.__split_history(x_history, x_names)
            variables_flat_names = f_flat_names + x_flat_names
            x_history = array(x_flat_values).real
            variables_history = hstack((f_history, x_history))
        else:
            variables_history = f_history

        return atleast_2d(variables_history), variables_flat_names, f_names

    @staticmethod
    def __split_history(
        history: list[list[float | ndarray]] | list[ndarray],
        names: Iterable[str],
    ) -> tuple[list[str], list[list[float]]]:
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
                    repr_variable(name, i, size) for i in range(size)
                ]

            flat_values.append(flat_value)

        return list(chain(*names_to_flat_names.values())), flat_values

    def to_ggobi(
        self,
        function_names: Iterable[str] = (),
        file_path: str | Path = "opt_hist.xml",
        input_names: str | Iterable[str] = (),
    ) -> None:
        """Export the database to an XML file for ggobi tool.

        Args:
            function_names: The names of functions.
                If empty, use all the functions.
            file_path: The path to the XML file.
            input_names: The names of the input variables.
                If empty, use :attr:`.input_names`.
        """
        values_array, variable_names, function_names = self.get_history_array(
            function_names=function_names, add_missing_tag=True, input_names=input_names
        )
        LOGGER.info("Export to ggobi for functions: %s", str(function_names))
        LOGGER.info("Export to ggobi file: %s", file_path)
        save_data_arrays_to_xml(
            variable_names=variable_names,
            values_array=values_array,
            file_path=file_path,
        )

    def update_from_opendace(self, database_file: str | Path) -> None:
        """Update the current database from an opendace XML database.

        Args:
            database_file: The path to an opendace database.
        """
        tree = parse_element(database_file)
        for link in tree.getroot().iter("link"):
            data = {}
            for information in link:
                for x_ydyddy in information:
                    data[x_ydyddy.tag] = literal_eval(x_ydyddy.text)

            data_reformat = data["y"]
            for key, value in data["dy"].items():
                data_reformat[self.get_gradient_name(key[1:])] = array(value)

            self.store(array(data.pop("x")), data_reformat)

    @classmethod
    def get_gradient_name(cls, name: str) -> str:
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
        return str(self.__data)

    def __get_index(self, iteration: int) -> int:
        """Return the index from an iteration.

        Args:
            iteration: The iteration.

        Returns:
            The index.

        Raises:
            ValueError: If the iteration is out of the possible range of iterations.
        """
        len_self = len(self)

        if iteration == 0 or not (-len_self <= iteration <= len_self):
            raise ValueError(
                "The iteration must be within {-N, ..., -1, 1, ..., N} "
                f"where N={len_self} is the number of iterations."
            )

        if iteration > 0:
            return iteration - 1

        return len_self + iteration
