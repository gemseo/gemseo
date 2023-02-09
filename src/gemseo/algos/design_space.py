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
#        :author: Charlie Vanaret, Benoit Pauwels, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Design space.

A design space is used to represent the optimization's unknowns,
a.k.a. design variables.

A :class:`.DesignSpace` describes this design space at a given state,
in terms of names, sizes, types, bounds and current values of the design variables.

Variables can easily be added to the :class:`.DesignSpace`
using the :meth:`.DesignSpace.add_variable` method
or removed using the :meth:`.DesignSpace.remove_variable` method.

We can also filter the design variables using the :meth:`.DesignSpace.filter` method.

Getters and setters are also available
to get or set the value of a given variable property.

Lastly,
an instance of :class:`.DesignSpace` can be stored in a txt or HDF file.
"""
from __future__ import annotations

import collections
import logging
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import Any
from typing import ClassVar
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import Union

import h5py
from numpy import abs as np_abs
from numpy import array
from numpy import atleast_1d
from numpy import complex128
from numpy import concatenate
from numpy import dtype
from numpy import empty
from numpy import equal
from numpy import finfo
from numpy import float64
from numpy import full
from numpy import genfromtxt
from numpy import hstack
from numpy import inf
from numpy import int32
from numpy import isinf
from numpy import isnan
from numpy import logical_or
from numpy import mod
from numpy import ndarray
from numpy import nonzero
from numpy import ones_like
from numpy import round_ as np_round
from numpy import string_
from numpy import vectorize
from numpy import where
from numpy import zeros_like

from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.cache import hash_data_dict
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.base_enum import BaseEnum
from gemseo.utils.data_conversion import flatten_nested_dict
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.string_tools import pretty_str

LOGGER = logging.getLogger(__name__)


class DesignVariableType(BaseEnum):
    """A type of design variable."""

    FLOAT = "float"
    INTEGER = "integer"


VarType = Union[
    str,
    Sequence[str],
    DesignVariableType,
    Sequence[DesignVariableType],
]

DesignVariable = collections.namedtuple(
    "DesignVariable",
    ["size", "var_type", "l_b", "u_b", "value"],
    defaults=(
        1,
        DesignVariableType.FLOAT,
        None,
        None,
        None,
    ),
)


class DesignSpace(collections.abc.MutableMapping):
    """Description of a design space.

    It defines a set of variables from their names, sizes, types and bounds.

    In addition,
    it provides the current values of these variables
    that can be used as the initial solution of an :class:`.OptimizationProblem`.

    A :class:`.DesignSpace` has the same API as a dictionary,
    e.g. ``variable = design_space["x"]``,
    ``other_design_space["x"] = design_space["x"]``,
    ``del design_space["x"]``,
    ``for name, value in design_space["x"].items()``, ...
    """

    name: str | None
    """The name of the space."""

    dimension: int
    """The total dimension of the space, corresponding to the sum of the sizes of the
    variables."""

    # TODO: API: rename all variables_x to variable_x
    variables_names: list[str]
    """The names of the variables."""

    variables_sizes: dict[str, int]
    """The sizes of the variables."""

    variables_types: dict[str, ndarray]
    """The types of the variables components, which can be any
    :attr:`.DesignSpace.DesignVariableType`."""

    normalize: dict[str, ndarray]
    """The normalization policies of the variables components indexed by the variables
    names; if `True`, the component can be normalized."""

    __VARIABLE_TYPES_TO_DTYPES: ClassVar[dict[str, str]] = {
        DesignVariableType.FLOAT.value: "float64",
        DesignVariableType.INTEGER.value: "int32",
    }

    FLOAT = DesignVariableType.FLOAT
    INTEGER = DesignVariableType.INTEGER
    AVAILABLE_TYPES = [FLOAT, INTEGER]
    __TYPE_NAMES = tuple(x.value for x in DesignVariableType)
    MINIMAL_FIELDS = ["name", "lower_bound", "upper_bound"]
    TABLE_NAMES = ["name", "lower_bound", "value", "upper_bound", "type"]

    DESIGN_SPACE_GROUP = "design_space"
    NAME_GROUP = "name"
    NAMES_GROUP = "names"
    LB_GROUP = "l_b"
    UB_GROUP = "u_b"
    VAR_TYPE_GROUP = "var_type"
    VALUE_GROUP = "value"
    SIZE_GROUP = "size"
    # separator that denotes a vector's components
    SEP = "!"
    __INT_DTYPE = dtype("int32")
    __FLOAT_DTYPE = dtype("float64")
    __COMPLEX_DTYPE = dtype("complex128")

    __DEFAULT_COMMON_DTYPE = __FLOAT_DTYPE
    """The default NumPy data type of the variables."""

    __current_value_array: ndarray
    """The current value stored as a concatenated array."""

    __norm_current_value: dict[str, ndarray]
    """The norm of the current value."""

    __norm_current_value_array: ndarray
    """The norm of the current value stored as a concatenated array."""

    def __init__(
        self,
        hdf_file: str | Path | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            hdf_file: The path to the file
                containing the description of an initial design space.
                If None, start with an empty design space.
            name: The name to be given to the design space,
                `None` if the design space is unnamed.
        """  # noqa: D205, D212, D415
        self.name = name
        self.variables_names = []
        self.dimension = 0
        self.variables_sizes = {}
        self.variables_types = {}
        self.normalize = {}
        self._lower_bounds = {}
        self._upper_bounds = {}
        # These attributes are stored for faster computation of normalization
        # and unnormalization
        self._norm_factor = None
        self._norm_factor_inv = None
        self.__lower_bounds_array = None
        self.__upper_bounds_array = None
        self.__integer_components = None
        self.__no_integer = True
        self.__norm_data_is_computed = False
        self.__norm_inds = None
        self.__to_zero = None
        self.__bound_tol = 100.0 * finfo(float64).eps
        self.__current_value = {}
        self.__has_current_value = False
        self.__common_dtype = self.__DEFAULT_COMMON_DTYPE
        self.__clear_dependent_data()
        if hdf_file is not None:
            self.import_hdf(hdf_file)

    @property
    def _current_value(self) -> dict[str, ndarray]:
        """The current design value."""
        return self.__current_value

    # TODO: API: this is never used in all our codes, remove.
    @_current_value.setter
    def _current_value(self, value: Mapping[str, ndarray]) -> None:
        self.__current_value = value
        self.__update_current_metadata()

    def __update_current_metadata(self) -> None:
        """Update information about the current design value for quick access."""
        self.__update_current_status()
        self.__update_common_dtype()
        if self.__has_current_value:
            self.__clear_dependent_data()

    def __clear_dependent_data(self):
        """Reset the data that depends on the current value."""
        self.__current_value_array = array([])
        self.__norm_current_value = {}
        self.__norm_current_value_array = array([])

    def __update_common_dtype(self) -> None:
        """Update the common data type of the variables."""
        if self.__has_current_value:
            self.__common_dtype = self.__get_common_dtype(self.__current_value)
        else:
            self.__common_dtype = self.__DEFAULT_COMMON_DTYPE

    def __update_current_status(self) -> None:
        """Update the availability of current design values for all the variables."""
        if not self.__current_value or self.__current_value.keys() != set(
            self.variables_names
        ):
            self.__has_current_value = False
            return

        for value in self.__current_value.values():
            if value is None:
                self.__has_current_value = False
                return

        self.__has_current_value = True

    def __delitem__(
        self,
        name: str,
    ) -> None:
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        self.remove_variable(name)

    def remove_variable(
        self,
        name: str,
    ) -> None:
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        self.__norm_data_is_computed = False
        self.dimension -= self.variables_sizes[name]
        self.variables_names.remove(name)
        del self.variables_sizes[name]
        del self.variables_types[name]
        del self.normalize[name]
        if name in self._lower_bounds:
            del self._lower_bounds[name]

        if name in self._upper_bounds:
            del self._upper_bounds[name]

        if name in self.__current_value:
            del self.__current_value[name]

        self.__update_current_metadata()

    def filter(
        self,
        keep_variables: str | Iterable[str],
        copy: bool = False,
    ) -> DesignSpace:
        """Filter the design space to keep a subset of variables.

        Args:
            keep_variables: The names of the variables to be kept.
            copy: If ``True``, then a copy of the design space is filtered,
                otherwise the design space itself is filtered.

        Returns:
            Either the filtered original design space or a copy.

        Raises:
            ValueError: If the variable is not in the design space.
        """
        if isinstance(keep_variables, str):
            keep_variables = [keep_variables]
        design_space = deepcopy(self) if copy else self
        for name in deepcopy(self.variables_names):
            if name not in keep_variables:
                design_space.remove_variable(name)
        for name in keep_variables:
            if name not in self.variables_names:
                raise ValueError(f"Variable '{name}' is not known.")
        return design_space

    def filter_dim(
        self,
        variable: str,
        keep_dimensions: Iterable[int],
    ) -> DesignSpace:
        """Filter the design space to keep a subset of dimensions for a variable.

        Args:
            variable: The name of the variable.
            keep_dimensions: The dimensions of the variable to be kept,
                between :math:`0` and :math:`d-1`
                where :math:`d` is the number of dimensions of the variable.

        Returns:
            The filtered design space.

        Raises:
            ValueError: If a dimension is unknown.
        """
        self.__norm_data_is_computed = False
        removed_dimensions = list(
            set(range(self.variables_sizes[variable])) - set(keep_dimensions)
        )
        bad_dimensions = list(
            set(keep_dimensions) - set(range(self.variables_sizes[variable]))
        )
        size = len(removed_dimensions)
        self.dimension -= size
        self.variables_sizes[variable] -= size
        types = []
        for dimension in keep_dimensions:
            if dimension in bad_dimensions:
                self.remove_variable(variable)
                raise ValueError(
                    "Dimension {} of variable '{}' is not known.".format(
                        dimension, variable
                    )
                )
            types.append(self.variables_types[variable][dimension])
            self.variables_types[variable] = array(types)

        idx = keep_dimensions
        self.normalize[variable] = self.normalize[variable][idx]
        if variable in self._lower_bounds:
            self._lower_bounds[variable] = self._lower_bounds[variable][idx]

        if variable in self._upper_bounds:
            self._upper_bounds[variable] = self._upper_bounds[variable][idx]

        if variable in self.__current_value:
            self.__current_value[variable] = self.__current_value[variable][idx]

        self.__update_current_metadata()
        return self

    def add_variable(
        self,
        name: str,
        size: int = 1,
        var_type: VarType = DesignVariableType.FLOAT,
        l_b: float | ndarray | None = None,
        u_b: float | ndarray | None = None,
        value: float | ndarray | None = None,
    ) -> None:
        r"""Add a variable to the design space.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            var_type: Either the type of the variable
                or the types of its components.
            l_b: The lower bound of the variable.
                If None, use :math:`-\infty`.
            u_b: The upper bound of the variable.
                If None, use :math:`+\infty`.
            value: The default value of the variable.
                If None, do not use a default value.

        Raises:
            ValueError: Either if the variable already exists
                or if the size is not a positive integer.
        """
        if name in self.variables_names:
            raise ValueError(f"Variable '{name}' already exists.")

        if size <= 0 or int(size) != size:
            raise ValueError(f"The size of '{name}' should be a positive integer.")

        # name and size
        self.variables_names.append(name)
        self.dimension += size
        self.variables_sizes[name] = size
        # type
        self._add_type(name, size, var_type)

        # bounds
        self._add_bound(name, size, l_b)
        self._add_bound(name, size, u_b, is_lower=False)
        self._check_variable_bounds(name)

        # normalization policy
        self._add_norm_policy(name)

        if value is not None:
            array_value = atleast_1d(value)
            self._check_value(array_value, name)
            if len(array_value) == 1 and size > 1:
                array_value = full(size, value)
            self.__current_value[name] = array_value.astype(
                self.__VARIABLE_TYPES_TO_DTYPES[self.variables_types[name][0]],
                copy=False,
            )
            try:
                self._check_current_value(name)
            except ValueError:
                # If a ValueError is raised,
                # we must remove the variable from the design space.
                # When using a python script, this has no interest.
                # When using a notebook, a cell can raise a ValueError,
                # but we can continue to the next cell,
                # and use a design space which contains a variables that leads to error.
                self.remove_variable(name)
                raise

        self.__update_current_metadata()

    def _add_type(
        self,
        name: str,
        size: int,
        var_type: VarType = DesignVariableType.FLOAT,
    ) -> None:
        """Add a type to a variable.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            var_type: Either the type of the variable (see
            :attr:`.DesignSpace.AVAILABLE_TYPES`)
                or the types of its components.

        Raises:
            ValueError: Either if the number of component types is different
                from the variable size or if a variable type is unknown.
        """
        if not hasattr(var_type, "__iter__") or isinstance(
            var_type, (str, DesignVariableType, bytes)
        ):
            var_type = [var_type] * size

        if len(var_type) != size:
            raise ValueError(
                "The list of types for variable '{}' should be of size {}.".format(
                    name, size
                )
            )

        var_types = []

        for v_type in var_type:
            if isinstance(v_type, bytes):
                v_type = v_type.decode()

            if isinstance(v_type, str) and v_type not in self.__TYPE_NAMES:
                msg = f'The type "{v_type}" of {name} is not known.'
                raise ValueError(msg)
            elif v_type in DesignVariableType:
                v_type = v_type.value

            var_types += [v_type]

        self.variables_types[name] = array(var_types)
        self.__norm_data_is_computed = False

    def _add_norm_policy(
        self,
        name: str,
    ) -> None:
        """Add a normalization policy to a variable.

        Unbounded variables are not normalized.
        Bounded variables (both from above and from below) are normalized.

        Args:
            name: The name of a variable.

        Raises:
            ValueError: Either if the variable is not in the design space,
                if its size is not set,
                if the types of its components are not set
                or if there is no implemented normalization policy
                for the type of this variable.
        """
        # Check that the variable is in the design space:
        if name not in self.variables_names:
            raise ValueError(f"Variable '{name}' is not known.")
        # Check that the variable size is set:
        size = self.get_size(name)
        if size is None:
            raise ValueError(f"The size of variable '{name}' is not set.")
        # Check that the variables types are set:
        variables_types = self.variables_types.get(name, None)
        if variables_types is None:
            raise ValueError(f"The components types of variable '{name}' are not set.")
        # Set the normalization policy:
        normalize = empty(size)
        for i in range(size):
            var_type = variables_types[i]
            if var_type in self.__VARIABLE_TYPES_TO_DTYPES:
                if (
                    self._lower_bounds[name][i] == -inf
                    or self._upper_bounds[name][i] == inf
                ):
                    # Unbounded variables are not normalized:
                    normalize[i] = False
                elif self._lower_bounds[name][i] == self._upper_bounds[name][i]:
                    # Constant variables are not normalized:
                    normalize[i] = False
                else:
                    normalize[i] = True
            else:
                msg = "The normalization policy for type {0} is not implemented."
                raise ValueError(msg.format(var_type))
        self.normalize[name] = normalize

    @staticmethod
    def __is_integer(
        values: ndarray | Number,
    ) -> ndarray:
        """Check if each value is an integer.

        Args:
            values: The array or number to be checked.

        Returns:
            Whether each of the given values is an integer.
        """
        values = atleast_1d(values)

        return array([isinf(x) or x is None or not mod(x, 1) for x in values])

    @staticmethod
    def __is_numeric(
        value: Any,
    ) -> bool:
        """Check that a value is numeric.

        Args:
            value: The value to be checked.

        Returns:
            Whether the value is numeric.
        """
        res = (value is None) or hasattr(value, "real")
        try:
            if not res:
                float(value)
            return True
        except TypeError:
            return False

    @staticmethod
    def __is_not_nan(
        value: ndarray,
    ) -> bool:
        """Check that a value is not a nan.

        Args:
            value: The value to be checked.

        Returns:
            Whether the value is not a nan.
        """
        return (value is None) or ~isnan(value)

    def _check_value(
        self,
        value: ndarray,
        name: str,
    ) -> bool:
        """Check that the value of a variable is valid.

        Args:
            value: The value to be checked.
            name: The name of the variable.

        Returns:
            Whether the value of the variable is valid.

        Raises:
            ValueError: Either if the array is not one-dimensional,
                if the value is not numerizable,
                if the value is nan
                or if there is a component value which is not an integer
                while the variable type is integer and ``allow_inf_int_bound``
                is set to ``False``.
        """
        all_indices = set(range(len(value)))
        # OK if the variable value is one-dimensional
        if len(value.shape) > 1:
            raise ValueError(
                "Value {} of variable '{}' has dimension greater than 1 "
                "while a float or a 1d iterable object "
                "(array, list, tuple, ...) "
                "while a scalar was expected.".format(value, name)
            )

        # OK if all components are None
        if all(equal(value, None)):
            return True

        test = vectorize(self.__is_numeric)(value)
        indices = all_indices - set(list(where(test)[0]))
        for idx in indices:
            raise ValueError(
                f"Value {value[idx]} of variable '{name}' is not numerizable."
            )

        test = vectorize(self.__is_not_nan)(value)
        indices = all_indices - set(list(where(test)[0]))
        for idx in indices:
            raise ValueError(f"Value {value[idx]} of variable '{name}' is NaN.")

        # Check if some components of an integer variable are not integer.
        if self.variables_types[name][0] == DesignVariableType.INTEGER.value:
            indices = all_indices - set(nonzero(self.__is_integer(value))[0])
            for idx in indices:
                raise ValueError(
                    f"Component value {value[idx]} of variable '{name}'"
                    " is not an integer "
                    "while variable is of type integer "
                    f"(index: {idx})."
                )

    def _add_bound(
        self,
        name: str,
        size: int,
        bound: ndarray | Number,
        is_lower: bool = True,
    ) -> None:
        """Add a lower or upper bound to a variable.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            bound: The bound of the variable.
            is_lower: If True, the bound is a lower bound.
                Otherwise, it is an upper bound.

        Raises:
            ValueError: If the size of the bound is different
                from the size of the variable.
        """
        self.__norm_data_is_computed = False

        if is_lower:
            bounds = self._lower_bounds
        else:
            bounds = self._upper_bounds

        if bound is None:
            infinity = full(size, inf)
            if is_lower:
                bound_to_update = -infinity
            else:
                bound_to_update = infinity
            bounds.update({name: bound_to_update})
            return

        if is_lower:
            infinity = -inf
        else:
            infinity = inf

        bound_to_update = atleast_1d(bound)
        if None in bound_to_update:
            bound_to_update = where(
                equal(bound_to_update, None), infinity, bound_to_update
            ).astype(self.__FLOAT_DTYPE)

        self._check_value(bound_to_update, name)

        if isinstance(bound, Number):
            # scalar: same lower bound for all components
            bound_to_update = full(size, bound)
        elif len(bound_to_update) != size:
            bound_prefix = "lower" if is_lower else "upper"
            raise ValueError(
                f"The {bound_prefix} bounds of '{name}' should be of size {size}."
            )

        bounds.update({name: bound_to_update})

    def _check_variable_bounds(
        self,
        name: str,
    ) -> None:
        """Check that the bounds of a variable are compatible and have the same size.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the bounds of the variable are not valid.
        """
        l_b = self._lower_bounds.get(name, None)
        u_b = self._upper_bounds.get(name, None)
        inds = where(u_b < l_b)[0]
        if inds.size != 0:
            raise ValueError(
                "The bounds of variable '{}'{} are not valid: {}!<{}.".format(
                    name, inds, l_b[inds], u_b[inds]
                )
            )

    def _check_current_value(
        self,
        name: str,
    ) -> None:
        """Check that the current value of a variable is between its bounds.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the current value of the variable is outside its bounds.
        """
        l_b = self._lower_bounds.get(name, None)
        u_b = self._upper_bounds.get(name, None)
        current_value = self.__current_value.get(name, None)
        not_none = ~equal(current_value, None)
        indices = where(
            logical_or(
                current_value[not_none] < l_b[not_none] - self.__bound_tol,
                current_value[not_none] > u_b[not_none] + self.__bound_tol,
            )
        )[0]
        for index in indices:
            raise ValueError(
                "The current value of variable '{}' ({}) is not "
                "between the lower bound {} and the upper bound {}.".format(
                    name, current_value[index], l_b[index], u_b[index]
                )
            )

    def has_current_value(self) -> bool:
        """Check if each variable has a current value.

        Returns:
            Whether the current design value is defined for all variables.
        """
        return self.__has_current_value

    def has_integer_variables(self) -> bool:
        """Check if the design space has at least one integer variable.

        Returns:
            Whether the design space has at least one integer variable.
        """
        return self.INTEGER.value in [
            self.variables_types[variable_name][0]
            for variable_name in self.variables_names
        ]

    def check(self) -> None:
        """Check the state of the design space.

        Raises:
            ValueError: If the design space is empty.
        """
        if not self.variables_names:
            raise ValueError("The design space is empty.")

        for name in self.variables_names:
            self._check_variable_bounds(name)

        if self.has_current_value():
            self._check_current_names()

    def check_membership(
        self,
        x_vect: Mapping[str, ndarray] | ndarray,
        variable_names: Sequence[str] | None = None,
    ) -> None:
        """Check whether the variables satisfy the design space requirements.

        Args:
            x_vect: The values of the variables.
            variable_names: The names of the variables.
                If None, use the names of the variables of the design space.

        Raises:
            ValueError: Either if the dimension of the values vector is wrong,
                if the values are not specified as an array or a dictionary,
                if the values are outside the bounds of the variables or
                if the component of an integer variable is not an integer.
        """
        if isinstance(x_vect, dict):
            self.__check_membership(x_vect, variable_names)
        elif isinstance(x_vect, ndarray):
            if x_vect.size != self.dimension:
                raise ValueError(
                    f"The array should be of size {self.dimension}; got {x_vect.size}."
                )
            if variable_names is None:
                if self.__lower_bounds_array is None:
                    self.__lower_bounds_array = self.get_lower_bounds()

                if self.__upper_bounds_array is None:
                    self.__upper_bounds_array = self.get_upper_bounds()

                self.__check_membership_x_vect(x_vect)
            else:
                self.__check_membership(
                    split_array_to_dict_of_arrays(
                        x_vect, self.variables_sizes, variable_names
                    ),
                    variable_names,
                )
        else:
            raise TypeError(
                "The input vector should be an array or a dictionary; "
                f"got a {type(x_vect)} instead."
            )

    def __check_membership_x_vect(self, x_vect: ndarray) -> None:
        """Check whether a vector is comprised between the lower and upper bounds.

        Args:
            x_vect: The vector.

        Raises:
            ValueError: When the values are outside the bounds of the variables.
        """
        l_b = self.__lower_bounds_array
        u_b = self.__upper_bounds_array
        indices = where(x_vect < l_b - self.__bound_tol)[0]
        if len(indices):
            value = x_vect[indices]
            lower_bound = l_b[indices]
            raise ValueError(
                f"The components {indices} of the given array ({value}) "
                f"are lower than the lower bound ({lower_bound}) "
                f"by {lower_bound - value}."
            )

        indices = where(x_vect > u_b + self.__bound_tol)[0]
        if len(indices):
            value = x_vect[indices]
            upper_bound = u_b[indices]
            raise ValueError(
                f"The components {indices} of the given array ({value}) "
                f"are greater than the upper bound ({upper_bound}) "
                f"by {value - upper_bound}."
            )

    def __check_membership(
        self,
        x_dict: Mapping[str, ndarray],
        variable_names: Iterable[str] | None,
    ) -> None:
        """Check whether the variables satisfy the design space requirements.

        Args:
            x_dict: The values of the variables.
            variable_names: The names of the variables.
                If None, use the names of the variables of the design space.

        Raises:
            ValueError: Either if the dimension of an array is wrong,
                if the values are outside the bounds of the variables or
                if the component of an integer variable is not an integer.
        """
        variable_names = variable_names or self.variables_names
        for name in variable_names:
            value = x_dict[name]
            if value is None:
                continue

            size = self.variables_sizes[name]
            l_b = self._lower_bounds.get(name, None)
            u_b = self._upper_bounds.get(name, None)

            if value.size != size:
                raise ValueError(
                    f"The variable {name} of size {size} "
                    f"cannot be set with an array of size {value.size}."
                )

            for i in range(size):
                x_real = value[i].real

                if l_b is not None and x_real < l_b[i] - self.__bound_tol:
                    raise ValueError(
                        f"The component {name}[{i}] of the given array ({x_real}) "
                        f"is lower than the lower bound ({l_b[i]}) "
                        f"by {l_b[i] - x_real:.1e}."
                    )

                if u_b is not None and u_b[i] + self.__bound_tol < x_real:
                    raise ValueError(
                        f"The component {name}[{i}] of the given array ({x_real}) "
                        f"is greater than the upper bound ({l_b[i]}) "
                        f"by {x_real - u_b[i]:.1e}."
                    )

                if (
                    self.variables_types[name][0] == DesignVariableType.INTEGER.value
                ) and not self.__is_integer(x_real):
                    raise ValueError(
                        f"The variable {name} is of type integer; "
                        f"got {name}[{i}] = {x_real}."
                    )

    def get_active_bounds(
        self,
        x_vec: ndarray | None = None,
        tol: float = 1e-8,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
        """Determine which bound constraints of a design value are active.

        Args:
            x_vec: The design value at which to check the bounds.
                If ``None``, use the current design value.
            tol: The tolerance of comparison of a scalar with a bound.

        Returns:
            Whether the components of the lower and upper bound constraints are active,
            the first returned value representing the lower bounds
            and the second one the upper bounds, e.g.

            .. code-block:: python

                   ({'x': array(are_x_lower_bounds_active),
                     'y': array(are_y_lower_bounds_active)},
                    {'x': array(are_x_upper_bounds_active),
                     'y': array(are_y_upper_bounds_active)}
                   )

            where:

            .. code-block:: python

                are_x_lower_bounds_active = [True, False]
                are_x_upper_bounds_active = [False, False]
                are_y_lower_bounds_active = [False]
                are_y_upper_bounds_active = [True]
        """
        if x_vec is None:
            current_x = self.__current_value
            self.check_membership(self.get_current_value())
        elif isinstance(x_vec, ndarray):
            current_x = self.array_to_dict(x_vec)
        elif isinstance(x_vec, dict):
            current_x = x_vec
        else:
            raise TypeError(
                "Expected dict or array for x_vec argument; "
                "got {}.".format(type(x_vec))
            )

        active_l_b = {}
        active_u_b = {}
        for name in self.variables_names:
            l_b = self._lower_bounds.get(name)
            l_b = where(equal(l_b, None), -inf, l_b)
            u_b = self._upper_bounds.get(name)
            u_b = where(equal(u_b, None), inf, u_b)
            x_vec_i = current_x[name]
            # lower bound saturated
            active_l_b[name] = where(np_abs(x_vec_i - l_b) <= tol, True, False)
            # upper bound saturated
            active_u_b[name] = where(np_abs(x_vec_i - u_b) <= tol, True, False)

        return active_l_b, active_u_b

    def _check_current_names(
        self,
        variable_names: Iterable[str] | None = None,
    ) -> None:
        """Check the names of the current design value.

        Args:
            variable_names: The names of the variables.
                If None, use the names of the variables of the design space.

        Raises:
            ValueError: If the names of the variables of the current design value
                and the names of the variables of the design space are different.
        """
        if sorted(set(self.variables_names)) != sorted(self.__current_value.keys()):
            raise ValueError(
                "Expected current_x variables: {}; got {}.".format(
                    self.variables_names, list(self.__current_value.keys())
                )
            )
        self.check_membership(self.__current_value, variable_names)

    def get_current_value(
        self,
        variable_names: Sequence[str] | None = None,
        complex_to_real: bool = False,
        as_dict: bool = False,
        normalize: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the current design value.

        If the names of the variables are empty then an empty data is returned.

        Args:
            variable_names: The names of the design variables.
                If ``None``, use all the design variables.
            complex_to_real: Whether to cast complex numbers to real ones.
            as_dict: Whether to return the current design value
                as a dictionary of the form ``{variable_name: variable_value}``.
            normalize: Whether to normalize the design values in :math:`[0,1]`
                with the bounds of the variables.

        Returns:
            The current design value.

        Raises:
            ValueError: If names in ``variable_names`` are not in the design space.

        Warnings:
            For performance purposes,
            :meth:`.get_current_value` does not return a copy of the current value.
            This means that modifying the returned object
            will make the :class:`.DesignSpace` inconsistent
            (the current design value stored as a NumPy array
            and the current design value stored as a dictionary of NumPy arrays
            will be different).
            To modify the returned object without impacting the :class:`.DesignSpace`,
            you shall copy this object and modify the copy.

        See Also:
            To modify the current value,
            please use :meth:`.set_current_value` or :meth:`.set_current_variable`.
        """
        if variable_names is not None:
            if not variable_names:
                return {} if as_dict else array([])

            not_variable_names = set(variable_names) - set(self.variables_names)
            if not_variable_names:
                raise ValueError(
                    f"There are no such variables named: {pretty_str(not_variable_names)}."
                )

        if self.__has_current_value and not len(self.__current_value_array):
            self.__current_value_array = self.dict_to_array(self.__current_value)

        if normalize:
            if self.__has_current_value and not len(self.__norm_current_value_array):
                self.__norm_current_value_array = self.normalize_vect(
                    self.__current_value_array
                )
                self.__norm_current_value = self.array_to_dict(
                    self.__norm_current_value_array
                )
            current_x_array = self.__norm_current_value_array
            current_x_dict = self.__norm_current_value
        else:
            current_x_array = self.__current_value_array
            current_x_dict = self.__current_value

        if variable_names is None or set(variable_names) == set(self.variables_names):
            if as_dict:
                if complex_to_real:
                    return {k: v.real for k, v in current_x_dict.items()}
                else:
                    return current_x_dict

            if not self.__has_current_value:
                variables = set(self.variables_names) - current_x_dict.keys()
                raise KeyError(
                    "There is no current value for the design variables: "
                    f"{pretty_str(variables)}."
                )

            if variable_names is None or list(variable_names) == self.variables_names:
                if complex_to_real:
                    return current_x_array.real
                else:
                    return current_x_array

        if as_dict:
            current_value = {name: current_x_dict[name] for name in variable_names}
            if complex_to_real:
                return {k: v.real for k, v in current_value.items()}
            else:
                return current_value
        else:
            current_x_array = self.dict_to_array(
                current_x_dict, variable_names=variable_names
            )
            if complex_to_real:
                return current_x_array.real
            return current_x_array

    def get_indexed_var_name(
        self,
        variable_name: str,
    ) -> str | list[str]:
        """Create the names of the components of a variable.

        If the size of the variable is equal to 1,
        this method returns the name of the variable.
        Otherwise,
        it concatenates the name of the variable,
        the separator :attr:`.DesignSpace.SEP` and the index of the component.

        Args:
            variable_name: The name of the variable.

        Returns:
            The names of the components of the variable.
        """
        size = self.variables_sizes[variable_name]
        if size == 1:
            return variable_name
        return [variable_name + self.SEP + str(i) for i in range(size)]

    def get_indexed_variables_names(self) -> list[str]:
        """Create the names of the components of all the variables.

        If the size of the variable is equal to 1,
        this method uses its name.
        Otherwise,
        it concatenates the name of the variable,
        the separator :attr:`.DesignSpace.SEP` and the index of the component.

        Returns:
            The name of the components of all the variables.
        """
        var_ind_names = []
        for var in self.variables_names:
            vnames = self.get_indexed_var_name(var)
            if isinstance(vnames, str):
                var_ind_names.append(vnames)
            else:
                var_ind_names += vnames
        return var_ind_names

    def get_variables_indexes(
        self,
        variable_names: Iterable[str],
    ) -> ndarray:
        """Return the indexes of a design array corresponding to the variables names.

        Args:
            variable_names: The names of the variables.

        Returns:
            The indexes of a design array corresponding to the variables names.
        """
        indexes = list()
        index = 0
        for name in self.variables_names:
            var_size = self.get_size(name)
            if name in variable_names:
                indexes.extend(range(index, index + var_size))
            index += var_size
        return array(indexes)

    def __update_normalization_vars(self) -> None:
        """Compute the inner attributes used for normalization and unnormalization."""
        self.__lower_bounds_array = self.get_lower_bounds()
        self.__upper_bounds_array = self.get_upper_bounds()
        self._norm_factor = self.__upper_bounds_array - self.__lower_bounds_array

        norm_array = self.dict_to_array(self.normalize)
        self.__norm_inds = where(norm_array)[0]
        # In case lb=ub
        self.__to_zero = where(self._norm_factor == 0.0)[0]
        self._norm_factor_inv = 1.0 / (
            self.__upper_bounds_array - self.__lower_bounds_array
        )
        self._norm_factor_inv[self.__to_zero] = 1.0

        self.__integer_components = concatenate(
            [
                self.variables_types[variable_name] == DesignVariableType.INTEGER.value
                for variable_name in self.variables_names
            ]
        )
        self.__no_integer = not self.__integer_components.any()
        self.__norm_data_is_computed = True

    def normalize_vect(
        self,
        x_vect: ndarray,
        minus_lb: bool = True,
        out: ndarray | None = None,
    ) -> ndarray:
        r"""Normalize a vector of the design space.

        If `minus_lb` is True:

        .. math::

           x_u = \frac{x-l_b}{u_b-l_b}

        where :math:`l_b` and :math:`u_b` are the lower and upper bounds of :math:`x`.

        Otherwise:

        .. math::

           x_u = \frac{x}{u_b-l_b}

        Unbounded variables are not normalized.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If True, remove the lower bounds at normalization.
            out: The array to store the normalized vector.
                If None, create a new array.

        Returns:
            The normalized vector.
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()

        if out is None:
            use_out = False
            out = x_vect.copy()
        else:
            use_out = True
            out[...] = x_vect

        # Normalize the relevant components:
        current_x_dtype = self.__common_dtype
        # Normalization will not work with integers.
        if current_x_dtype.kind == "i":
            current_x_dtype = self.__FLOAT_DTYPE

        if out.dtype != current_x_dtype:
            if use_out:
                out[...] = out.astype(current_x_dtype, copy=False)
            else:
                out = out.astype(current_x_dtype, copy=False)

        norm_inds = self.__norm_inds
        if minus_lb:
            out[..., norm_inds] -= self.__lower_bounds_array[norm_inds]

        out[..., norm_inds] *= self._norm_factor_inv[norm_inds]

        # In case lb=ub, put value to 0.
        to_zero = self.__to_zero
        if to_zero.size > 0:
            out[..., to_zero] = 0.0

        return out

    def normalize_grad(
        self,
        g_vect: ndarray,
    ) -> ndarray:
        r"""Normalize an unnormalized gradient.

        This method is based on the chain rule:

        .. math::

           \frac{df(x)}{dx}
           = \frac{df(x)}{dx_u}\frac{dx_u}{dx}
           = \frac{df(x)}{dx_u}\frac{1}{u_b-l_b}

        where
        :math:`x_u = \frac{x-l_b}{u_b-l_b}` is the normalized input vector,
        :math:`x` is the unnormalized input vector
        and :math:`l_b` and :math:`u_b` are the lower and upper bounds of :math:`x`.

        Then,
        the normalized gradient reads:

        .. math::

           \frac{df(x)}{dx_u} = (u_b-l_b)\frac{df(x)}{dx}

        where :math:`\frac{df(x)}{dx}` is the unnormalized one.

        Args:
            g_vect: The gradient to be normalized.

        Returns:
            The normalized gradient.
        """
        return self.unnormalize_vect(g_vect, minus_lb=False, no_check=True)

    def unnormalize_grad(
        self,
        g_vect: ndarray,
    ) -> ndarray:
        r"""Unnormalize a normalized gradient.

        This method is based on the chain rule:

        .. math::

           \frac{df(x)}{dx}
           = \frac{df(x)}{dx_u}\frac{dx_u}{dx}
           = \frac{df(x)}{dx_u}\frac{1}{u_b-l_b}

        where
        :math:`x_u = \frac{x-l_b}{u_b-l_b}` is the normalized input vector,
        :math:`x` is the unnormalized input vector,
        :math:`\frac{df(x)}{dx_u}` is the unnormalized gradient
        :math:`\frac{df(x)}{dx}` is the normalized one,
        and :math:`l_b` and :math:`u_b` are the lower and upper bounds of :math:`x`.

        Args:
            g_vect: The gradient to be unnormalized.

        Returns:
            The unnormalized gradient.
        """
        return self.normalize_vect(g_vect, minus_lb=False)

    def unnormalize_vect(
        self,
        x_vect: ndarray,
        minus_lb: bool = True,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        """Unnormalize a normalized vector of the design space.

        If `minus_lb` is True:

        .. math::

           x = x_u(u_b-l_b) + l_b

        where
        :math:`x_u` is the normalized input vector,
        :math:`x` is the unnormalized input vector
        and :math:`l_b` and :math:`u_b` are the lower and upper bounds of :math:`x`.

        Otherwise:

        .. math::

           x = x_u(u_b-l_b)

        Args:
            x_vect: The values of the design variables.
            minus_lb: Whether to remove the lower bounds at normalization.
            no_check: Whether to check if the components are in :math:`[0,1]`.
            out: The array to store the unnormalized vector.
                If None, create a new array.

        Returns:
            The unnormalized vector.
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()

        norm_inds = self.__norm_inds
        lower_bounds = self.__lower_bounds_array
        if not no_check:
            value = x_vect[..., norm_inds]
            lower_bounds_violated = value < -self.__bound_tol
            upper_bounds_violated = value > 1 + self.__bound_tol
            any_lower_bound_violated = lower_bounds_violated.any()
            any_upper_bound_violated = upper_bounds_violated.any()
            msg = "All components of the normalized vector should be between 0 and 1; "
            if any_lower_bound_violated:
                msg += "lower bounds violated: {}; ".format(
                    value[lower_bounds_violated]
                )

            if any_upper_bound_violated:
                msg += "upper bounds violated: {}; ".format(
                    value[upper_bounds_violated]
                )

            if any_lower_bound_violated or any_upper_bound_violated:
                msg = msg[:-2] + "."
                LOGGER.warning(msg)

        if out is None:
            out = x_vect.copy()
        else:
            out *= 0
            out = x_vect

        # Unnormalize the relevant components:
        recast_to_int = False
        current_x_dtype = self.__common_dtype
        # Normalization will not work with integers.
        if current_x_dtype.kind == "i":
            current_x_dtype = self.__FLOAT_DTYPE
            recast_to_int = True

        if out.dtype != current_x_dtype:
            out = out.astype(current_x_dtype, copy=False)

        out[..., norm_inds] *= self._norm_factor[norm_inds]
        if minus_lb:
            out[..., norm_inds] += lower_bounds[norm_inds]

        # In case lb=ub, put value to lower bound.
        to_lower_bounds = self.__to_zero
        if to_lower_bounds.size > 0:
            out[..., to_lower_bounds] = lower_bounds[to_lower_bounds]

        if not self.__no_integer:
            self.round_vect(out, copy=False)
            if recast_to_int:
                out = out.astype(int32)

        return out

    def transform_vect(
        self,
        vector: ndarray,
        out: ndarray | None = None,
    ) -> ndarray:
        """Map a point of the design space to a vector with components in :math:`[0,1]`.

        Args:
            vector: A point of the design space.
            out: The array to store the transformed vector.
                If None, create a new array.

        Returns:
            A vector with components in :math:`[0,1]`.
        """
        return self.normalize_vect(vector, out=out)

    def untransform_vect(
        self,
        vector: ndarray,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        """Map a vector with components in :math:`[0,1]` to the design space.

        Args:
            vector: A vector with components in :math:`[0,1]`.
            no_check: Whether to check if the components are in :math:`[0,1]`.
            out: The array to store the untransformed vector.
                If None, create a new array.

        Returns:
            A point of the variables space.
        """
        return self.unnormalize_vect(vector, no_check=no_check, out=out)

    def round_vect(
        self,
        x_vect: ndarray,
        copy: bool = True,
    ) -> ndarray:
        """Round the vector where variables are of integer type.

        Args:
            x_vect: The values to be rounded.
            copy: Whether to round a copy of ``x_vect``.

        Returns:
            The rounded values.
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()

        if self.__no_integer:
            return x_vect

        if copy:
            rounded_x_vect = x_vect.copy()
        else:
            rounded_x_vect = x_vect

        are_integers = self.__integer_components
        rounded_x_vect[..., are_integers] = np_round(x_vect[..., are_integers])
        return rounded_x_vect

    def set_current_value(
        self,
        value: ndarray | Mapping[str, ndarray] | OptimizationResult,
    ) -> None:
        """Set the current design value.

        Args:
            value: The value of the current design.

        Raises:
            ValueError: If the value has a wrong dimension.
            TypeError: If the value is neither a mapping of NumPy arrays,
                a NumPy array nor an :class:`.OptimizationResult`.
        """
        if isinstance(value, dict):
            self.__current_value = value
        elif isinstance(value, ndarray):
            if value.size != self.dimension:
                raise ValueError(
                    "Invalid current_x, "
                    "dimension mismatch: {} != {}.".format(self.dimension, value.size)
                )
            self.__current_value = self.array_to_dict(value)
        elif isinstance(value, OptimizationResult):
            if value.x_opt.size != self.dimension:
                raise ValueError(
                    "Invalid x_opt, "
                    "dimension mismatch: {} != {}.".format(
                        self.dimension, value.x_opt.size
                    )
                )
            self.__current_value = self.array_to_dict(value.x_opt)
        else:
            raise TypeError(
                "The current design value should be either an array, "
                "a dictionary of arrays "
                "or an optimization result; "
                f"got {type(value)} instead."
            )

        for name, value in self.__current_value.items():
            if value is not None:
                variable_type = self.variables_types[name]
                if isinstance(variable_type, ndarray):
                    variable_type = variable_type[0]
                if variable_type == DesignVariableType.INTEGER.value:
                    value = value.astype(self.__VARIABLE_TYPES_TO_DTYPES[variable_type])
                self.__current_value[name] = value

        self.__update_current_metadata()
        if self.__current_value:
            self._check_current_names()

    def set_current_variable(
        self,
        name: str,
        current_value: ndarray,
    ) -> None:
        """Set the current value of a single variable.

        Args:
            name: The name of the variable.
            current_value: The current value of the variable.
        """
        if name in self.variables_names:
            self.__current_value[name] = current_value
            self.__update_current_metadata()
        else:
            raise ValueError(f"Variable '{name}' is not known.")

    def get_size(
        self,
        name: str,
    ) -> int | None:
        """Get the size of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The size of the variable, None if it is not known.
        """
        return self.variables_sizes.get(name, None)

    def get_type(
        self,
        name: str,
    ) -> str | None:
        """Return the type of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The type of the variable, None if it is not known.
        """
        return self.variables_types.get(name, None)

    def get_lower_bound(
        self,
        name: str,
    ) -> ndarray:
        """Return the lower bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite).
        """
        return self._lower_bounds.get(name)

    def get_upper_bound(
        self,
        name: str,
    ) -> ndarray:
        """Return the upper bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite).
        """
        return self._upper_bounds.get(name)

    def get_lower_bounds(
        self,
        variable_names: Sequence[str] | None = None,
    ) -> ndarray:
        """Generate an array of the variables' lower bounds.

        Args:
            variable_names: The names of the variables
                of which the lower bounds are required.
                If None, use the variables of the design space.

        Returns:
            The lower bounds of the variables.
        """
        if self.__norm_data_is_computed and variable_names is None:
            return self.__lower_bounds_array
        return self.dict_to_array(self._lower_bounds, variable_names=variable_names)

    def get_upper_bounds(
        self,
        variable_names: Sequence[str] | None = None,
    ) -> ndarray:
        """Generate an array of the variables' upper bounds.

        Args:
            variable_names: The names of the variables
                of which the upper bounds are required.
                If None, use the variables of the design space.

        Returns:
            The upper bounds of the variables.
        """
        if self.__norm_data_is_computed and variable_names is None:
            return self.__upper_bounds_array
        return self.dict_to_array(self._upper_bounds, variable_names=variable_names)

    def set_lower_bound(
        self,
        name: str,
        lower_bound: ndarray | None,
    ) -> None:
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The value of the lower bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        if name not in self.variables_names:
            raise ValueError(f"Variable '{name}' is not known.")

        self._add_bound(name, self.variables_sizes[name], lower_bound)
        self._add_norm_policy(name)

    def set_upper_bound(
        self,
        name: str,
        upper_bound: ndarray | None,
    ) -> None:
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The value of the upper bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        if name not in self.variables_names:
            raise ValueError(f"Variable '{name}' is not known.")

        self._add_bound(name, self.variables_sizes[name], upper_bound, is_lower=False)
        self._add_norm_policy(name)

    def array_to_dict(
        self,
        x_array: ndarray,
    ) -> dict[str, ndarray]:
        """Convert a design array into a dictionary indexed by the variables names.

        Args:
            x_array: A design value expressed as a NumPy array.

        Returns:
            The design value expressed as a dictionary of NumPy arrays.
        """
        return split_array_to_dict_of_arrays(
            x_array,
            self.variables_sizes,
            self.variables_names,
        )

    @classmethod
    def __get_common_dtype(
        cls,
        x_dict: Mapping[str, ndarray],
    ) -> dtype:
        """Return the common data type.

        Use the following rules by parsing the values `x_dict`:

        - there is a complex value: returns `numpy.complex128`,
        - there are real and mixed float/int values: returns `numpy.float64`,
        - there are only integer values: returns `numpy.int32`.

        Args:
            x_dict: The values to be parsed.

        Raises:
            TypeError: If the values of the data dictionary are not NumPy arrays.
        """
        has_float = False
        has_int = False
        for val_arr in x_dict.values():
            if not isinstance(val_arr, ndarray):
                raise TypeError("x_dict values must be ndarray.")

            if val_arr.dtype.kind == "c":
                return cls.__COMPLEX_DTYPE

            if val_arr.dtype.kind == "i":
                has_int = True

            if val_arr.dtype.kind == "f":
                has_float = True

        if has_float:
            return cls.__FLOAT_DTYPE

        if has_int:
            return cls.__INT_DTYPE

        return cls.__FLOAT_DTYPE

    def dict_to_array(
        self,
        design_values: dict[str, ndarray],
        variable_names: Iterable[str] = None,
    ) -> ndarray:
        """Convert a point as dictionary into an array.

        Args:
            design_values: The design point to be converted.
            variable_names: The variables to be considered.
                If None, use the variables of the design space.

        Returns:
            The point as an array.
        """
        if variable_names is None:
            variable_names = self.variables_names

        data = [design_values[name] for name in variable_names]
        return hstack(data).astype(self.__get_common_dtype(design_values))

    def get_pretty_table(
        self,
        fields: Sequence[str] | None = None,
        with_index: bool = False,
    ) -> PrettyTable:
        """Build a tabular view of the design space.

        Args:
            fields: The name of the fields to be exported.
                If ``None``, export all the fields.
            with_index: Whether to show index of names for arrays.
                This is ignored for scalars.

        Returns:
            A tabular view of the design space.
        """
        if fields is None:
            fields = self.TABLE_NAMES
        table = PrettyTable(fields)
        table.float_format = "%.16g"
        for name in self.variables_names:
            size = self.variables_sizes[name]
            l_b = self._lower_bounds.get(name)
            u_b = self._upper_bounds.get(name)
            var_type = self.variables_types[name]
            curr = self.__current_value.get(name)

            name_template = "{name}"
            if with_index and size > 1:
                name_template += "[{index}]"

            for i in range(size):
                data = {
                    "name": name_template.format(name=name, index=i),
                    "value": None,
                    "lower_bound": float("-inf"),
                    "upper_bound": float("inf"),
                    "type": var_type[i],
                }
                if l_b is not None and l_b[i] is not None:
                    data["lower_bound"] = l_b[i]
                if u_b is not None and u_b[i] is not None:
                    data["upper_bound"] = u_b[i]
                if curr is not None:
                    value = curr[i]
                    # The current value of a float variable can be a complex array
                    # when approximating gradients with complex step.
                    if var_type[i] == "float":
                        value = value.real
                    data["value"] = value
                table.add_row([data[key] for key in fields])

        table.align["name"] = "l"
        table.align["type"] = "l"
        return table

    def export_hdf(
        self,
        file_path: str | Path,
        append: bool = False,
    ) -> None:
        """Export the design space to an HDF file.

        Args:
            file_path: The path to the file to export the design space.
            append: If True, appends the data in the file.
        """
        mode = "a" if append else "w"

        with h5py.File(file_path, mode) as h5file:
            design_vars_grp = h5file.require_group(self.DESIGN_SPACE_GROUP)
            design_vars_grp.create_dataset(
                self.NAMES_GROUP, data=array(self.variables_names, dtype=string_)
            )

            for name in self.variables_names:
                var_grp = design_vars_grp.require_group(name)
                var_grp.create_dataset(self.SIZE_GROUP, data=self.variables_sizes[name])

                l_b = self._lower_bounds.get(name)
                if l_b is not None:
                    var_grp.create_dataset(self.LB_GROUP, data=l_b)

                u_b = self._upper_bounds.get(name)
                if u_b is not None:
                    var_grp.create_dataset(self.UB_GROUP, data=u_b)

                var_type = self.variables_types[name]
                if var_type is not None:
                    data_array = array(var_type, dtype="bytes")
                    var_grp.create_dataset(
                        self.VAR_TYPE_GROUP,
                        data=data_array,
                        dtype=data_array.dtype,
                    )

                value = self.__current_value.get(name)
                if value is not None:
                    var_grp.create_dataset(self.VALUE_GROUP, data=self.__to_real(value))

    def import_hdf(
        self,
        file_path: str | Path,
    ) -> None:
        """Import a design space from an HDF file.

        Args:
            file_path: The path to the file
                containing the description of a design space.
        """
        with h5py.File(file_path) as h5file:
            design_vars_grp = get_hdf5_group(h5file, self.DESIGN_SPACE_GROUP)
            variable_names = get_hdf5_group(design_vars_grp, self.NAMES_GROUP)

            for name in variable_names:
                name = name.decode()
                var_group = get_hdf5_group(design_vars_grp, name)
                l_b = self.__read_opt_attr_array(var_group, self.LB_GROUP)
                u_b = self.__read_opt_attr_array(var_group, self.UB_GROUP)
                var_type = self.__read_opt_attr_array(var_group, self.VAR_TYPE_GROUP)
                value = self.__read_opt_attr_array(var_group, self.VALUE_GROUP)
                size = get_hdf5_group(var_group, self.SIZE_GROUP)[()]
                self.add_variable(name, size, var_type, l_b, u_b, value)

        self.check()

    @staticmethod
    def __read_opt_attr_array(
        var_group: h5py.Group,
        dataset_name: str,
    ) -> ndarray | None:
        """Read data in a group.

        Args:
            var_group: The variable group.
            dataset_name: The name of the dataset.

        Returns:
            The data found in the group, if it exists.
            Otherwise, None.
        """
        data = var_group.get(dataset_name)
        if data is not None:
            data = array(data)
        return data

    @staticmethod
    def __to_real(
        data: ndarray,
    ) -> ndarray:
        """Convert complex to real NumPy array.

        Args:
            data: A complex NumPy array.

        Returns:
            A real NumPy array.
        """
        return array(array(data, copy=False).real, dtype=float64)

    def to_complex(self) -> None:
        """Cast the current value to complex."""
        for name, val in self.__current_value.items():
            self.__current_value[name] = array(val, dtype=complex128)

        self.__update_common_dtype()

    def export_to_txt(
        self,
        output_file: str | Path,
        fields: Sequence[str] | None = None,
        header_char: str = "",
        **table_options: Any,
    ) -> None:
        """Export the design space to a text file.

        Args:
            output_file: The path to the file.
            fields: The fields to be exported.
                If None, export all fields.
            header_char: The header character.
            **table_options: The names and values of additional attributes
                for the :class:`.PrettyTable` view
                generated by :meth:`.DesignSpace.get_pretty_table`.
        """
        output_file = Path(output_file)
        table = self.get_pretty_table(fields=fields)
        table.border = False
        for option, val in table_options.items():
            table.__setattr__(option, val)
        with output_file.open("w") as outf:
            table_str = header_char + table.get_string()
            outf.write(table_str)

    @staticmethod
    def read_from_txt(
        input_file: str | Path,
        header: Iterable[str] | None = None,
    ) -> DesignSpace:
        """Create a design space from a text file.

        Args:
            input_file: The path to the file.
            header: The names of the fields saved in the file.
                If None, read them in the file.

        Returns:
            The design space read from the file.

        Raises:
            ValueError: If the file does not contain the minimal variables
                in its header.
        """
        float_data = genfromtxt(input_file, dtype="float")
        str_data = genfromtxt(input_file, dtype="str")
        if header is None:
            header = str_data[0, :].tolist()
            start_read = 1
        else:
            start_read = 0
        if not set(DesignSpace.MINIMAL_FIELDS).issubset(set(header)):
            raise ValueError(
                "Malformed DesignSpace input file {} does not contain "
                "minimal variables in header:"
                "{}; got instead: {}.".format(
                    input_file, DesignSpace.MINIMAL_FIELDS, header
                )
            )
        col_map = {field: i for i, field in enumerate(header)}
        var_names = str_data[start_read:, 0].tolist()
        unique_names = []
        prev_name = None
        for name in var_names:  # set([]) does not preserve order !
            if name not in unique_names:
                unique_names.append(name)
                prev_name = name
            elif prev_name != name:
                raise ValueError(
                    f"Malformed DesignSpace input file {input_file} contains some "
                    f"variables ({name}) in a non-consecutive order."
                )

        k = start_read
        design_space = DesignSpace()
        lower_bounds_field = DesignSpace.MINIMAL_FIELDS[1]
        upper_bounds_field = DesignSpace.MINIMAL_FIELDS[2]
        value_field = DesignSpace.TABLE_NAMES[2]
        var_type_field = DesignSpace.TABLE_NAMES[-1]
        for name in unique_names:
            size = var_names.count(name)
            l_b = float_data[k : k + size, col_map[lower_bounds_field]]
            u_b = float_data[k : k + size, col_map[upper_bounds_field]]
            if value_field in col_map:
                value = float_data[k : k + size, col_map[value_field]]
                if "None" in str_data[k : k + size, col_map[value_field]]:
                    value = None
            else:
                value = None
            if var_type_field in col_map:
                var_type = str_data[k : k + size, col_map[var_type_field]].tolist()
            else:
                var_type = DesignVariableType.FLOAT
            design_space.add_variable(name, size, var_type, l_b, u_b, value)
            k += size
        design_space.check()
        return design_space

    def __str__(self) -> str:
        return (
            f"Design space: {self.name or ''}\n"
            + self.get_pretty_table(with_index=True).get_string()
        )

    def project_into_bounds(
        self,
        x_c: ndarray,
        normalized: bool = False,
    ) -> ndarray:
        """Project a vector onto the bounds, using a simple coordinate wise approach.

        Args:
            normalized: If True, then the vector is assumed to be normalized.
            x_c: The vector to be projected onto the bounds.

        Returns:
            The projected vector.
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        if not normalized:
            l_b = self.__lower_bounds_array
            u_b = self.__upper_bounds_array
        else:
            l_b = zeros_like(x_c)
            u_b = ones_like(x_c)
        x_p = array(x_c)
        l_inds = where(x_c < l_b)
        x_p[l_inds] = l_b[l_inds]

        u_inds = where(x_c > u_b)
        x_p[u_inds] = u_b[u_inds]
        return x_p

    def __contains__(
        self,
        variable: str,
    ) -> bool:
        return variable in self.variables_names

    def __len__(self) -> int:
        return len(self.variables_names)

    def __iter__(self) -> Iterable[str]:
        return iter(self.variables_names)

    def __setitem__(
        self,
        name: str,
        item: DesignVariable,
    ) -> None:
        self.add_variable(
            name,
            size=item.size,
            var_type=item.var_type,
            l_b=item.l_b,
            u_b=item.u_b,
            value=item.value,
        )

    def __eq__(
        self,
        other: DesignSpace,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        if len(other) != len(self):
            return False

        for key, val in self.items():
            if key not in other:
                return False

            hash1 = hash_data_dict(flatten_nested_dict(val._asdict()))
            hash2 = hash_data_dict(flatten_nested_dict(other[key]._asdict()))
            if hash1 != hash2:
                return False

        return True

    def __getitem__(
        self,
        name: str,
    ) -> DesignVariable:
        """Return the data associated with a given variable.

        These data are: type, size, lower bound, upper bound and current value.

        Args:
            name: The name of the variable.

        Returns:
            The data associated with the variable.

        Raises:
            ValueError: If the variable name does not exist.
        """
        if name not in self.variables_names:
            raise KeyError(f"Variable '{name}' is not known.")

        try:
            value = self.get_current_value([name])
        except KeyError:
            value = None

        return DesignVariable(
            size=self.get_size(name),
            var_type=self.get_type(name),
            l_b=self.get_lower_bound(name),
            u_b=self.get_upper_bound(name),
            value=value,
        )

    def extend(
        self,
        other: DesignSpace,
    ) -> None:
        """Extend the design space with another design space.

        Args:
            other: The design space to be appended to the current one.
        """
        for name in other.variables_names:
            size = other.get_size(name)
            var_type = other.get_type(name)
            l_b = other.get_lower_bound(name)
            u_b = other.get_upper_bound(name)
            value = other.get_current_value(as_dict=True)[name]
            self.add_variable(name, size, var_type, l_b, u_b, value)

    @staticmethod
    def __cast_array_to_list(
        value: str | int | ndarray,
    ) -> str | int | list[str | int]:
        """Convert a value to a ``List`` if it is a NumPy array.

        Args:
            value: The value to be cast.

        Returns:
            Either the original value or the NumPy array converted to a ``List``.
        """
        return value if not isinstance(value, ndarray) else value.tolist()

    @classmethod
    def __cast_mapping(
        cls,
        mapping: Mapping[str, str | int | ndarray],
    ) -> dict[str, str | int | list[str | int]]:
        """Convert the NumPy arrays of a mapping to ``List``.

        Args:
            mapping: The value to be cast.

        Returns:
            The original mapping with NumPy values converted to a ``List``.
        """
        return {
            key: {
                sub_key: cls.__cast_array_to_list(sub_val)
                for sub_key, sub_val in val.items()
            }
            for key, val in mapping.items()
        }

    def rename_variable(
        self,
        current_name: str,
        new_name: str,
    ) -> None:
        """Rename a variable.

        Args:
            current_name: The name of the variable to rename.
            new_name: The new name of the variable.
        """
        if current_name not in self.variables_names:
            raise ValueError(f"The variable {current_name} is not in the design space.")

        self.variables_names[self.variables_names.index(current_name)] = new_name
        for dictionary in [
            self.variables_sizes,
            self.variables_types,
            self.normalize,
            self._lower_bounds,
            self._upper_bounds,
            self._current_value,
        ]:
            dictionary[new_name] = dictionary.pop(current_name)

    def initialize_missing_current_values(self) -> None:
        """Initialize the current values of the design variables when missing.

        Use:

        - the center of the design space when the lower and upper bounds are finite,
        - the lower bounds when the upper bounds are infinite,
        - the upper bounds when the lower bounds are infinite,
        - zero when the lower and upper bounds are infinite.
        """
        for name, value in self.items():
            if value.value is not None:
                continue

            current_value = []
            for l_b_i, u_b_i in zip(value.l_b, value.u_b):
                if l_b_i == -inf:
                    if u_b_i == inf:
                        current_value_i = 0
                    else:
                        current_value_i = u_b_i
                else:
                    if u_b_i == inf:
                        current_value_i = l_b_i
                    else:
                        current_value_i = (l_b_i + u_b_i) / 2

                current_value.append(current_value_i)

            if self.FLOAT.value in value.var_type:
                var_type = self.FLOAT.value
            else:
                var_type = self.INTEGER.value

            self.set_current_variable(
                name,
                array(
                    current_value,
                    dtype=self.__VARIABLE_TYPES_TO_DTYPES[var_type],
                ),
            )
