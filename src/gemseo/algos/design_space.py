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

from __future__ import division, unicode_literals

import collections
import logging
import sys
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import h5py
from numpy import abs as np_abs
from numpy import (
    array,
    atleast_1d,
    complex128,
    concatenate,
    empty,
    equal,
    finfo,
    float64,
    genfromtxt,
    inf,
    int32,
    isnan,
    logical_or,
    mod,
    ndarray,
    ones,
    ones_like,
)
from numpy import round_ as np_round
from numpy import string_, vectorize, where, zeros_like
from six import string_types

from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.cache import hash_data_dict
from gemseo.third_party.prettytable import PrettyTable
from gemseo.utils.base_enum import BaseEnum
from gemseo.utils.data_conversion import flatten_mapping
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.py23_compat import Path, string_array, strings_to_unicode_list

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

if sys.version_info < (3, 7, 0):
    DesignVariable = collections.namedtuple(
        "DesignVariable", ["size", "var_type", "l_b", "u_b", "value"]
    )
    DesignVariable.__new__.__defaults__ = (
        1,
        DesignVariableType.FLOAT,
        None,
        None,
        None,
    )
else:
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


class DesignSpace(collections.MutableMapping):
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

    Attributes:
        name (Optional[str]): The name of the space.
        variables_names (List[str]): The names of the variables.
        dimension (int): The total dimension of the space,
            corresponding to the sum of the sizes of the variables.
        variables_sizes (Dict[str,int]): The sizes of the variables.
        variables_types (Dict[str,ndarray]): The types of the variables components,
            which can be any :attr:`DesignVariableType`.
        normalize (Dict[str,ndarray]): The normalization policies
            of the variables components indexed by the variables names;
            if `True`, the component can be normalized.
    """

    FLOAT = DesignVariableType.FLOAT
    INTEGER = DesignVariableType.INTEGER
    AVAILABLE_TYPES = [FLOAT, INTEGER]
    __TYPE_NAMES = tuple(x.value for x in DesignVariableType)
    __TYPES_TO_DTYPES = {
        DesignVariableType.FLOAT.value: "float64",
        DesignVariableType.INTEGER.value: "int32",
    }
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

    def __init__(
        self,
        hdf_file=None,  # type: Optional[Union[str,Path]]
        name=None,  # type: Optional[str]
    ):  # type: (...) -> None
        """
        Args:
            hdf_file: The path to the file
                containing the description of an initial design space.
                If None, start with an empty design space.
            name: The name to be given to the design space,
                `None` if the design space is unnamed.
        """
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
        self.__lower_bounds_array = None
        self.__upper_bounds_array = None
        self.__int_vars_indices = None
        self.__norm_data_is_computed = False
        self.__norm_inds = None
        self.__to_zero = None
        self.__bound_tol = 100.0 * finfo(float64).eps
        self._current_x = {}
        if hdf_file is not None:
            self.import_hdf(hdf_file)

    def __delitem__(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        self.remove_variable(name)

    def remove_variable(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        self.__norm_data_is_computed = False
        size = self.variables_sizes[name]
        self.dimension -= size
        self.variables_names.remove(name)
        del self.variables_sizes[name]
        del self.variables_types[name]
        del self.normalize[name]
        if name in self._lower_bounds:
            del self._lower_bounds[name]
        if name in self._upper_bounds:
            del self._upper_bounds[name]
        if name in self._current_x:
            del self._current_x[name]

    def filter(
        self,
        keep_variables,  # type: Union[str,Iterable[str]]
        copy=False,  # type: bool
    ):  # type: (...) -> DesignSpace
        """Filter the design space to keep a subset of variables.

        Args:
            keep_variables: The names of the variables to be kept.
            copy: If True, then a copy of the design space is filtered,
                otherwise the design space itself is filtered.

        Returns:
            Either the filtered original design space or a copy.

        Raises:
            ValueError: If the variable is not in the design space.
        """
        if isinstance(keep_variables, string_types):
            keep_variables = [keep_variables]
        design_space = deepcopy(self) if copy else self
        for name in deepcopy(self.variables_names):
            if name not in keep_variables:
                design_space.remove_variable(name)
        for name in keep_variables:
            if name not in self.variables_names:
                raise ValueError("Variable '{}' is not known.".format(name))
        return design_space

    def filter_dim(
        self,
        variable,  # type: str
        keep_dimensions,  # type: Iterable[int]
    ):  # type: (...) -> DesignSpace
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
        if variable in self._current_x:
            self._current_x[variable] = self._current_x[variable][idx]
        return self

    def add_variable(
        self,
        name,  # type: str
        size=1,  # type: int
        var_type=DesignVariableType.FLOAT,  # type: VarType
        l_b=None,  # type: Optional[Union[float,ndarray]]
        u_b=None,  # type: Optional[Union[float,ndarray]]
        value=None,  # type:  Optional[Union[float,ndarray]]
    ):  # type: (...) -> None
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
            raise ValueError("Variable '{}' already exists.".format(name))

        if size <= 0 or int(size) != size:
            raise ValueError(
                "The size of '{}' should be a positive integer.".format(name)
            )

        # name and size
        self.variables_names.append(name)
        self.dimension += size
        self.variables_sizes[name] = size
        # type
        self._add_type(name, size, var_type)

        # bounds
        self._add_bound(name, size, l_b, is_lower=True)
        self._add_bound(name, size, u_b, is_lower=False)
        self._check_variable_bounds(name)

        # normalization policy
        self._add_norm_policy(name)

        if value is not None:
            array_value = atleast_1d(value)
            self._check_value(array_value, name)
            if len(array_value) == 1 and size > 1:
                array_value = array_value * ones(size)
            self._current_x[name] = array_value
            self._check_current_x_value(name)

    def _add_type(
        self,
        name,  # type: str
        size,  # type: int
        var_type=DesignVariableType.FLOAT,  # type: VarType
    ):  # type: (...) -> None
        """Add a type to a variable.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            var_type: Either the type of the variable (see :attr:`AVAILABLE_TYPES`)
                or the types of its components.

        Raises:
            ValueError: Either if the number of component types is different
                from the variable size or if a variable type is unknown.
        """
        if not hasattr(var_type, "__iter__") or isinstance(
            var_type, (string_types, DesignVariableType, bytes)
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

            if isinstance(v_type, string_types) and v_type not in self.__TYPE_NAMES:
                msg = 'The type "{0}" of {1} is not known.'.format(v_type, name)
                raise ValueError(msg)
            elif v_type in DesignVariableType:
                v_type = v_type.value

            var_types += [v_type]

        self.variables_types[name] = array(var_types)
        self.__norm_data_is_computed = False

    def _add_norm_policy(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Add a normalization policy to a variable.

        Unbounded variables are not normalized.
        Bounded variables (both from above and from below) are normalized.

        Args:
            name: The name of a variable.

        Raises:
            ValueError: Either if the variable is not in the design space,
                it its size is not set,
                if the types of its components are not set
                or if there is no implemented normalization policy
                for the type of this variable.
        """
        # Check that the variable is in the design space:
        if name not in self.variables_names:
            raise ValueError("Variable '{}' is not known.".format(name))
        # Check that the variable size is set:
        size = self.get_size(name)
        if size is None:
            raise ValueError("The size of variable '{}' is not set.".format(name))
        # Check that the variables types are set:
        variables_types = self.variables_types.get(name, None)
        if variables_types is None:
            raise ValueError(
                "The components types of variable '{}' are not set.".format(name)
            )
        # Set the normalization policy:
        normalize = empty(size)
        for i in range(size):
            var_type = variables_types[i]
            if var_type in self.__TYPES_TO_DTYPES:
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
        value,  # type: ndarray
    ):  # type: (...) -> bool
        """Check that all values in an array are integers.

        Args:
            value: The array to be checked.

        Returns:
            Whether all values in the array are integers.
        """
        are_none = equal(value, None)
        are_int = equal(mod(value.astype("f"), 1), 0)
        return logical_or(are_none, are_int)

    @staticmethod
    def __is_numeric(
        value,  # type: Any
    ):  # type: (...) -> bool
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
        value,  # type: ndarray
    ):  # type: (...) -> bool
        """Check that a value is not a nan.

        Args:
            value: The value to be checked.

        Returns:
            Whether the value is not a nan.
        """
        return (value is None) or ~isnan(value)

    def _check_value(
        self,
        value,  # type: ndarray
        name,  # type: str
    ):  # type: (...) -> bool
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
                while the variable type is integer.
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
                "Value {} of variable '{}' is not numerizable.".format(value[idx], name)
            )

        test = vectorize(self.__is_not_nan)(value)
        indices = all_indices - set(list(where(test)[0]))
        for idx in indices:
            raise ValueError(
                "Value {} of variable '{}' is nan.".format(value[idx], name)
            )

        # Check if some components of an integer variable are not integer.
        if self.variables_types[name][0] == DesignVariableType.INTEGER.value:
            indices = all_indices - set(list(where(self.__is_integer(value))[0]))
            for idx in indices:
                raise ValueError(
                    "Component value {} of variable '{}' is not an integer "
                    "while variable is of type integer "
                    "(index: {}).".format(value[idx], name, idx)
                )

    def _add_bound(
        self,
        name,  # type: str
        size,  # type: int
        bound,  # type: ndarray
        is_lower=True,  # type: bool
    ):  # type: (...) -> None
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
            bound_dict = self._lower_bounds
        else:
            bound_dict = self._upper_bounds
        infinity = inf * ones(size)
        if bound is None:
            if is_lower:
                bound_array = -infinity
            else:
                bound_array = infinity
            bound_dict.update({name: bound_array})
            return
        bound_array = atleast_1d(bound)
        self._check_value(bound_array, name)
        if hasattr(bound, "__iter__"):
            # iterable structure
            if len(bound_array) != size:
                bound_str = "lower" if is_lower else "upper"
                raise ValueError(
                    "The {} bounds of '{}' should be of size {}.".format(
                        bound_str, name, size
                    )
                )
            if is_lower:
                bound_array = where(equal(bound_array, None), -infinity, bound_array)
            else:
                bound_array = where(equal(bound_array, None), infinity, bound_array)
            bound_dict.update({name: bound_array})
        else:
            # scalar: same lower bound for all components
            bound_array = bound * ones(size)
            bound_dict.update({name: bound_array})
        return

    def _check_variable_bounds(
        self,
        name,  # type: str
    ):  # type: (...) -> None
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
                    name, inds, u_b[inds], l_b[inds]
                )
            )

    def _check_current_x_value(
        self,
        name,  # type: str
    ):  # type: (...) -> None
        """Check that the current value of a variable is between its bounds.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the current value of the variable is outside its bounds.
        """
        l_b = self._lower_bounds.get(name, None)
        u_b = self._upper_bounds.get(name, None)
        c_x = self._current_x.get(name, None)
        not_none = ~equal(c_x, None)
        inds = where(
            logical_or(
                c_x[not_none] < l_b[not_none] - 1e-14,
                c_x[not_none] > u_b[not_none] + 1e-14,
            )
        )[0]
        for idx in inds:
            raise ValueError(
                "The current value of variable '{}' ({}) is not "
                "between the lower bound {} and the upper bound {}.".format(
                    name, c_x[idx], l_b[idx], u_b[idx]
                )
            )

    def has_current_x(self):  # type: (...) -> bool
        """Check if the current design value is defined for all variables.

        Returns:
            Whether the current design value is defined for all variables.
        """
        if self._current_x is None or len(self._current_x) != len(self.variables_names):
            return False
        for val in self._current_x.values():
            if val is None:
                return False
        return True

    def check(self):  # type: (...) -> None
        """Check the state of the design space.

        Raises:
            ValueError: If the design space is empty.
        """
        if not self.variables_names:
            raise ValueError("The design space is empty.")

        for name in self.variables_names:
            self._check_variable_bounds(name)

        if self.has_current_x():
            self._check_current_x()

    def check_membership(
        self,
        x_vect,  # type: Union[Mapping[str,ndarray],ndarray]
        variables_names=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
        """Check whether the variables satisfy the design space requirements.

        Args:
            x_vect: The values of the variables.
            variables_names: The names of the variables.
                If None, use the names of the variables of the design space.

        Raises:
            ValueError: Either if the dimension of the values vector is wrong,
                if the values are not specified as an array or a dictionary,
                if the values are outside the bounds of the variables or
                if the component of an integer variable is an integer.
        """
        # Convert the input vector into a dictionary if necessary:
        if isinstance(x_vect, dict):
            x_dict = x_vect
        elif isinstance(x_vect, ndarray):
            if x_vect.size != self.dimension:
                raise ValueError(
                    "The dimension of the input array ({}) should be {}.".format(
                        x_vect.size, self.dimension
                    )
                )
            x_dict = self.array_to_dict(x_vect)
        else:
            raise TypeError(
                "The input vector should be an array or a dictionary; "
                "got {} instead.".format(type(x_vect))
            )
        # Check the membership of the input vector to the design space:
        variables_names = variables_names or self.variables_names
        for name in variables_names:
            if x_dict[name] is None:
                continue
            size = self.variables_sizes[name]
            l_b = self._lower_bounds.get(name, None)
            u_b = self._upper_bounds.get(name, None)

            if atleast_1d(x_dict[name]).size != size:
                raise ValueError(
                    "The component '{}' of the given array should have size {}.".format(
                        name, size
                    )
                )
            for i in range(size):
                x_real = atleast_1d(x_dict[name])[i].real
                if l_b is not None and x_real < l_b[i] - self.__bound_tol:
                    msg = (
                        "The component {}{}{} of the given array ({}) "
                        "is lower than the lower bound ({}) by {:.1e}.".format(
                            name, self.SEP, i, x_real, l_b[i], l_b[i] - x_real
                        )
                    )
                    raise ValueError(msg)
                if u_b is not None and u_b[i] + self.__bound_tol < x_real:
                    msg = (
                        "The component '{}'{}{} of the given array ({}) "
                        "is greater than the upper bound ({}) by {:.1e}.".format(
                            name, self.SEP, i, x_real, u_b[i], u_b[i] - x_real
                        )
                    )
                    raise ValueError(msg)
                if (
                    self.variables_types[name][0] == DesignVariableType.INTEGER.value
                ) and not self.__is_integer(x_real):
                    msg = (
                        "The component '{}'{}{} of the given array is not an integer "
                        "while variable is of type integer! Value = {}".format(
                            name, self.SEP, i, x_real
                        )
                    )
                    raise ValueError(msg)

    def get_active_bounds(
        self,
        x_vec=None,  # type: Optional[ndarray]
        tol=1e-8,  # type: float
    ):  # type: (...) -> Tuple[Dict[str,ndarray],Dict[str,ndarray]]
        """Determine which bound constraints of the current point are active.

        Args:
            x_vec: The point at which to check the bounds.
                If None, use the current point.
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
            x_dict = self._current_x
            self.check_membership(self.get_current_x())
        elif isinstance(x_vec, ndarray):
            x_dict = self.array_to_dict(x_vec)
        elif isinstance(x_vec, dict):
            x_dict = x_vec
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
            x_vec_i = x_dict[name]
            # lower bound saturated
            albi = where(np_abs(x_vec_i - l_b) <= tol, True, False)
            active_l_b[name] = albi
            # upper bound saturated
            aubi = where(np_abs(x_vec_i - u_b) <= tol, True, False)
            active_u_b[name] = aubi
        return active_l_b, active_u_b

    def _check_current_x(
        self,
        variables_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        """Check the names of the current point.

        Args:
            variables_names: The names of the variables.
                If None, use the names of the variables of the design space.

        Raises:
            ValueError: If the names of the variables of the current point
                and the names of the variables of the design space are different.
        """
        if sorted(set(self.variables_names)) != sorted(set(self._current_x.keys())):
            raise ValueError(
                "Expected current_x variables: {}; got {}.".format(
                    self.variables_names, list(self._current_x.keys())
                )
            )
        self.check_membership(self._current_x, variables_names)

    def get_current_x(
        self,
        variables_names=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> ndarray
        """Return the current point in the design space.

        Args:
            variables_names: The names of the required variables.
                If None, use the names of the variables of the design space.

        Raises:
            KeyError: If a variable has no current value.
        """
        try:
            x_arr = self.dict_to_array(self._current_x, all_var_list=variables_names)
            return x_arr
        except KeyError as err:
            raise KeyError(
                "The design space has no current value for '{}'.".format(err.args[0])
            )

    def get_indexed_var_name(
        self,
        variable_name,  # type: str
    ):  # type: (...) -> Union[str,List[str]]
        """Create the names of the components of a variable.

        If the size of the variable is equal to 1,
        this method returns the name of the variable.
        Otherwise,
        it concatenates the name of the variable,
        the separator :attr:`SEP` and the index of the component.

        Args:
            variable_name: The name of the variable.

        Returns:
            The names of the components of the variable.
        """
        size = self.variables_sizes[variable_name]
        if size == 1:
            return variable_name
        return [variable_name + self.SEP + str(i) for i in range(size)]

    def get_indexed_variables_names(self):  # type: (...) -> List[str]
        """Create the names of the components of all the variables.

        If the size of the variable is equal to 1,
        this method uses its name.
        Otherwise,
        it concatenates the name of the variable,
        the separator :attr:`SEP` and the index of the component.

        Returns:
            The name of the components of all the variables.
        """
        var_ind_names = []
        for var in self.variables_names:
            vnames = self.get_indexed_var_name(var)
            if isinstance(vnames, string_types):
                var_ind_names.append(vnames)
            else:
                var_ind_names += vnames
        return var_ind_names

    def get_variables_indexes(
        self,
        variables_names,  # type: Iterable[str]
    ):  # type: (...) -> ndarray
        """Return the indexes of a design array corresponding to the variables names.

        Args:
            variables_names: The names of the variables.

        Returns:
            The indexes of a design array corresponding to the variables names.
        """
        indexes = list()
        index = 0
        for name in self.variables_names:
            var_size = self.get_size(name)
            if name in variables_names:
                indexes.extend(range(index, index + var_size))
            index += var_size
        return array(indexes)

    def __update_normalization_vars(self):  # type: (...) -> None
        """Compute the inner attributes used for normalization and unnormalization."""

        self.__lower_bounds_array = self.get_lower_bounds()
        self.__upper_bounds_array = self.get_upper_bounds()
        self._norm_factor = self.__upper_bounds_array - self.__lower_bounds_array
        norm_array = self.dict_to_array(self.normalize)
        self.__norm_inds = where(norm_array)[0]
        # In case lb=ub
        self.__to_zero = where(self._norm_factor == 0.0)[0]

        var_ind_list = []
        for var in self.variables_names:
            # Store the mask of int variables
            to_one = self.variables_types[var] == DesignVariableType.INTEGER.value
            var_ind_list.append(to_one)
        self.__int_vars_indices = concatenate(var_ind_list)
        self.__norm_data_is_computed = True

    def normalize_vect(
        self,
        x_vect,  # type: ndarray
        minus_lb=True,  # type: bool
    ):  # type: (...) -> ndarray
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

        Returns:
            The normalized vector.

        Raises:
            ValueError: If the array to be normalized is not one- or two-dimensional.
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()

        # Normalize the relevant components:
        if self.has_current_x():
            current_dtype = self.get_current_x().dtype
        else:
            current_dtype = float
        norm_vect = x_vect.astype(current_dtype, copy=True)

        # In case lb=ub
        norm_inds = self.__norm_inds

        if len(x_vect.shape) == 1:
            if minus_lb:
                norm_vect[norm_inds] -= self.__lower_bounds_array[norm_inds]
            norm_vect[norm_inds] /= self._norm_factor[norm_inds]
            # In case lb=ub put value to 0
        elif len(x_vect.shape) == 2:
            if minus_lb:
                norm_vect[:, norm_inds] -= self.__lower_bounds_array[norm_inds]
            norm_vect[:, norm_inds] /= self._norm_factor[norm_inds]
        else:
            raise ValueError("The array to be normalized must be 1d or 2d.")

        to_zero = self.__to_zero
        if to_zero.size > 0:
            norm_vect[to_zero] = 0.0
        return norm_vect

    def normalize_grad(
        self,
        g_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
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
        g_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
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
        x_vect,  # type: ndarray
        minus_lb=True,  # type: bool
        no_check=False,  # type: bool
    ):  # type: (...) -> ndarray
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
            minus_lb: If True, remove the lower bounds at normalization.
            no_check: If True, do not check that the values are in [0,1].

        Returns:
            The unnormalized vector.

        Raises:
            ValueError: If the array to be unnormalized is not one- or two-dimensional.
        """
        n_dims = x_vect.ndim
        if n_dims not in [1, 2]:
            raise ValueError(
                "The array to be unnormalized must be 1d or 2d, got {}d.".format(n_dims)
            )
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        norm_inds = self.__norm_inds
        l_bounds = self.__lower_bounds_array
        if not no_check:
            # Check whether the input components are between 0 and 1:
            if n_dims == 1:
                bounded = x_vect[norm_inds]
            else:
                bounded = x_vect[:, norm_inds]
            if (bounded < -1e-12).any() or (bounded > 1 + 1e-12).any():
                msg = (
                    "All components of the normalized vector should be between 0 and 1."
                )
                lb_viol = bounded[bounded < -1e-12]
                if lb_viol.size != 0:
                    msg += " Lower bounds violated: {}.".format(lb_viol)
                ub_viol = bounded[bounded > 1 + 1e-12]
                if ub_viol.size != 0:
                    msg += " Upper bounds violated: {}.".format(ub_viol)

                LOGGER.warning(msg)
        # Unnormalize the relevant components:
        if self.has_current_x():
            current_dtype = self.get_current_x().dtype
        else:
            current_dtype = float
        unnorm_vect = x_vect.astype(current_dtype, copy=True)

        if n_dims == 1:
            unnorm_vect[norm_inds] *= self._norm_factor[norm_inds]
            if minus_lb:
                unnorm_vect[norm_inds] += l_bounds[norm_inds]
        else:
            unnorm_vect[:, norm_inds] *= self._norm_factor[norm_inds]
            if minus_lb:
                unnorm_vect[:, norm_inds] += l_bounds[norm_inds]
        inds_fixed = self.__to_zero
        if inds_fixed.size > 0:
            if n_dims == 1:
                unnorm_vect[inds_fixed] = l_bounds[inds_fixed]
            else:
                unnorm_vect[:, inds_fixed] = l_bounds[inds_fixed]

        r_xvec = self.round_vect(unnorm_vect)
        return r_xvec

    def transform_vect(
        self, vector  # type: ndarray
    ):  # type:(...) -> ndarray
        """Map a point of the design space to a vector with components in :math:`[0,1]`.

        Args:
            vector: A point of the design space.

        Returns:
            A vector with components in :math:`[0,1]`.
        """
        return self.normalize_vect(vector)

    def untransform_vect(
        self, vector  # type: ndarray
    ):  # type:(...) -> ndarray
        """Map a vector with components in :math:`[0,1]` to the design space.

        Args:
            vector: A vector with components in :math:`[0,1]`.

        Returns:
            A point of the variables space.
        """
        return self.unnormalize_vect(vector)

    def round_vect(
        self,
        x_vect,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Round the vector where variables are of integer type.

        Args:
            x_vect: The values to be rounded.

        Returns:
            The rounded values values.

        Raises:
            ValueError: If the values is not a one- or two-dimensional
        """
        if not self.__norm_data_is_computed:
            self.__update_normalization_vars()
        int_vars = self.__int_vars_indices
        if not int_vars.any():
            return x_vect
        n_dims = len(x_vect.shape)
        rounded_x_vect = x_vect.copy()
        if n_dims == 1:
            rounded_x_vect[int_vars] = np_round(rounded_x_vect[int_vars])
        elif n_dims == 2:
            rounded_x_vect[:, int_vars] = np_round(rounded_x_vect[:, int_vars])
        else:
            raise ValueError(
                "The array to be unnormalized must be 1d or 2d; got {}d.".format(n_dims)
            )

        return rounded_x_vect

    def get_current_x_normalized(self):  # type: (...) -> ndarray
        """Return the current point normalized.

        Returns:
            The current point as an array normalized by the bounds of the variables.

        Returns:
            KeyError: If the current point cannot be normalized.
        """
        try:
            current_x = self.get_current_x()
        except KeyError as err:
            raise KeyError(
                "Cannot compute normalized current value since {}.".format(err.args[0])
            )
        return self.normalize_vect(current_x)

    def get_current_x_dict(self):  # type:(...) -> Dict[str,ndarray]
        """Return the current point in the design space as a dictionary.

        Returns:
            The current point in the design space as a dictionary,
            whose keys are the names of the variables
            and values are the values of the variables.
        """
        return self._current_x

    def set_current_x(
        self,
        current_x,  # type: Union[ndarray,Mapping[str,ndarray], OptimizationResult]
    ):  # type: (...) -> None
        """Set the current point.

        Args:
            current_x: The value of the current point.

        Raises:
            ValueError: If the value has a wrong dimension.
            TypeError: If the current point is neither a mapping of NumPy arrays,
                a NumPy array nor an :class:`.OptimizationResult`.
        """
        if isinstance(current_x, dict):
            self._current_x = current_x
        elif isinstance(current_x, ndarray):
            if current_x.size != self.dimension:
                raise ValueError(
                    "Invalid current_x, "
                    "dimension mismatch: {} != {}.".format(
                        self.dimension, current_x.size
                    )
                )
            self._current_x = self.array_to_dict(current_x)
        elif isinstance(current_x, OptimizationResult):
            if current_x.x_opt.size != self.dimension:
                raise ValueError(
                    "Invalid x_opt, "
                    "dimension mismatch: {} != {}.".format(
                        self.dimension, current_x.x_opt.size
                    )
                )
            self._current_x = self.array_to_dict(current_x.x_opt)
        else:
            raise TypeError(
                "The current point should be either an array, "
                "a dictionary of arrays "
                "or an optimization result; "
                "got {} instead.".format(type(current_x))
            )

        for name, value in self._current_x.items():
            if value is not None:
                variable_type = self.variables_types[name]
                if isinstance(variable_type, ndarray):
                    variable_type = variable_type[0]
                if variable_type == DesignVariableType.INTEGER.value:
                    value = value.astype(self.__TYPES_TO_DTYPES[variable_type])
                self._current_x[name] = value

        self._check_current_x()

    def set_current_variable(
        self,
        name,  # type: str
        current_value,  # type: ndarray
    ):
        """Set the current value of a single variable.

        Args:
            name: The name of the variable.
            current_value: The current value of the variable.
        """
        if name in self.variables_names:
            self._current_x[name] = current_value
        else:
            raise ValueError("Variable '{}' is not known.".format(name))

    def get_size(
        self,
        name,  # type: str
    ):  # type:(...) -> Optional[int]
        """Get the size of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The size of the variable, None if it is not known.
        """
        return self.variables_sizes.get(name, None)

    def get_type(
        self,
        name,  # type: str
    ):  # type:(...) -> Optional[str]
        """Return the type of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The type of the variable, None if it is not known.
        """
        return self.variables_types.get(name, None)

    def get_lower_bound(
        self,
        name,  # type: str
    ):  # type:(...) -> ndarray
        """Return the lower bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite).
        """
        return self._lower_bounds.get(name)

    def get_upper_bound(
        self,
        name,  # type: str
    ):  # type:(...) -> ndarray
        """Return the upper bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite).
        """
        return self._upper_bounds.get(name)

    def get_lower_bounds(
        self,
        variables_names=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> ndarray
        """Generate an array of the variables' lower bounds.

        Args:
            variables_names: The names of the variables
                of which the lower bounds are required.
                If None, use the variables of the design space.

        Returns:
            The lower bounds of the variables.
        """
        if self.__norm_data_is_computed and variables_names is None:
            return self.__lower_bounds_array
        return self.dict_to_array(self._lower_bounds, all_var_list=variables_names)

    def get_upper_bounds(
        self,
        variables_names=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> ndarray
        """Generate an array of the variables' upper bounds.

        Args:
            variables_names: The names of the variables
                of which the upper bounds are required.
                If None, use the variables of the design space.

        Returns:
            The upper bounds of the variables.
        """
        if self.__norm_data_is_computed and variables_names is None:
            return self.__upper_bounds_array
        return self.dict_to_array(self._upper_bounds, all_var_list=variables_names)

    def set_lower_bound(
        self,
        name,  # type: str
        lower_bound,  # type: ndarray
    ):  # type: (...) -> None
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The value of the lower bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        if name not in self.variables_names:
            raise ValueError("Variable '{}' is not known.".format(name))

        self._add_bound(name, self.variables_sizes[name], lower_bound, is_lower=True)
        self._add_norm_policy(name)

    def set_upper_bound(
        self,
        name,  # type: str
        upper_bound,  # type: ndarray
    ):  # type: (...) -> None
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The value of the upper bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        if name not in self.variables_names:
            raise ValueError("Variable '{}' is not known.".format(name))

        self._add_bound(name, self.variables_sizes[name], upper_bound, is_lower=False)
        self._add_norm_policy(name)

    def array_to_dict(
        self,
        x_array,  # type: ndarray
    ):  # type: (...) -> Dict[str,ndarray]
        """Convert the current point into a dictionary indexed by the variables names.

        Args:
            x_array: The current point.

        Returns:
            The dictionary version of the current point.
        """
        x_dict = {}
        current_index = 0
        # order given by self.variables_names
        for name in self.variables_names:
            size = self.variables_sizes[name]
            x_dict[name] = x_array[current_index : current_index + size]
            current_index += size
        return x_dict

    @staticmethod
    def __get_common_dtype(
        x_dict,  # type: Mapping[str,ndarray]
    ):  # type: (...) -> Union[complex128,float64,int32]
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
            if val_arr.dtype == complex128:
                return complex128
            if val_arr.dtype == int32:
                has_int = True
            if val_arr.dtype == float64:
                has_float = True
        if has_float:
            return float64
        if has_int:
            return int32
        return float64

    def dict_to_array(
        self,
        x_dict,  # type: Dict[str,ndarray]
        all_vars=True,  # type:bool
        all_var_list=None,  # type: Sequence[str]
    ):  # type: (...) -> ndarray
        """Convert an point as dictionary into an array.

        Args:
            x_dict: The point to be converted.
            all_vars: If True, all the variables to be considered
                shall be in the provided point.
            all_var_list: The variables to be considered.
                If None, use the variables of the design space.

        Returns:
            The point as an array.
        """
        dtype = self.__get_common_dtype(x_dict)
        if all_var_list is None:
            all_var_list = self.variables_names
        if all_vars:
            array_list = [array(x_dict[name], dtype=dtype) for name in all_var_list]
        else:
            array_list = [
                array(x_dict[name], dtype=dtype)
                for name in all_var_list
                if name in x_dict
            ]
        return concatenate(array_list)

    def get_pretty_table(
        self,
        fields=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> PrettyTable
        """Build a tabular view of the design space.

        Args:
            fields: The name of the fields to be exported.
                If None, export all the fields.

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
            curr = self._current_x.get(name)
            for i in range(size):
                data = {
                    "name": name,
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
                    data["value"] = curr[i]
                table.add_row([data[key] for key in fields])

        table.align["name"] = "l"
        table.align["type"] = "l"
        return table

    def export_hdf(
        self,
        file_path,  # type: Union[str,Path]
        append=False,  # type: bool
    ):  # type: (...) -> None
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
                    data_array = string_array(var_type)
                    var_grp.create_dataset(
                        self.VAR_TYPE_GROUP, data=data_array, dtype=data_array.dtype
                    )

                value = self._current_x.get(name)
                if value is not None:
                    var_grp.create_dataset(self.VALUE_GROUP, data=self.__to_real(value))

    def import_hdf(
        self,
        file_path,  # type: Union[str,Path]
    ):  # type: (...) -> None
        """Import a design space from an HDF file.

        Args:
            file_path: The path to the file
                containing the description of a design space.
        """
        with h5py.File(file_path, "r") as h5file:
            design_vars_grp = get_hdf5_group(h5file, self.DESIGN_SPACE_GROUP)
            variables_names = get_hdf5_group(design_vars_grp, self.NAMES_GROUP)

            for name in variables_names:
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
        var_group,  # type: h5py.Group
        dataset_name,  # type: str
    ):  # type: (...) -> Optional[ndarray]
        """Read data in a group.

        Args:
            var_group: The variable group.
            dataset_name: The name of the dataset.

        Returns:
            The data found in the group, if exist.
            Otherwise, None.
        """
        data = var_group.get(dataset_name)
        if data is not None:
            data = array(data)
        return data

    @staticmethod
    def __to_real(
        data,  # type:ndarray
    ):  # type: (...) -> ndarray
        """Convert complex to real NumPy array.

        Args:
            data: A complex NumPy array.

        Returns:
            A real NumPy array.
        """
        return array(array(data, copy=False).real, dtype=float64)

    def to_complex(self):  # type: (...) -> None
        """Cast the current value to complex."""
        for name, val in self._current_x.items():
            self._current_x[name] = array(val, dtype=complex128)

    def export_to_txt(
        self,
        output_file,  # type: Union[str,Path],
        fields=None,  # type: Optional[Sequence[str]]
        header_char="",  # type: str
        **table_options  # type: Any
    ):  # type: (...) -> None
        """Export the design space to a text file.

        Args:
            output_file: The path to the file.
            fields: The fields to be exported.
                If None, export all fields.
            header_char: The header character.
            **table_options: The names and values of additional attributes
                for the :class:`.PrettyTable` view
                generated by :meth:`get_pretty_table`.
        """
        output_file = Path(output_file)
        table = self.get_pretty_table(fields)
        table.border = False
        for option, val in table_options.items():
            table.__setattr__(option, val)
        with output_file.open("w") as outf:
            table_str = header_char + table.get_string()
            outf.write(table_str)

    @staticmethod
    def read_from_txt(
        input_file,  # type: Union[str,Path]
        header=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> DesignSpace
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
            header = strings_to_unicode_list(str_data[0, :].tolist())
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
        var_names = strings_to_unicode_list(str_data[start_read:, 0].tolist())
        unique_names = []
        prev_name = None
        for name in var_names:  # set([]) does not preserve order !
            if name not in unique_names:
                unique_names.append(name)
                prev_name = name
            elif prev_name != name:
                raise ValueError(
                    "Malformed DesignSpace input file {} contains some variables ({}) "
                    "in a non-consecutive order.".format(input_file, name)
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

    def __str__(self):  # type:(...) -> str
        if self.name is None:
            header_suffix = ""
        else:
            header_suffix = " {}".format(self.name)
        header = "Design space:{}\n".format(header_suffix)

        return header + self.get_pretty_table().get_string()

    def project_into_bounds(
        self,
        x_c,  # type: ndarray
        normalized=False,  # type: bool
    ):  # type: (...) -> ndarray
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
        variable,  # type: str
    ):  # type: (...) -> bool
        """Return whether the variable is in the design space.

        Args:
            variable: The name of the variable.
        """
        return variable in self.variables_names

    def __len__(self):  # type: (...) -> int
        """The length of the design space, corresponding to the number of variables."""
        return len(self.variables_names)

    def __iter__(self):  # type: (...) -> Iterable[str]
        return iter(self.variables_names)

    def __setitem__(
        self,
        name,  # type: str
        item,  # type: DesignVariable
    ):  # type: (...) -> None
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
        other,  # type: DesignSpace
    ):  # type: (...) -> bool
        if not isinstance(other, self.__class__):
            return False

        if len(other) != len(self):
            return False

        for key, val in self.items():
            if key not in other:
                return False

            hash1 = hash_data_dict(flatten_mapping(val._asdict()))
            hash2 = hash_data_dict(flatten_mapping(other[key]._asdict()))
            if hash1 != hash2:
                return False

        return True

    def __getitem__(
        self,
        name,  # type: str
    ):  # type: (...) -> DesignVariable
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
            raise KeyError("Variable '{}' is not known.".format(name))

        try:
            value = self.get_current_x([name])
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
        other,  # type: DesignSpace
    ):  # type: (...) -> None
        """Extend the design space with another design space.

        Args:
            other: The design space to be appended to the current one.
        """
        for name in other.variables_names:
            size = other.get_size(name)
            var_type = other.get_type(name)
            l_b = other.get_lower_bound(name)
            u_b = other.get_upper_bound(name)
            value = other.get_current_x_dict()[name]
            self.add_variable(name, size, var_type, l_b, u_b, value)

    @staticmethod
    def __cast_array_to_list(
        value,  # type: Union[str,int,ndarray]
    ):  # type: (...) -> Union[str,int,List[Union[str,int]]]
        """Convert a value to a ``List`` if it is a NumPy array.

        Args:
            value: The value to be casted.

        Returns:
            Either the original value or the NumPy array converted to a ``List``.
        """
        return value if not isinstance(value, ndarray) else value.tolist()

    @classmethod
    def __cast_mapping(
        cls,
        mapping,  # type: Mapping[str,Union[str,int,ndarray]]
    ):  # type: (...) -> Dict[str,Union[str,int,List[Union[str,int]]]]
        """Convert the NumPy arrays of a mapping to ``List``.

        Args:
            mapping: The value to be casted.

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
