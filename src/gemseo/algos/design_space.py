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
a.k.a. the design variables.

A :class:`.DesignSpace` describes this design space at a given state, in terms of names,
sizes, types, bounds and current values of the design variables.

Variables can easily be added to the :class:`.DesignSpace` using the
:meth:`.DesignSpace.add_variable` method or removed using the
:meth:`.DesignSpace.remove_variable` method.

We can also filter the design variables using the :meth:`.DesignSpace.filter` method.

Getters and setters are also available to get or set the value of a given variable
property.

Lastly, an instance of :class:`.DesignSpace` can be stored in a txt or HDF file.
"""

from __future__ import annotations

import logging
import re
from copy import deepcopy
from numbers import Complex
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final
from typing import Literal
from typing import overload

import h5py
from numpy import abs as np_abs
from numpy import array
from numpy import array_equal
from numpy import atleast_1d
from numpy import bytes_
from numpy import complex128
from numpy import concatenate
from numpy import dtype
from numpy import equal
from numpy import finfo
from numpy import float64
from numpy import full
from numpy import genfromtxt
from numpy import inf
from numpy import int64
from numpy import isin
from numpy import isinf
from numpy import isnan
from numpy import logical_and
from numpy import logical_or
from numpy import mod
from numpy import ndarray
from numpy import ones_like
from numpy import round as np_round
from numpy import vectorize
from numpy import where
from numpy import zeros_like
from prettytable import PrettyTable

from gemseo.algos._variable import TYPE_MAP
from gemseo.algos._variable import DataType
from gemseo.algos._variable import Variable
from gemseo.algos.optimization_result import OptimizationResult
from gemseo.utils.compatibility.scipy import sparse_classes
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.hdf5 import get_hdf5_group
from gemseo.utils.repr_html import REPR_HTML_WRAPPER
from gemseo.utils.string_tools import _format_value_in_pretty_table_16
from gemseo.utils.string_tools import convert_strings_to_iterable
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.typing import BooleanArray
    from gemseo.typing import IntegerArray
    from gemseo.typing import RealOrComplexArrayT

LOGGER = logging.getLogger(__name__)


class DesignSpace:
    """Description of a design space.

    It defines a set of variables from their names, sizes, types and bounds.

    In addition,
    it provides the current values of these variables
    that can be used as the initial solution of an :class:`.OptimizationProblem`.
    """

    name: str
    """The name of the space."""

    _variables: dict[str, Variable]
    """The variables."""

    dimension: int
    """The total dimension of the space, corresponding to the sum of the sizes of the
    variables."""

    normalize: dict[str, BooleanArray]
    """The normalization policies of the variables components indexed by the variables
    names; if `True`, the component can be normalized."""

    DesignVariableType = DataType

    # TODO: API: the values are not dtypes but types, either fix the values or the name.
    VARIABLE_TYPES_TO_DTYPES: Final[dict[str, type[int64 | float64]]] = TYPE_MAP
    """One NumPy ``dtype`` per design variable type."""

    MINIMAL_FIELDS: ClassVar[list[str]] = ["name", "lower_bound", "upper_bound"]
    TABLE_NAMES: ClassVar[list[str]] = [
        "name",
        "lower_bound",
        "value",
        "upper_bound",
        "type",
    ]

    DESIGN_SPACE_GROUP: ClassVar[str] = "design_space"
    NAME_GROUP: ClassVar[str] = "name"
    NAMES_GROUP: ClassVar[str] = "names"
    LB_GROUP: ClassVar[str] = "l_b"
    UB_GROUP: ClassVar[str] = "u_b"
    VAR_TYPE_GROUP: ClassVar[str] = "var_type"
    VALUE_GROUP: ClassVar[str] = "value"
    SIZE_GROUP: ClassVar[str] = "size"

    __INT_DTYPE: Final[dtype[int64]] = dtype(
        VARIABLE_TYPES_TO_DTYPES[DesignVariableType.INTEGER]
    )
    __FLOAT_DTYPE: Final[dtype[float64]] = dtype(
        VARIABLE_TYPES_TO_DTYPES[DesignVariableType.FLOAT]
    )
    __COMPLEX_DTYPE: Final[dtype[complex128]] = dtype("complex128")

    __DEFAULT_COMMON_DTYPE: Final[dtype[[float64]]] = __FLOAT_DTYPE
    """The default NumPy data type of the variables."""

    __CAMEL_CASE_REGEX: Final[re.Pattern] = re.compile(r"[A-Z][^A-Z]*")
    """A regular expression to decompose a CamelCase string."""

    __current_value_array: ndarray
    """The current value stored as a concatenated array."""

    __norm_current_value: dict[str, ndarray]
    """The norm of the current value."""

    __norm_current_value_array: ndarray
    """The norm of the current value stored as a concatenated array."""

    __normalize_integer_variables: bool = False
    """Whether to normalize integer variables.
    This can be used when forcing the execution of an optimization library
    restricted to float variables on a problem containing integer variables."""

    __names_to_indices: dict[str, range]
    """The names bound to the indices in a design vector."""

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name to be given to the design space.
                If empty, the design space is unnamed.
        """  # noqa: D205, D212
        self.dimension = 0
        self.name = name
        self.normalize = {}
        self._variables = {}

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
        self.__bound_tol = 100.0 * finfo(float64).eps
        self.__current_value = {}
        self.__has_current_value = False
        self.__common_dtype = self.__DEFAULT_COMMON_DTYPE
        self.__clear_dependent_data()
        self.__names_to_indices = {}

    @property
    def _current_value(self) -> dict[str, ndarray]:
        """The current design value."""
        return self.__current_value

    @property
    def variable_names(self) -> list[str]:
        """The variable names."""
        return list(self._variables)

    @property
    def variable_sizes(self) -> dict[str, int]:
        """The variable sizes."""
        return {name: variable.size for name, variable in self._variables.items()}

    @property
    def variable_types(self) -> dict[str, str]:
        """The variable types."""
        return {name: variable.type for name, variable in self._variables.items()}

    def __update_current_metadata(self) -> None:
        """Update information about the current design value for quick access."""
        self.__update_current_status()
        if self.__has_current_value:
            self.__clear_dependent_data()

    def __clear_dependent_data(self) -> None:
        """Reset the data that depends on the current value."""
        self.__current_value_array = array([])
        self.__norm_current_value = {}
        self.__norm_current_value_array = array([])

    def __update_current_status(self) -> None:
        """Update the availability of current design values for all the variables."""
        if (
            not self.__current_value
            or self.__current_value.keys() != self._variables.keys()
        ):
            self.__has_current_value = False
            return

        for value in self.__current_value.values():
            if value is None:
                self.__has_current_value = False
                return

        self.__has_current_value = True

    def remove_variable(
        self,
        name: str,
    ) -> None:
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        self.__norm_data_is_computed = False
        size = self._variables[name].size
        self.dimension -= size
        del self.__names_to_indices[name]
        variable_is_reached = False
        for variable_name in self:
            if variable_name == name:
                variable_is_reached = True
            elif variable_is_reached:
                indices = self.__names_to_indices[variable_name]
                # N.B. the steps of the ranges of indices are assumed equal to 1
                self.__names_to_indices[variable_name] = range(
                    indices.start - size,
                    indices.stop - size,
                )

        del self.normalize[name]

        if name in self.__current_value:
            del self.__current_value[name]

        del self._variables[name]
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
        keep_variables = convert_strings_to_iterable(keep_variables)
        design_space = deepcopy(self) if copy else self
        for name in self.variable_names:
            if name not in keep_variables:
                design_space.remove_variable(name)
        for name in keep_variables:
            self.__check_known_variable(name)
        return design_space

    def filter_dimensions(self, name: str, dimensions: Sequence[int]) -> DesignSpace:
        """Filter the design space to keep a subset of dimensions for a variable.

        Args:
            name: The name of the variable.
            dimensions: The dimensions of the variable to be kept,
                between :math:`0` and :math:`d-1`
                where :math:`d` is the number of dimensions of the variable.

        Returns:
            The filtered design space.

        Raises:
            ValueError: If a dimension does not exist.
        """
        nonexistent_dimensions = {i for i in dimensions if i >= self.get_size(name)}
        if nonexistent_dimensions:
            plural = len(nonexistent_dimensions) > 1
            msg = (
                f"Dimension{'s' if plural else ''}"
                f" {pretty_str(nonexistent_dimensions, use_and=True)}"
                f" of variable '{name}' {'do' if plural else 'does'} not exist."
            )
            raise ValueError(msg)

        self.__norm_data_is_computed = False
        n_kept = len(dimensions)
        n_removed = self.get_size(name) - n_kept
        self.dimension -= n_removed
        variable = self._variables[name]
        self._variables[name] = Variable(
            size=variable.size - n_removed,
            type=variable.type,
            lower_bound=variable.lower_bound[dimensions],
            upper_bound=variable.upper_bound[dimensions],
        )
        if name in self.__current_value:
            self.set_current_variable(name, self.get_current_value(name)[dimensions])

        # Update the mapping from names to array indices
        name_reached = False
        for _name, indices in self.__names_to_indices.items():
            if _name == name:
                name_reached = True
                self.__names_to_indices[_name] = range(
                    indices.start,
                    indices.stop - n_removed,
                )
            elif name_reached:
                self.__names_to_indices[_name] = range(
                    indices.start - n_removed,
                    indices.stop - n_removed,
                )

        self.__update_current_metadata()
        return self

    def add_variable(
        self,
        name: str,
        size: int = 1,
        type_: DataType = DesignVariableType.FLOAT,
        lower_bound: Number | Iterable[Number] = -inf,
        upper_bound: Number | Iterable[Number] = inf,
        value: Number | Iterable[Number] | None = None,
    ) -> None:
        r"""Add a variable to the design space.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            type_: Either the type of the variable
                or the types of its components.
            lower_bound: The lower bound of the variable.
                If ``None``, use :math:`-\infty`.
            upper_bound: The upper bound of the variable.
                If ``None``, use :math:`+\infty`.
            value: The default value of the variable.
                If ``None``, do not use a default value.

        Raises:
            ValueError: Either if the variable already exists
                or if the size is not a positive integer.
        """
        self._check_variable_name(name)
        self.__norm_data_is_computed = False
        self._variables[name] = Variable(
            size=size,
            type=type_,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self.__names_to_indices[name] = range(self.dimension, self.dimension + size)
        self.dimension += size
        self._add_norm_policy(name)
        if value is not None:
            array_value = atleast_1d(value)
            self._check_value(array_value, name)
            if len(array_value) == 1 and size > 1:
                array_value = full(size, value)
            self.__current_value[name] = array_value.astype(
                self.VARIABLE_TYPES_TO_DTYPES[self.get_type(name)],
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

    def _check_variable_name(self, name: str) -> None:
        """Check if the space contains a variable.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: When the variable already exists.
        """
        if name in self:
            msg = f"The variable '{name}' already exists."
            raise ValueError(msg)

    @property
    def names_to_indices(self) -> dict[str, range]:
        """The names bound to the indices."""
        return self.__names_to_indices

    def _add_norm_policy(
        self,
        name: str,
    ) -> None:
        """Add a normalization policy to a variable.

        Unbounded variables are not normalized.
        Bounded variables (both from above and from below) are normalized.

        Args:
            name: The name of a variable.
        """
        # Check that the variable is in the design space:
        self.__check_known_variable(name)
        variable = self._variables[name]
        # Set the normalization policy:
        if (
            variable.type == self.DesignVariableType.FLOAT
            or self.enable_integer_variables_normalization
        ):
            # Only bounded float variables are normalized.
            normalize = logical_and(
                variable.lower_bound != -inf, variable.upper_bound != inf
            )
        else:
            # Integer variables are not normalized (unless treated as float).
            normalize = full(variable.size, False)

        self.normalize[name] = normalize

    @staticmethod
    def __is_integer(
        values: ndarray | Number,
    ) -> ndarray:
        """Check whether each value is either an integer, infinite, or None.

        Args:
            values: The array or number to be checked.

        Returns:
            Whether each of the given values is either an integer, infinite, or None.
        """
        return array([
            isinf(x) or x is None or not mod(x, 1) for x in atleast_1d(values)
        ])

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
        return value is None or isinstance(value, Complex)

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
            msg = (
                f"The value {value} of variable '{name}' "
                "has a dimension greater than 1 "
                "while a scalar or a 1D iterable object "
                "(array, list, tuple, ...) "
                "was expected."
            )
            raise ValueError(msg)

        # OK if all components are None
        if all(equal(value, None)):
            return True

        test = vectorize(self.__is_numeric)(value)
        indices = all_indices - set(test.nonzero()[0])
        if indices:
            plural = len(indices) > 1
            msg = (
                f"The following value{'s' if plural else ''} of variable '{name}' "
                f"{'are' if plural else 'is'} "
                "neither None nor complex and cannot be cast to float: "
                f"{', '.join([f'{value[i]} (index {i})' for i in indices])}."
            )
            raise ValueError(msg)

        test = vectorize(self.__is_not_nan)(value)
        indices = all_indices - set(test.nonzero()[0])
        if indices:
            plural = len(indices) > 1
            msg = (
                f"The following value{'s' if plural else ''} of variable '{name}' "
                f"{'are' if plural else 'is'} neither None nor "
                f"{'numbers' if plural else 'a number'}: "
                f"{', '.join([f'{value[i]} (index {i})' for i in indices])}."
            )
            raise ValueError(msg)

        # Check if some components of an integer variable are not integer.
        if self.variable_types[name] == self.DesignVariableType.INTEGER:
            indices = all_indices - set(self.__is_integer(value).nonzero()[0])
            if indices:
                plural = len(indices) > 1
                msg = (
                    f"The following value{'s' if plural else ''} of variable '{name}' "
                    f"{'are ' if plural else 'is '} neither None nor integer"
                    f"while variable '{name}' is of type integer: "
                    f"{', '.join([f'{value[i]} (index {i})' for i in indices])}."
                )
                raise ValueError(msg)

        return True

    @property
    def _lower_bounds(self) -> ndarray[float]:
        return {
            name: variable.lower_bound for name, variable in self._variables.items()
        }

    @property
    def _upper_bounds(self) -> ndarray[float]:
        return {
            name: variable.upper_bound for name, variable in self._variables.items()
        }

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
        l_b = self.get_lower_bound(name)
        u_b = self.get_upper_bound(name)
        current_value = self.__current_value.get(name, None)
        not_none = ~equal(current_value, None)
        indices = (
            logical_or(
                current_value[not_none] < l_b[not_none] - self.__bound_tol,
                current_value[not_none] > u_b[not_none] + self.__bound_tol,
            )
        ).nonzero()[0]
        for index in indices:
            msg = (
                f"The current value of variable '{name}' ({current_value[index]}) is "
                f"not between the lower bound {l_b[index]} and the upper bound "
                f"{u_b[index]}."
            )
            raise ValueError(msg)

    @property
    def has_current_value(self) -> bool:
        """Check if each variable has a current value.

        Returns:
            Whether the current design value is defined for all variables.
        """
        return self.__has_current_value

    @property
    def has_integer_variables(self) -> bool:
        """Check if the design space has at least one integer variable.

        Returns:
            Whether the design space has at least one integer variable.
        """
        return self.DesignVariableType.INTEGER in [
            self.get_type(variable_name) for variable_name in self
        ]

    def check(self) -> None:
        """Check the state of the design space.

        Raises:
            ValueError: If the design space is empty.
        """
        if not self._variables:
            msg = "The design space is empty."
            raise ValueError(msg)

        if self.has_current_value:
            self._check_current_names()

    def check_membership(
        self,
        x_vect: Mapping[str, ndarray] | ndarray,
        variable_names: Sequence[str] = (),
    ) -> None:
        """Check whether the variables satisfy the design space requirements.

        Args:
            x_vect: The values of the variables.
            variable_names: The names of the variables.
                If empty, use the names of the variables of the design space.

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
                msg = (
                    f"The array should be of size {self.dimension}; got {x_vect.size}."
                )
                raise ValueError(msg)
            if variable_names:
                self.__check_membership(
                    split_array_to_dict_of_arrays(
                        x_vect,
                        self.variable_sizes,
                        variable_names,
                    ),
                    variable_names,
                )
            else:
                if self.__lower_bounds_array is None:
                    self.__lower_bounds_array = self.get_lower_bounds()

                if self.__upper_bounds_array is None:
                    self.__upper_bounds_array = self.get_upper_bounds()

                self.__check_membership_x_vect(x_vect)
        else:
            msg = (
                "The input vector should be an array or a dictionary; "
                f"got a {type(x_vect)} instead."
            )
            raise TypeError(msg)

    def __check_membership_x_vect(self, x_vect: ndarray) -> None:
        """Check whether a vector is comprised between the lower and upper bounds.

        Args:
            x_vect: The vector.

        Raises:
            ValueError: When the values are outside the bounds of the variables.
        """
        l_b = self.__lower_bounds_array
        u_b = self.__upper_bounds_array
        indices = (x_vect < l_b - self.__bound_tol).nonzero()[0]
        if len(indices):
            value = x_vect[indices]
            lower_bound = l_b[indices]
            msg = (
                f"The components {indices} of the given array ({value}) "
                f"are lower than the lower bound ({lower_bound}) "
                f"by {lower_bound - value}."
            )
            raise ValueError(msg)

        indices = (x_vect > u_b + self.__bound_tol).nonzero()[0]
        if len(indices):
            value = x_vect[indices]
            upper_bound = u_b[indices]
            msg = (
                f"The components {indices} of the given array ({value}) "
                f"are greater than the upper bound ({upper_bound}) "
                f"by {value - upper_bound}."
            )
            raise ValueError(msg)

    def __check_membership(
        self,
        x_dict: Mapping[str, ndarray],
        variable_names: Iterable[str],
    ) -> None:
        """Check whether the variables satisfy the design space requirements.

        Args:
            x_dict: The values of the variables.
            variable_names: The names of the variables.
                If empty, use the names of the variables of the design space.

        Raises:
            ValueError: Either if the dimension of an array is wrong,
                if the values are outside the bounds of the variables or
                if the component of an integer variable is not an integer.
        """
        variable_names = variable_names or self._variables
        for name in variable_names:
            variable = self._variables[name]
            value = x_dict[name]
            if value is None:
                continue

            if value.size != variable.size:
                msg = (
                    f"The variable {name} of size {variable.size} "
                    f"cannot be set with an array of size {value.size}."
                )
                raise ValueError(msg)

            for i in range(variable.size):
                x_real = value[i].real
                lower_bound = variable.lower_bound[i]
                if x_real < lower_bound - self.__bound_tol:
                    msg = (
                        f"The component {name}[{i}] of the given array ({x_real}) "
                        f"is lower than the lower bound ({lower_bound}) "
                        f"by {lower_bound - x_real:.1e}."
                    )
                    raise ValueError(msg)

                upper_bound = variable.upper_bound[i]
                if upper_bound + self.__bound_tol < x_real:
                    msg = (
                        f"The component {name}[{i}] of the given array ({x_real}) "
                        f"is greater than the upper bound ({upper_bound}) "
                        f"by {x_real - upper_bound:.1e}."
                    )
                    raise ValueError(msg)

                if (
                    variable.type == self.DesignVariableType.INTEGER
                ) and not self.__is_integer(x_real):
                    msg = (
                        f"The variable {name} is of type integer; "
                        f"got {name}[{i}] = {x_real}."
                    )
                    raise ValueError(msg)

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

                   (
                       {
                           "x": array(are_x_lower_bounds_active),
                           "y": array(are_y_lower_bounds_active),
                       },
                       {
                           "x": array(are_x_upper_bounds_active),
                           "y": array(are_y_upper_bounds_active),
                       },
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
            current_x = self.convert_array_to_dict(x_vec)
        elif isinstance(x_vec, dict):
            current_x = x_vec
        else:
            msg = f"Expected dict or array for x_vec argument; got {type(x_vec)}."
            raise TypeError(msg)

        active_l_b = {}
        active_u_b = {}
        for name in self:
            l_b = self.get_lower_bound(name)
            l_b = where(equal(l_b, None), -inf, l_b)
            u_b = self.get_upper_bound(name)
            u_b = where(equal(u_b, None), inf, u_b)
            x_vec_i = current_x[name]
            # lower bound saturated
            active_l_b[name] = np_abs(x_vec_i - l_b) <= tol
            # upper bound saturated
            active_u_b[name] = np_abs(x_vec_i - u_b) <= tol

        return active_l_b, active_u_b

    def _check_current_names(
        self,
        variable_names: Iterable[str] = (),
    ) -> None:
        """Check the names of the current design value.

        Args:
            variable_names: The names of the variables.
                If empty, use the names of the variables of the design space.

        Raises:
            ValueError: If the names of the variables of the current design value
                and the names of the variables of the design space are different.
        """
        if sorted(self) != sorted(self.__current_value.keys()):
            msg = (
                f"Expected current_x variables:"
                f" {pretty_str(self, use_and=True)}; "
                f"got {pretty_str(self.__current_value.keys(), use_and=True)}."
            )
            raise ValueError(msg)
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
                N.B. Normalization is possible if and only if
                *all* the current design values are set.

        Returns:
            The current design value.

        Raises:
            ValueError: If names in ``variable_names`` are not in the design space.
            KeyError: If one of the required design variables has no current value.

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
        if variable_names is not None and not variable_names:
            return {} if as_dict else array([])

        # Whether to return the current values of all the design variables
        return_all = (
            variable_names is None or set(variable_names) == self._variables.keys()
        )

        if not self.__has_current_value:
            # A design variable has no current value.
            if return_all and as_dict and not normalize:
                # TODO: API break: The cases `as_dict is True` (current block) and
                # `as_dict is False` are handled differently: in the former an empty
                # dictionary is returned (which does not make sense) while in the latter
                # an exception is raised.
                # For consistency, the current block should be removed.
                # This will break the API because an exception will be raised (as in the
                # case `as_dict is False`) instead of returning an empty dictionary.
                # N.B. the good practice is for the user to either catch the exception
                # or, even better, to check the ``DesignSpace.has_current_value`` flag
                # before calling the ``DesignSpace.get_current_value``.
                # This break is simple to handle in GEMSEO, but make sure to take care
                # of the plugins as well.
                return self._current_value

            if return_all or normalize:
                variables = self._variables.keys() - self.__current_value.keys()
                msg = (
                    "There is no current value for the design variables: "
                    f"{pretty_str(variables, use_and=True)}."
                )
                if not normalize:
                    raise KeyError(msg)

                msg = (
                    "The current value of a design space cannot be normalized "
                    f"when some variables have no current value. {msg}"
                )
                raise KeyError(msg)

        if normalize:
            # Make sure the normalized current value is computed.
            self.__normalize_current_value()

        if (
            variable_names is None or list(variable_names) == self.variable_names
        ) and not as_dict:
            # Return the current value of all the variables in the design space order.
            return self.__format_current_value_array(
                self.__norm_current_value_array
                if normalize
                else self.__get_current_value_array(),
                complex_to_real,
            )

        if return_all and as_dict:
            return self.__format_current_value_dict(
                self.__norm_current_value if normalize else self.__current_value,
                complex_to_real,
            )

        # Check that the required variables exist.
        not_variable_names = set(variable_names) - set(self._variables)
        if not_variable_names:
            msg = (
                "There are no such variables named: "
                f"{pretty_str(not_variable_names, use_and=True)}."
            )
            raise ValueError(msg)

        # Check that the required variables have a current value.
        # N.B. when `normalize` is `True`, this has already been checked.
        if not normalize:
            missing_values = set(variable_names) - set(self.__current_value)
            if missing_values:
                msg = (
                    "There is no current value for the design variables: "
                    f"{pretty_str(missing_values, use_and=True)}."
                )
                raise KeyError(msg)

        dict_ = self.__norm_current_value if normalize else self.__current_value
        current_value = {name: dict_[name] for name in variable_names}

        if as_dict:
            return self.__format_current_value_dict(current_value, complex_to_real)

        return self.__format_current_value_array(
            self.convert_dict_to_array(current_value, variable_names), complex_to_real
        )

    def __get_current_value_array(self) -> ndarray:
        """Return the current value as a NumPy array.

        Returns:
            The current value as a NumPy array.
        """
        if not len(self.__current_value_array):
            self.__current_value_array = self.convert_dict_to_array(
                self.__current_value
            )

        return self.__current_value_array

    def __normalize_current_value(self) -> None:
        """Normalize the current value."""
        if not len(self.__norm_current_value_array):
            self.__norm_current_value_array = self.normalize_vect(
                self.__get_current_value_array(),
            )
            self.__norm_current_value = self.convert_array_to_dict(
                self.__norm_current_value_array,
            )
            for name, to_normalize in self.normalize.items():
                if (
                    not to_normalize.any()
                    and self.variable_types[name] is self.DesignVariableType.INTEGER
                ):
                    self.__norm_current_value[name] = self.__norm_current_value[
                        name
                    ].astype(self.__INT_DTYPE, copy=False)

    @staticmethod
    def __format_current_value_dict(
        current_value: dict[str, ndarray], complex_to_real: bool
    ) -> dict[str, ndarray]:
        """Return a current value as a dictionary of real or complex NumPy arrays.

        Args:
            current_value: The current value.
            complex_to_real: Whether to cast complex numbers to real ones.

        Returns:
            The current value as a dictionary of real or complex NumPy arrays.
        """
        if complex_to_real:
            return {name: value.real for name, value in current_value.items()}

        return current_value

    @staticmethod
    def __format_current_value_array(
        current_value: ndarray, complex_to_real: bool
    ) -> ndarray:
        """Return a current value as a real or complex NumPy array.

        Args:
            current_value: The current value.
            complex_to_real: Whether to cast complex numbers to real ones.

        Returns:
            The current value as a real or complex NumPy array.
        """
        if complex_to_real:
            return current_value.real

        return current_value

    def get_indexed_variable_names(
        self, variable_names: str | Sequence[str] = ()
    ) -> list[str]:
        """Create the names of the components of variables.

        If the size of the variable is equal to 1,
        its name remains unaltered.
        Otherwise,
        it concatenates the name of the variable and the index of the component.

        Args:
            variable_names: The names of the design variables.
                If ``empty``, use all the design variables.

        Returns:
            The name of the components of the variables.
        """
        if not variable_names:
            variable_names = self.variable_names

        elif isinstance(variable_names, str):
            variable_names = [variable_names]

        var_ind_names = []
        for variable_name in variable_names:
            size = self.get_size(variable_name)
            var_ind_names.extend([
                repr_variable(variable_name, i, size) for i in range(size)
            ])

        return var_ind_names

    def get_variables_indexes(
        self,
        variable_names: Iterable[str],
        use_design_space_order: bool = True,
    ) -> IntegerArray:
        """Return the indexes of a design array corresponding to variables names.

        Args:
            variable_names: The names of the variables.
            use_design_space_order: Whether to order the indexes according to
                the order of the variables names in the design space.
                Otherwise, the indexes will be ordered in the same order as
                the variables names were required.

        Returns:
            The indexes of a design array corresponding to the variables names.
        """
        if use_design_space_order:
            names = [name for name in self if name in variable_names]
        else:
            names = variable_names

        return concatenate([self.__names_to_indices[name] for name in names])

    def __update_normalization_vars(self) -> None:
        """Compute the inner attributes used for normalization and unnormalization."""
        self.__lower_bounds_array = self.get_lower_bounds()
        self.__upper_bounds_array = self.get_upper_bounds()
        self._norm_factor = self.__upper_bounds_array - self.__lower_bounds_array
        self.__norm_inds = self.convert_dict_to_array(self.normalize).nonzero()[0]
        # In case lb=ub
        norm_factor_is_zero = self._norm_factor == 0.0
        self._norm_factor_inv = 1.0 / where(norm_factor_is_zero, 1, self._norm_factor)
        integer = self.DesignVariableType.INTEGER
        self.__integer_components = concatenate(
            tuple(
                [variable.type == integer] * variable.size
                for variable in self._variables.values()
            )
        )
        self.__no_integer = not self.__integer_components.any()
        self.__norm_data_is_computed = True
        if self.__has_current_value:
            self.__common_dtype = self.__get_common_dtype(self.__current_value.values())
        else:
            self.__common_dtype = self.__DEFAULT_COMMON_DTYPE

    def normalize_vect(
        self,
        x_vect: RealOrComplexArrayT,
        minus_lb: bool = True,
        out: RealOrComplexArrayT | None = None,
    ) -> RealOrComplexArrayT:
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
            minus_lb: If ``True``, remove the lower bounds at normalization.
            out: The array to store the normalized vector.
                If ``None``, create a new array.

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

        norm_inds = self.__norm_inds
        if norm_inds.size == 0:
            # There is no variable index to normalize.
            return out

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

        if minus_lb:
            out[..., norm_inds] -= self.__lower_bounds_array[norm_inds]

        if isinstance(out, sparse_classes):
            # Construct a mask to only scale the required columns
            column_mask = isin(out.indices, norm_inds)
            # Scale the corresponding coefficients
            out.data[column_mask] *= self._norm_factor_inv[out.indices][column_mask]
        else:
            out[..., norm_inds] *= self._norm_factor_inv[norm_inds]

        return out

    def normalize_grad(
        self,
        g_vect: RealOrComplexArrayT,
    ) -> RealOrComplexArrayT:
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
        g_vect: RealOrComplexArrayT,
    ) -> RealOrComplexArrayT:
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
        x_vect: RealOrComplexArrayT,
        minus_lb: bool = True,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> RealOrComplexArrayT:
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
                If ``None``, create a new array.

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
                msg += f"lower bounds violated: {value[lower_bounds_violated]}; "

            if any_upper_bound_violated:
                msg += f"upper bounds violated: {value[upper_bounds_violated]}; "

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

        if norm_inds.size:
            if isinstance(out, sparse_classes):
                # Construct a mask to only scale the required columns
                column_mask = isin(out.indices, norm_inds)
                # Scale the corresponding coefficients
                out.data[column_mask] *= self._norm_factor[out.indices][column_mask]
            else:
                out[..., norm_inds] *= self._norm_factor[norm_inds]

            if minus_lb:
                out[..., norm_inds] += lower_bounds[norm_inds]

        if not self.__no_integer:
            self.round_vect(out, copy=False)
            if recast_to_int:
                out = out.astype(self.__INT_DTYPE)

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
                If ``None``, create a new array.

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
                If ``None``, create a new array.

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

        rounded_x_vect = x_vect.copy() if copy else x_vect

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
            self.__current_value = {k: v for k, v in value.items() if k in self}
        elif isinstance(value, ndarray):
            if value.size != self.dimension:
                msg = (
                    "Invalid current_x, "
                    f"dimension mismatch: {self.dimension} != {value.size}."
                )
                raise ValueError(msg)
            self.__current_value = self.convert_array_to_dict(value)
        elif isinstance(value, OptimizationResult):
            if value.x_opt.size != self.dimension:
                msg = (
                    "Invalid x_opt, "
                    f"dimension mismatch: {self.dimension} != {value.x_opt.size}."
                )
                raise ValueError(msg)
            self.__current_value = self.convert_array_to_dict(value.x_opt)
        else:
            msg = (
                "The current design value should be either an array, "
                "a dictionary of arrays "
                "or an optimization result; "
                f"got {type(value)} instead."
            )
            raise TypeError(msg)

        for name, value in self.__current_value.items():
            if value is not None:
                variable_type = self.get_type(name)
                if variable_type == self.DesignVariableType.INTEGER:
                    value = value.astype(self.VARIABLE_TYPES_TO_DTYPES[variable_type])
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
        self.__check_known_variable(name)
        self.__current_value[name] = current_value
        self.__update_current_metadata()

    def __check_known_variable(self, name: str) -> None:
        """Check whether a variable is known.

        Raises:
            ValueError: If the variable is not known.
        """
        if name not in self:
            msg = f"Variable '{name}' is not known."
            raise ValueError(msg)

    def get_size(
        self,
        name: str,
    ) -> int:
        """Get the size of a variable.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the variable is not known.

        Returns:
            The size of the variable.
        """
        self.__check_known_variable(name)
        return self._variables[name].size

    def get_type(
        self,
        name: str,
    ) -> str:
        """Return the type of a variable.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the variable is not known.

        Returns:
            The type of the variable.
        """
        self.__check_known_variable(name)
        return str(self._variables[name].type)

    def get_lower_bound(self, name: str) -> ndarray:
        """Return the lower bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite).
        """
        return self._variables[name].lower_bound

    def get_upper_bound(self, name: str) -> ndarray:
        """Return the upper bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite).
        """
        return self._variables[name].upper_bound

    @overload
    def get_lower_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: Literal[False] = False,
    ) -> ndarray: ...

    @overload
    def get_lower_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: Literal[True] = False,
    ) -> dict[str, ndarray]: ...

    def get_lower_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the lower bounds of design variables.

        Args:
            variable_names: The names of the design variables.
                If empty, the lower bounds of all the design variables are returned.
            as_dict: Whether to return the lower bounds
                as a dictionary of the form ``{variable_name: variable_lower_bound}``.

        Returns:
            The lower bounds of the design variables.
        """
        return self.__get_values(
            variable_names,
            as_dict,
            self._lower_bounds,
            self.__lower_bounds_array,
        )

    @overload
    def get_upper_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: Literal[False] = False,
    ) -> ndarray: ...

    @overload
    def get_upper_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: Literal[True] = False,
    ) -> dict[str, ndarray]: ...

    def get_upper_bounds(
        self,
        variable_names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the upper bounds of design variables.

        Args:
            variable_names: The names of the design variables.
                If empty, the upper bounds of all the design variables are returned.
            as_dict: Whether to return the upper bounds
                as a dictionary of the form ``{variable_name: variable_upper_bound}``.

        Returns:
            The upper bounds of the design variables.
        """
        return self.__get_values(
            variable_names,
            as_dict,
            self._upper_bounds,
            self.__upper_bounds_array,
        )

    @overload
    def __get_values(
        self,
        variable_names: Sequence[str],
        as_dict: Literal[False],
        value_as_dict: dict[str, ndarray],
        value_as_array: ndarray,
    ) -> ndarray: ...

    @overload
    def __get_values(
        self,
        variable_names: Sequence[str],
        as_dict: Literal[True],
        value_as_dict: dict[str, ndarray],
        value_as_array: ndarray,
    ) -> dict[str, ndarray]: ...

    def __get_values(
        self,
        variable_names: Sequence[str],
        as_dict: bool,
        value_as_dict: dict[str, ndarray],
        value_as_array: ndarray,
    ) -> ndarray | dict[str, ndarray]:
        """Return the (lower or upper) bounds of design variables.

        Args:
            variable_names: The names of the design variables.
                If empty, then the values of all the design variables are returned.
                If empty, then the values of all the design variables are returned.
            as_dict: Whether to return the value
                as a dictionary of the form ``{variable_name: variable_value}``.
            value_as_dict: A dictionary of the values of all the design variables.
            value_as_array: The NumPy array of the values of all the design variables

        Returns:
            The bounds of the design variables.
        """
        if self.__norm_data_is_computed and not variable_names and not as_dict:
            # The array of all the bounds is up-to-date.
            return value_as_array

        if not as_dict:
            return self.convert_dict_to_array(
                value_as_dict, variable_names=variable_names
            )

        if not variable_names:
            return value_as_dict

        return {name: value_as_dict[name] for name in variable_names}

    def set_lower_bound(
        self, name: str, lower_bound: Number | Iterable[Number]
    ) -> None:
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The value of the lower bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        self.__check_known_variable(name)

        self._variables[name].lower_bound = lower_bound
        self._add_norm_policy(name)
        self.__norm_data_is_computed = False

    def set_upper_bound(
        self,
        name: str,
        upper_bound: Number | Iterable[Number],
    ) -> None:
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The value of the upper bound.

        Raises:
            ValueError: If the variable does not exist.
        """
        self.__check_known_variable(name)

        self._variables[name].upper_bound = upper_bound
        self._add_norm_policy(name)
        self.__norm_data_is_computed = False

    def convert_array_to_dict(
        self,
        x_array: ndarray,
    ) -> dict[str, ndarray]:
        """Convert a design array into a dictionary indexed by the variables names.

        Args:
            x_array: A design value expressed as a NumPy array.

        Returns:
            The design value expressed as a dictionary of NumPy arrays.
        """
        return split_array_to_dict_of_arrays(x_array, self.variable_sizes, self)

    @classmethod
    def __get_common_dtype(
        cls,
        arrays: Iterable[ndarray],
    ) -> dtype[int64] | dtype[float64] | dtype[complex128]:
        """Return the common NumPy data type of the variable arrays.

        Use the following rules by parsing the arrays:

        - there is a complex value: returns `numpy.complex128`,
        - there are real and mixed float/int values: returns `numpy.float64`,
        - there are only integer values: returns `numpy.int64`.

        Args:
            arrays: The arrays to be parsed.
        """
        has_float = False
        has_int = False
        for val_arr in arrays:
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

    def convert_dict_to_array(
        self,
        design_values: Mapping[str, ndarray],
        variable_names: Iterable[str] = (),
    ) -> ndarray:
        """Convert a mapping of design values into a NumPy array.

        Args:
            design_values: The mapping of design values.
            variable_names: The design variables to be considered.
                If empty, consider all the design variables.

        Returns:
            The design values as a NumPy array.

        Notes:
            The data type of the returned NumPy array is the most general data type
            of the values of the mapping ``design_values`` corresponding to
            the keys iterable from ``variables_names``.
        """
        if not variable_names:
            variable_names = self
        data = tuple(design_values[name] for name in variable_names)
        # TODO: remove astype when numpy >= 2,
        # since int62 will be the default on windows.
        return concatenate(data).astype(self.__get_common_dtype(data))

    def get_pretty_table(
        self,
        fields: Sequence[str] = (),
        with_index: bool = False,
        capitalize: bool = False,
        simplify: bool = False,
    ) -> PrettyTable:
        """Build a tabular view of the design space.

        Args:
            fields: The name of the fields to be exported.
                If empty, export all the fields.
            with_index: Whether to show index of names for arrays.
                This is ignored for scalars.
            capitalize: Whether to capitalize the field names
                and replace ``"_"`` by ``" "``.
            simplify: Whether to return a simplified tabular view.

        Returns:
            A tabular view of the design space.
        """
        if not fields:
            fields = self.TABLE_NAMES

        if capitalize:
            field_names = [field.capitalize().replace("_", " ") for field in fields]
        else:
            field_names = fields

        table = PrettyTable(field_names)
        table.custom_format = _format_value_in_pretty_table_16
        for name, variable in self._variables.items():
            curr = self.__current_value.get(name)
            name_template = f"{name}"
            if with_index and variable.size > 1:
                name_template += "[{index}]"

            for i in range(variable.size):
                data = {
                    "name": name_template.format(name=name, index=i),
                    "value": None,
                    "lower_bound": float("-inf"),
                    "upper_bound": float("inf"),
                    "type": variable.type,
                }
                data["lower_bound"] = variable.lower_bound[i]
                data["upper_bound"] = variable.upper_bound[i]
                if curr is not None:
                    value = curr[i]
                    # The current value of a float variable can be a complex array
                    # when approximating gradients with complex step.
                    if variable.type == "float":
                        value = value.real

                    data["value"] = value

                table.add_row([data[key] for key in fields])

        for name in ("Name", "Type") if capitalize else ("name", "type"):
            table.align[name] = "l"
        return table

    def to_hdf(
        self,
        file_path: str | Path,
        append: bool = False,
        hdf_node_path: str = "",
    ) -> None:
        """Export the design space to an HDF file.

        Args:
            file_path: The path to the file to export the design space.
            append: If ``True``, appends the data in the file.
            hdf_node_path: The path of the HDF node in which
                the design space should be exported.
                If empty, the root node is considered.
        """
        mode = "a" if append else "w"

        with h5py.File(file_path, mode) as h5file:
            if hdf_node_path:
                h5file = h5file.require_group(hdf_node_path)
            design_vars_grp = h5file.require_group(self.DESIGN_SPACE_GROUP)
            design_vars_grp.create_dataset(
                self.NAMES_GROUP,
                data=array(self.variable_names, dtype=bytes_),
            )

            for name, variable in self._variables.items():
                var_grp = design_vars_grp.require_group(name)
                var_grp.create_dataset(self.SIZE_GROUP, data=variable.size)
                var_grp.create_dataset(self.LB_GROUP, data=variable.lower_bound)
                var_grp.create_dataset(self.UB_GROUP, data=variable.upper_bound)
                data_array = array([variable.type] * variable.size, dtype="bytes")
                var_grp.create_dataset(
                    self.VAR_TYPE_GROUP,
                    data=data_array,
                    dtype=data_array.dtype,
                )
                value = self.__current_value.get(name)
                if value is not None:
                    var_grp.create_dataset(self.VALUE_GROUP, data=self.__to_real(value))

    @classmethod
    def from_hdf(cls, file_path: str | Path, hdf_node_path: str = "") -> DesignSpace:
        """Create a design space from an HDF file.

        Args:
            file_path: The path to the HDF file.
            hdf_node_path: The path of the HDF node from which
                the database should be imported.
                If empty, the root node is considered.

        Returns:
            The design space defined in the file.
        """
        design_space = cls()
        with h5py.File(file_path) as h5file:
            h5file = get_hdf5_group(h5file, hdf_node_path)
            design_vars_grp = get_hdf5_group(h5file, design_space.DESIGN_SPACE_GROUP)
            variable_names = get_hdf5_group(design_vars_grp, design_space.NAMES_GROUP)

            for name in variable_names:
                name = name.decode()
                var_group = get_hdf5_group(design_vars_grp, name)
                l_b = design_space.__read_opt_attr_array(
                    var_group,
                    design_space.LB_GROUP,
                )
                u_b = design_space.__read_opt_attr_array(
                    var_group,
                    design_space.UB_GROUP,
                )
                var_type = design_space.__read_opt_attr_array(
                    var_group,
                    design_space.VAR_TYPE_GROUP,
                )[0]
                value = design_space.__read_opt_attr_array(
                    var_group,
                    design_space.VALUE_GROUP,
                )
                size = get_hdf5_group(var_group, design_space.SIZE_GROUP)[()]
                design_space.add_variable(name, size, var_type, l_b, u_b, value)

        design_space.check()
        return design_space

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

    @classmethod
    def from_file(
        cls,
        file_path: str | Path,
        hdf_node_path: str = "",
        **options: Any,
    ) -> DesignSpace:
        """Create a design space from a file.

        Args:
            file_path: The path to the file.
                If the extension starts with `"hdf"`,
                the file will be considered as an HDF file.
            hdf_node_path: The path of the HDF node from which
                the database should be imported.
                If empty, the root node is considered.
            **options: The keyword reading options.

        Returns:
            The design space defined in the file.
        """
        if h5py.is_hdf5(file_path):
            return cls.from_hdf(file_path, hdf_node_path)
        return cls.from_csv(file_path, **options)

    def to_file(self, file_path: str | Path, **options) -> None:
        """Save the design space.

        Args:
            file_path: The file path to save the design space.
                If the extension starts with `"hdf"`,
                the design space will be saved in an HDF file.
            **options: The keyword reading options.
        """
        file_path = Path(file_path)
        if file_path.suffix.startswith((".hdf", ".h5")):
            self.to_hdf(file_path, append=options.get("append", False))
        else:
            self.to_csv(file_path, **options)

    def to_csv(
        self,
        output_file: str | Path,
        fields: Sequence[str] = (),
        header_char: str = "",
        **table_options: Any,
    ) -> None:
        """Export the design space to a CSV file.

        Args:
            output_file: The path to the file.
            fields: The fields to be exported.
                If empty, export all fields.
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

    @classmethod
    def from_csv(cls, file_path: str | Path, header: Iterable[str] = ()) -> DesignSpace:
        """Create a design space from a CSV file.

        Args:
            file_path: The path to the CSV file.
            header: The names of the fields saved in the file.
                If empty, read them in the file.

        Returns:
            The design space defined in the file.

        Raises:
            ValueError: If the file does not contain the minimal variables
                in its header.
        """
        design_space = cls()
        float_data = genfromtxt(file_path, dtype="float")
        str_data = genfromtxt(file_path, dtype="str")
        if header:
            start_read = 0
        else:
            header = str_data[0, :].tolist()
            start_read = 1
        if not set(cls.MINIMAL_FIELDS).issubset(set(header)):
            msg = (
                f"Malformed DesignSpace input file {file_path} does not contain "
                "minimal variables in header:"
                f"{cls.MINIMAL_FIELDS}; got instead: {header}."
            )
            raise ValueError(msg)
        col_map = {field: i for i, field in enumerate(header)}
        var_names = str_data[start_read:, 0].tolist()
        unique_names = []
        prev_name = None
        for name in var_names:  # set([]) does not preserve order !
            if name not in unique_names:
                unique_names.append(name)
                prev_name = name
            elif prev_name != name:
                msg = (
                    f"Malformed DesignSpace input file {file_path} contains some "
                    f"variables ({file_path}) in a non-consecutive order."
                )
                raise ValueError(msg)

        k = start_read
        lower_bounds_field = cls.MINIMAL_FIELDS[1]
        upper_bounds_field = cls.MINIMAL_FIELDS[2]
        value_field = cls.TABLE_NAMES[2]
        var_type_field = cls.TABLE_NAMES[-1]
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
                var_type = str_data[k, col_map[var_type_field]]
            else:
                var_type = cls.DesignVariableType.FLOAT
            design_space.add_variable(name, size, var_type, l_b, u_b, value)
            k += size
        design_space.check()
        return design_space

    def _get_string_representation(
        self,
        use_html: bool,
        title: str = "",
        simplify: bool = False,
    ) -> str:
        """Return the string representation of the design space.

        Args:
            use_html: Whether the string representation is HTML code.
            title: The title of the table.
                If empty, use the name of the class.
            simplify: Whether to return a simplified string representation.

        Returns:
            The string representation of the design space.
        """
        if not title:
            title = " ".join(
                self.__CAMEL_CASE_REGEX.findall(self.__class__.__name__),
            ).lower()

        title = title.capitalize()
        post_title = ": " if self.name else ":"
        new_line = "<br/>" if use_html else "\n"
        pretty_table = self.get_pretty_table(
            with_index=True,
            capitalize=True,
            simplify=simplify,
        )
        pretty_table_method = "get_html_string" if use_html else "get_string"
        table = getattr(pretty_table, pretty_table_method)()
        return f"{title}{post_title}{self.name}{new_line}{table}"

    def __repr__(self) -> str:
        return self._get_string_representation(False)

    def __str__(self) -> str:
        return self._get_string_representation(False, simplify=True)

    def _repr_html_(self) -> str:
        return REPR_HTML_WRAPPER.format(self._get_string_representation(True))

    def project_into_bounds(
        self,
        x_c: ndarray,
        normalized: bool = False,
    ) -> ndarray:
        """Project a vector onto the bounds, using a simple coordinate wise approach.

        Args:
            normalized: If ``True``, then the vector is assumed to be normalized.
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
        l_inds = (x_c < l_b).nonzero()
        x_p[l_inds] = l_b[l_inds]

        u_inds = (x_c > u_b).nonzero()
        x_p[u_inds] = u_b[u_inds]
        return x_p

    def __contains__(
        self,
        variable: str,
    ) -> bool:
        return variable in self._variables

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __eq__(
        self,
        other: DesignSpace,
    ) -> bool:
        if not isinstance(other, self.__class__):
            return False

        variables = self._variables
        other_variables = other._variables
        if variables.keys() != other_variables.keys():
            return False

        for name in self:
            if variables[name] != other_variables[name]:
                return False

        current_value = self._current_value
        other_current_value = other._current_value
        if current_value.keys() != other_current_value.keys():
            return False

        for name, value in current_value.items():
            if not array_equal(other_current_value[name], value):
                return False

        return True

    def extend(
        self,
        other: DesignSpace,
    ) -> None:
        """Extend the design space with another design space.

        Args:
            other: The design space to be appended to the current one.
        """
        for name, variable in other._variables.items():
            self.add_variable(
                name,
                variable.size,
                variable.type,
                variable.lower_bound,
                variable.upper_bound,
                other._current_value.get(name),
            )

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
        if current_name not in self:
            msg = f"The variable {current_name} is not in the design space."
            raise ValueError(msg)

        for dictionary in [self.normalize, self._variables, self.__names_to_indices]:
            dictionary[new_name] = dictionary.pop(current_name)

        current_value = self._current_value.pop(current_name, None)
        if current_value is not None:
            self._current_value[new_name] = current_value

    def initialize_missing_current_values(self) -> None:
        """Initialize the current values of the design variables when missing.

        Use:

        - the center of the design space when the lower and upper bounds are finite,
        - the lower bounds when the upper bounds are infinite,
        - the upper bounds when the lower bounds are infinite,
        - zero when the lower and upper bounds are infinite.
        """
        for name, variable in self._variables.items():
            if self.__current_value.get(name) is not None:
                continue

            current_value = []
            for l_b_i, u_b_i in zip(variable.lower_bound, variable.upper_bound):
                if l_b_i == -inf:
                    current_value_i = 0 if u_b_i == inf else u_b_i
                else:
                    current_value_i = l_b_i if u_b_i == inf else (l_b_i + u_b_i) / 2

                current_value.append(current_value_i)

            if self.DesignVariableType.FLOAT in variable.type:
                var_type = self.DesignVariableType.FLOAT
            else:
                var_type = self.DesignVariableType.INTEGER

            self.set_current_variable(
                name,
                array(
                    current_value,
                    dtype=self.VARIABLE_TYPES_TO_DTYPES[var_type],
                ),
            )

    def add_variables_from(self, space: DesignSpace, *names: str) -> None:
        """Add variables from another variable space.

        Args:
            space: The other variable space.
            *names: The names of the variables.
        """
        for name in names:
            self._add_variable_from(space, name)

    def _add_variable_from(self, space: DesignSpace, name: str) -> None:
        """Add a variable from another variable space.

        Args:
            space: The other variable space.
            name: The name of the variable.
        """
        variable = space._variables[name]
        self.add_variable(
            name,
            size=variable.size,
            type_=variable.type,
            lower_bound=variable.lower_bound,
            upper_bound=variable.upper_bound,
            value=space._current_value.get(name),
        )

    def to_scalar_variables(self) -> DesignSpace:
        """Create a new design space with the variables splitted into scalar variables.

        Returns:
            The design space of scalar variables.
        """
        design_space = self.__class__()
        for name in self:
            size = self.get_size(name)
            type_ = self.get_type(name)
            lower_bounds = self.get_lower_bound(name)
            upper_bounds = self.get_upper_bound(name)

            try:
                current_value = self.get_current_value([name])
            except KeyError:
                # The variable has no current value.
                current_value = full(size, None)

            for index, indexed_name in enumerate(self.get_indexed_variable_names(name)):
                design_space.add_variable(
                    indexed_name,
                    1,
                    type_,
                    lower_bounds[index],
                    upper_bounds[index],
                    current_value[index],
                )

        return design_space

    @property
    def enable_integer_variables_normalization(self) -> bool:
        """Whether to enable the normalization of integer variables.

        Note:
            Switching the normalization of integer variables shall trigger
            the (re-)computation of the normalization data
            at the next normalization (or unnormalization).
        """
        return self.__normalize_integer_variables

    @enable_integer_variables_normalization.setter
    def enable_integer_variables_normalization(self, value: bool) -> None:
        if value != self.__normalize_integer_variables:
            self.__normalize_integer_variables = value
            # Update the normalization policies of the integer variables
            for name, variable in self._variables.items():
                if variable.type == self.DesignVariableType.INTEGER:
                    self._add_norm_policy(name)

            self.__norm_data_is_computed = False
