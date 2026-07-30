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

A [DesignSpace][gemseo.space.design.DesignSpace] represents the optimization
unknowns, a.k.a. the design variables, at a given state: their names, sizes, types,
bounds and current values.

Variables can be added with
[DesignSpace.add_variable()][gemseo.space.design.DesignSpace.add_variable],
removed with
[DesignSpace.remove_variable()][gemseo.space.design.DesignSpace.remove_variable],
and filtered with
[DesignSpace.filter()][gemseo.space.design.DesignSpace.filter].
Getters and setters give access to each variable property.

A [DesignSpace][gemseo.space.design.DesignSpace] can be stored in a CSV or HDF
file.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Mapping
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import Literal
from typing import overload

from docstring_inheritance import GoogleDocstringInheritanceMeta
from numpy import array_equal
from numpy import atleast_1d
from numpy import concatenate
from numpy import full
from numpy import inf
from numpy import ndarray

from gemseo.optimization.result import OptimizationResult
from gemseo.space._variable import TYPE_MAP
from gemseo.space._variable import DataType
from gemseo.space._variable import Variable
from gemseo.space.design import _checking
from gemseo.space.design import _io as _design_space_io
from gemseo.space.design import _view
from gemseo.space.design._bounds import Bounds
from gemseo.space.design._codec import concatenate_values as _convert_dict_to_array
from gemseo.space.design._codec import split_full_value as _convert_array_to_dict
from gemseo.space.design._integer_rounder import IntegerRounder
from gemseo.space.design._normalizer import Normalizer
from gemseo.space.design._value import Value
from gemseo.space.design._variables import Variables
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import pretty_str
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence

    from numpy import float64
    from numpy import int64
    from prettytable import PrettyTable

    from gemseo.util.read_only_mapping import ReadOnlyMapping
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import RealOrComplexArrayT
    from gemseo.util.typing import StrPath

LOGGER = logging.getLogger(__name__)


class DesignSpace(metaclass=GoogleDocstringInheritanceMeta):
    """Description of a design space.

    It defines a set of variables from their names, sizes, types and bounds.

    In addition,
    it provides the current values of these variables
    that can be used as the initial solution of
    an [OptimizationProblem][gemseo.optimization.problem.OptimizationProblem].
    """

    name: str
    """The name of the space."""

    _variables: Variables
    """The versioned variables."""

    _bounds: Bounds
    """The bounds of the variables."""

    _integer_rounder: IntegerRounder
    """The rounder of the integer components of the variables."""

    _normalizer: Normalizer
    """The normalizer of the values of the variables."""

    _current: Value
    """The current value of the variables."""

    DesignVariableType = DataType

    # TODO: API: the values are not dtypes but types, either fix the values or the name.
    VARIABLE_TYPES_TO_DTYPES: Final[dict[str, type[int64 | float64]]] = TYPE_MAP
    """One NumPy `dtype` per design variable type."""

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name to be given to the design space.
                If empty, the design space is unnamed.
        """  # noqa: D205, D212
        self.name = name
        self._variables = Variables()
        self._bounds = Bounds(self._variables)
        self._integer_rounder = IntegerRounder(self._variables)
        self._normalizer = Normalizer(
            self._variables, self._bounds, self._integer_rounder
        )
        self._current = Value(self._variables, self._bounds, self._normalizer)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the design space from a pickled state.

        Supports both the current component-based layout and the flat
        attribute layout of design spaces pickled before the refactoring
        into components.
        """
        if "_bounds" in state:
            self.__dict__.update(state)
            return
        # Pre-refactor pickle: replay the flat layout through the components,
        # recomputing the derived data (indices, normalization policies).
        self.__init__(state.get("name", ""))
        # Seed the flag through the public setter:
        # the _variables attribute is still empty,
        # so there is no normalization policy to recompute.
        self._variables.enable_integer_variables_normalization = bool(
            state.get("_DesignSpace__normalize_integer_variables")
        )
        for name, variable in state.get("_variables", {}).items():
            self._variables[name] = variable
        # Seed a current-value entry for every variable, using the saved value
        # or None (no value), so that each variable always has an entry.
        saved_current_value = state.get("_DesignSpace__current_value", {})
        for name in self._variables:
            self._current.set_variable(name, saved_current_value.get(name))
        # The flat DesignSpace attributes have been replayed through the
        # components above; restore whatever is left so that a subclass state
        # (e.g. the distributions of a ParameterSpace) is not silently dropped.
        # Skip the obsolete flat DesignSpace layout: its private
        # "_DesignSpace__*" internals and these public/protected attributes.
        obsolete_keys = {
            "dimension",
            "name",
            "normalize",
            "_variables",
            "_norm_factor",
            "_norm_factor_inv",
        }
        for key, value in state.items():
            if key not in obsolete_keys and not key.startswith("_DesignSpace__"):
                self.__dict__[key] = value

    @property
    def _current_value(self) -> Mapping[str, ndarray | None]:
        """The current design value.

        Maps every variable to its current value,
        or to `None` when the variable has no value.
        """
        return self._current.name_to_value

    @property
    def dimension(self) -> int:
        """The total dimension of the space, sum of the variable sizes."""
        return self._variables.size

    @property
    def name_to_normalization_mask(self) -> ReadOnlyMapping[str, BooleanArray]:
        """The map from a variable name to a normalization mask."""
        return self._variables.name_to_normalization_mask

    @property
    def normalize(self) -> ReadOnlyMapping[str, BooleanArray]:
        """The map from a variable name to a normalization mask.

        Deprecated:
            Use
            [name_to_normalization_mask][gemseo.space.design.DesignSpace.name_to_normalization_mask]
            instead.
        """
        warnings.warn(
            "DesignSpace.normalize is deprecated; "
            "use DesignSpace.name_to_normalization_mask instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.name_to_normalization_mask

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

    def remove_variable(
        self,
        name: str,
    ) -> None:
        """Remove a variable from the design space.

        Args:
            name: The name of the variable to be removed.
        """
        del self._variables[name]
        self._current.pop(name)

    def filter(
        self,
        keep_variables: str | Iterable[str],
        copy: bool = False,
    ) -> DesignSpace:
        """Filter the design space to keep a subset of variables.

        Args:
            keep_variables: The names of the variables to be kept.
            copy: If `True`, then a copy of the design space is filtered,
                otherwise the design space itself is filtered.

        Returns:
            Either the filtered original design space or a copy.
        """
        keep_variables = convert_strings_to_iterable(keep_variables)
        # Validate the requested names before removing anything,
        # so an unknown name does not leave the design space partially emptied.
        for name in keep_variables:
            # Validate via __getitem__ so an unknown name raises before mutating.
            self._variables[name]
        design_space = deepcopy(self) if copy else self
        for name in self.variable_names:
            if name not in keep_variables:
                design_space.remove_variable(name)
        return design_space

    def filter_dimensions(self, name: str, dimensions: Sequence[int]) -> DesignSpace:
        """Filter the design space to keep a subset of dimensions for a variable.

        Args:
            name: The name of the variable.
            dimensions: The dimensions of the variable to be kept,
                between $0$ and $d-1$
                where $d$ is the number of dimensions of the variable.

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

        had_current_value = self._current_value.get(name) is not None
        if had_current_value:
            sliced_current_value = self.get_current_value([name])[list(dimensions)]
        self._variables.filter_components(name, dimensions)
        if had_current_value:
            self.set_current_variable(name, sliced_current_value)
        return self

    def add_variable(
        self,
        name: str,
        size: int = 1,
        type_: DataType = DesignVariableType.FLOAT,
        lower_bound: complex | Iterable[complex] = -inf,
        upper_bound: complex | Iterable[complex] = inf,
        value: complex | Iterable[complex] | None = None,
    ) -> None:
        r"""Add a variable to the design space.

        Args:
            name: The name of the variable.
            size: The size of the variable.
            type_: Either the type of the variable
                or the types of its components.
            lower_bound: The lower bound of the variable.
                If `None`, use $-\infty$.
            upper_bound: The upper bound of the variable.
                If `None`, use $+\infty$.
            value: The default value of the variable.
                If `None`, do not use a default value.

        Raises:
            ValueError: Either if the variable already exists,
                if a size, type or bound is wrong
                or if the value is not within the bounds.
        """
        if name in self._variables:
            msg = f"The variable {name!r} already exists."
            raise ValueError(msg)

        variable = Variable(
            size=size,
            type=type_,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
        self._variables[name] = variable
        if value is None:
            # Register the variable with no value so that every variable of the
            # design space always has an entry in the current value.
            self._current.set_variable(name, None)
        else:
            try:
                array_value = atleast_1d(value)
                _checking.check_addable_value(self._variables, array_value, name)
                if len(array_value) == 1 and size > 1:
                    array_value = full(size, value)
                self._current.set_variable(
                    name,
                    array_value.astype(
                        self.VARIABLE_TYPES_TO_DTYPES[self.get_type(name)],
                        copy=False,
                    ),
                )
                self._current.check_value(name)
            except ValueError:
                # If a ValueError is raised,
                # we must remove the variable from the design space.
                # When using a python script, this has no interest.
                # When using a notebook, a cell can raise a ValueError,
                # but we can continue to the next cell,
                # and use a design space which contains variables that leads to error.
                self.remove_variable(name)
                raise

    @property
    def name_to_indices(self) -> ReadOnlyMapping[str, range]:
        """The names bound to the indices."""
        return self._variables.name_to_indices

    @property
    def has_current_value(self) -> bool:
        """Check if each variable has a current value.

        Returns:
            Whether the current design value is defined for all variables.
        """
        return self._current.has_value

    @property
    def has_integer_variables(self) -> bool:
        """Check if the design space has at least one integer variable.

        Returns:
            Whether the design space has at least one integer variable.
        """
        return self._variables.has_integer_variable

    def check(self) -> None:
        """Check the state of the design space.

        Raises:
            ValueError: If the design space is empty.
        """
        _checking.check(
            self._variables,
            lambda: self.__check_current_names() if self.has_current_value else None,
        )

    def check_membership(
        self,
        x_vect: Mapping[str, ndarray | None] | ndarray,
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
        _checking.check_membership(
            self._variables, self._bounds, x_vect, variable_names
        )

    def get_active_bounds(
        self,
        x_vect: ndarray | None = None,
        tol: float = 1e-8,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
        """Determine which bound constraints of a design value are active.

        Args:
            x_vect: The design value at which to check the bounds.
                If `None`, use the current design value.
            tol: The tolerance of comparison of a scalar with a bound.

        Returns:
            Whether the components of the lower and upper bound constraints are active,
            the first returned value representing the lower bounds
            and the second one the upper bounds, e.g.

            ```python
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
            ```

            where:

            ```python
                are_x_lower_bounds_active = [True, False]
                are_x_upper_bounds_active = [False, False]
                are_y_lower_bounds_active = [False]
                are_y_upper_bounds_active = [True]
            ```
        """
        if x_vect is None:
            current_x = self._current_value
            self.check_membership(self.get_current_value())
        elif isinstance(x_vect, ndarray):
            current_x = self.convert_array_to_dict(x_vect)
        elif isinstance(x_vect, dict):
            current_x = x_vect
        else:
            msg = f"Expected dict or array for x_vect argument; got {type(x_vect)}."
            raise TypeError(msg)

        return self._bounds.get_active_bounds_masks(current_x, atol=tol)

    def __check_current_names(
        self,
        variable_names: Iterable[str] = (),
    ) -> None:
        """Check that the current design value satisfies the space requirements.

        The completeness of a current value passed as a mapping is validated
        upstream in
        [set_current_value][gemseo.space.design.DesignSpace.set_current_value],
        and a current value passed as an array always covers every variable,
        so only the membership to the bounds is checked here.

        Args:
            variable_names: The names of the variables.
                If empty, use the names of the variables of the design space.

        Raises:
            ValueError: If the current design value is outside the bounds.
        """
        self.check_membership(self._current.name_to_value, variable_names)

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
                If `None`, use all the design variables.
            complex_to_real: Whether to cast complex numbers to real ones.
            as_dict: Whether to return the current design value
                as a dictionary of the form `{variable_name: variable_value}`.
            normalize: Whether to normalize the design values in $[0,1]$
                with the bounds of the variables.
                N.B. Normalization is possible if and only if
                *all* the current design values are set.

        Returns:
            The current design value.

        Raises:
            KeyError: If one of the required design variables has no current value.

        Warning:
            For performance purposes,
            [DesignSpace.get_current_value()][gemseo.space.design.DesignSpace.get_current_value]
            does not return a copy of the current value.
            This means that modifying the returned object
            will make
            the [DesignSpace][gemseo.space.design.DesignSpace] inconsistent
            (the current design value stored as a NumPy array
            and the current design value stored as a dictionary of NumPy arrays
            will be different).
            To modify the returned object
            without impacting the [DesignSpace][gemseo.space.design.DesignSpace],
            you shall copy this object and modify the copy.

        See Also:
            To modify the current value,
            please use
            [DesignSpace.set_current_value()][gemseo.space.design.DesignSpace.set_current_value]
            or
            [DesignSpace.set_current_variable()][gemseo.space.design.DesignSpace.set_current_variable].
        """
        return self._current.get(
            names=variable_names,
            complex_to_real=complex_to_real,
            as_dict=as_dict,
            normalize=normalize,
        )

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
                If `empty`, use all the design variables.

        Returns:
            The name of the components of the variables.
        """
        if variable_names:
            variable_names = convert_strings_to_iterable(variable_names)
        else:
            variable_names = self.variable_names

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

        return concatenate([self.name_to_indices[name] for name in names])

    def normalize_vect(
        self,
        x_vect: RealOrComplexArrayT,
        minus_lb: bool = True,
        out: RealOrComplexArrayT | None = None,
    ) -> RealOrComplexArrayT:
        r"""Normalize a vector of the design space.

        If `minus_lb` is True:

        $$x_u = \frac{x-l_b}{u_b-l_b}$$

        where $l_b$ and $u_b$ are the lower and upper bounds of $x$.

        Otherwise:

        $$x_u = \frac{x}{u_b-l_b}$$

        Unbounded variables are not normalized.

        Args:
            x_vect: The values of the design variables.
            minus_lb: If `True`, remove the lower bounds at normalization.
            out: The array to store the normalized vector.
                Its dtype and shape must be those of the normalized vector.
                If `None`, create a new array.

        Returns:
            The normalized vector.

        Raises:
            ValueError: When `out` cannot store the result.
        """
        return self._normalizer.normalize(
            x_vect,
            self._current.common_dtype,
            subtract_lower_bound=minus_lb,
            out=out,
        )

    def normalize_grad(
        self,
        g_vect: RealOrComplexArrayT,
    ) -> RealOrComplexArrayT:
        r"""Normalize a gradient.

        This method is based on the chain rule:

        $$\frac{df(x)}{dx}
           = \frac{df(x)}{dx_u}\frac{dx_u}{dx}
           = \frac{df(x)}{dx_u}\frac{1}{u_b-l_b}
        $$

        where
        $x_u = \frac{x-l_b}{u_b-l_b}$ is the normalized input vector,
        $x$ is the original input vector
        and $l_b$ and $u_b$ are the lower and upper bounds of $x$.

        Then,
        the normalized gradient reads:

        $$\frac{df(x)}{dx_u} = (u_b-l_b)\frac{df(x)}{dx}$$

        where $\frac{df(x)}{dx}$ is the original one.

        Args:
            g_vect: The original gradient.

        Returns:
            The normalized gradient.
        """
        return self.denormalize_vect(g_vect, minus_lb=False, no_check=True)

    def denormalize_grad(
        self,
        g_vect: RealOrComplexArrayT,
    ) -> RealOrComplexArrayT:
        r"""Denormalize a normalized gradient.

        This method is based on the chain rule:

        $$
           \frac{df(x)}{dx}
           = \frac{df(x)}{dx_u}\frac{dx_u}{dx}
           = \frac{df(x)}{dx_u}\frac{1}{u_b-l_b}
        $$

        where
        $x_u = \frac{x-l_b}{u_b-l_b}$ is the normalized input vector,
        $x$ is the original input vector,
        $\frac{df(x)}{dx_u}$ is the original gradient
        $\frac{df(x)}{dx}$ is the normalized one,
        and $l_b$ and $u_b$ are the lower and upper bounds of $x$.

        Args:
            g_vect: The normalized gradient.

        Returns:
            The original gradient.
        """
        return self.normalize_vect(g_vect, minus_lb=False)

    def denormalize_vect(
        self,
        x_vect: RealOrComplexArrayT,
        minus_lb: bool = True,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> RealOrComplexArrayT:
        """Denormalize a normalized vector of the design space.

        If `minus_lb` is True:

        $$x = x_u(u_b-l_b) + l_b$$

        where
        $x_u$ is the normalized input vector,
        $x$ is the original input vector
        and $l_b$ and $u_b$ are the lower and upper bounds of $x$.

        Otherwise:

        $$x = x_u(u_b-l_b)$$

        Args:
            x_vect: The values of the design variables.
            minus_lb: Whether to remove the lower bounds at normalization.
            no_check: Whether to check if the components are in $[0,1]$.
            out: The array to store the original vector.
                Its dtype and shape must be those of the original vector.
                If `None`, create a new array.

        Returns:
            The original vector.

        Raises:
            ValueError: When `out` cannot store the result.
        """
        return self._normalizer.denormalize(
            x_vect,
            self._current.common_dtype,
            add_lower_bound=minus_lb,
            no_check=no_check,
            out=out,
        )

    def unnormalize_grad(
        self,
        g_vect: RealOrComplexArrayT,
    ) -> RealOrComplexArrayT:
        """Denormalize a normalized gradient.

        Deprecated:
            Use
            [denormalize_grad][gemseo.space.design.DesignSpace.denormalize_grad]
            instead.

        Args:
            g_vect: The normalized gradient.

        Returns:
            The original gradient.
        """
        warnings.warn(
            "DesignSpace.unnormalize_grad is deprecated; "
            "use DesignSpace.denormalize_grad instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.denormalize_grad(g_vect)

    def unnormalize_vect(
        self,
        x_vect: RealOrComplexArrayT,
        minus_lb: bool = True,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> RealOrComplexArrayT:
        """Denormalize a normalized vector of the design space.

        Deprecated:
            Use
            [denormalize_vect][gemseo.space.design.DesignSpace.denormalize_vect]
            instead.

        Args:
            x_vect: The values of the design variables.
            minus_lb: Whether to remove the lower bounds at normalization.
            no_check: Whether to check if the components are in $[0,1]$.
            out: The array to store the original vector.
                Its dtype and shape must be those of the original vector.
                If `None`, create a new array.

        Returns:
            The original vector.
        """
        warnings.warn(
            "DesignSpace.unnormalize_vect is deprecated; "
            "use DesignSpace.denormalize_vect instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.denormalize_vect(
            x_vect, minus_lb=minus_lb, no_check=no_check, out=out
        )

    def transform_vect(
        self,
        x_vect: ndarray,
        out: ndarray | None = None,
    ) -> ndarray:
        """Map a point of the design space to a vector with components in $[0,1]$.

        Args:
            x_vect: A point of the design space.
            out: The array to store the transformed vector.
                Its dtype and shape must be those of the transformed vector.
                If `None`, create a new array.

        Returns:
            A vector with components in $[0,1]$.

        Raises:
            ValueError: When `out` cannot store the result.
        """
        return self.normalize_vect(x_vect, out=out)

    def untransform_vect(
        self,
        x_vect: ndarray,
        no_check: bool = False,
        out: ndarray | None = None,
    ) -> ndarray:
        """Map a vector with components in $[0,1]$ to the design space.

        Args:
            x_vect: A vector with components in $[0,1]$.
            no_check: Whether to check if the components are in $[0,1]$.
            out: The array to store the untransformed vector.
                Its dtype and shape must be those of the untransformed vector.
                If `None`, create a new array.

        Returns:
            A point of the variables space.

        Raises:
            ValueError: When `out` cannot store the result.
        """
        return self.denormalize_vect(x_vect, no_check=no_check, out=out)

    def round_vect(
        self,
        x_vect: ndarray,
        copy: bool = True,
    ) -> ndarray:
        """Round the vector where variables are of integer type.

        Args:
            x_vect: The values to be rounded.
            copy: Whether to round a copy of `x_vect`.

        Returns:
            The rounded values.
        """
        return self._integer_rounder.round(x_vect, copy=copy)

    def set_current_value(
        self,
        value: ndarray | Mapping[str, ndarray | None] | OptimizationResult,
    ) -> None:
        """Set the current design value of all the variables.

        The value of a variable is either a NumPy array
        or `None` when the variable has no value.

        Args:
            value: The value of the current design.
                When passed as a non-empty mapping,
                it must cover all the variables of the design space;
                a variable mapped to `None` is marked as having no value.
                An empty mapping clears the current value of every variable.
                Unknown variable names are ignored.
                When passed as a NumPy array or an
                [OptimizationResult][gemseo.optimization.result.OptimizationResult],
                every variable is given a value.

        Raises:
            ValueError: If the value has a wrong dimension,
                if a mapping does not cover all the variables,
                or if it violates the bounds or integer-type constraints
                of a variable.
            TypeError: If the value is neither a mapping of NumPy arrays,
                a NumPy array nor an
                [OptimizationResult][gemseo.optimization.result.OptimizationResult].
        """
        if isinstance(value, Mapping) and value:
            # An empty mapping clears the current value of every variable;
            # a non-empty mapping must cover all the variables.
            missing = self._variables.keys() - value.keys()
            if missing:
                got = [name for name in value if name in self._variables]
                msg = (
                    f"Expected current_x variables:"
                    f" {pretty_str(self, use_and=True)}; "
                    f"got {pretty_str(got, use_and=True)}."
                )
                raise ValueError(msg)

            # Validate before mutating: a value rejected below must never
            # reach the store, so a caller retains a consistent current value.
            self.check_membership(value)  # ty: ignore[invalid-argument-type]
        elif isinstance(value, ndarray) and value.size == self._variables.size:
            # Validate before mutating, as for the mapping case above.
            # A size mismatch is left for Value.set to reject: converting to a
            # dict here would silently mis-split a wrong-size array instead.
            self.check_membership(self.convert_array_to_dict(value))
        elif (
            isinstance(value, OptimizationResult)
            and value.x_opt.size == self._variables.size
        ):
            self.check_membership(self.convert_array_to_dict(value.x_opt))

        self._current.set(value)
        if self._current.name_to_value:
            self.__check_current_names()

    def set_current_variable(
        self,
        name: str,
        current_value: ndarray | None,
    ) -> None:
        """Set the current value of a single variable.

        Args:
            name: The name of the variable.
            current_value: The current value of the variable,
                or `None` to mark the variable as having no value.

        Raises:
            ValueError: If the value does not match the size of the variable.
        """
        if current_value is not None:
            size = self.get_size(name)
            if current_value.size != size:
                msg = (
                    f"The variable {name} of size {size} "
                    f"cannot be set with an array of size {current_value.size}."
                )
                raise ValueError(msg)

        self._current.set_variable(name, current_value)

    def get_size(
        self,
        name: str,
    ) -> int:
        """Get the size of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The size of the variable.
        """
        return self._variables[name].size

    def get_type(
        self,
        name: str,
    ) -> str:
        """Return the type of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The type of the variable.
        """
        return str(self._variables[name].type)

    def get_lower_bound(self, name: str) -> ndarray:
        """Return the lower bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite);
            this array is read-only.
        """
        return self._bounds.get_lower_bound(name)

    def get_upper_bound(self, name: str) -> ndarray:
        """Return the upper bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite);
            this array is read-only.
        """
        return self._bounds.get_upper_bound(name)

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
                as a dictionary of the form `{variable_name: variable_lower_bound}`.

        Returns:
            The lower bounds of the design variables;
            the arrays are read-only.
        """
        return self._bounds.get_lower_bounds(variable_names, as_dict)

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
                as a dictionary of the form `{variable_name: variable_upper_bound}`.

        Returns:
            The upper bounds of the design variables;
            the arrays are read-only.
        """
        return self._bounds.get_upper_bounds(variable_names, as_dict)

    def set_lower_bound(
        self, name: str, lower_bound: complex | Iterable[complex]
    ) -> None:
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The value of the lower bound.
        """
        self._bounds.set_lower_bound(name, lower_bound)

    def set_upper_bound(
        self,
        name: str,
        upper_bound: complex | Iterable[complex],
    ) -> None:
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The value of the upper bound.
        """
        self._bounds.set_upper_bound(name, upper_bound)

    def convert_array_to_dict(
        self,
        x_vect: ndarray,
    ) -> dict[str, ndarray]:
        """Convert a design array into a dictionary indexed by the variables names.

        Args:
            x_vect: A design value expressed as a NumPy array.

        Returns:
            The design value expressed as a dictionary of NumPy arrays.
        """
        return _convert_array_to_dict(x_vect, self._variables)

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
            of the values of the mapping `design_values` corresponding to
            the keys iterable from `variables_names`.
        """
        if not variable_names:
            variable_names = self._variables
        return _convert_dict_to_array(design_values, variable_names)

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
                and replace `"_"` by `" "`.
            simplify: Whether to return a simplified tabular view.

        Returns:
            A tabular view of the design space.

        Note:
            `simplify` has no effect on the base design space; it is an
            extension point honored by subclasses such as
            [ParameterSpace][gemseo.space.parameter.ParameterSpace].
        """
        return _view.get_pretty_table(
            self,
            fields=fields,
            with_index=with_index,
            capitalize=capitalize,
        )

    def to_hdf(
        self,
        file_path: StrPath,
        append: bool = False,
        hdf_node_path: str = "",
    ) -> None:
        """Export the design space to an HDF file.

        Args:
            file_path: The path to the file to export the design space.
            append: If `True`, appends the data in the file.
            hdf_node_path: The path of the HDF node in which
                the design space should be exported.
                If empty, the root node is considered.
        """
        _design_space_io.to_hdf(
            self, file_path, append=append, hdf_node_path=hdf_node_path
        )

    @classmethod
    def from_hdf(cls, file_path: StrPath, hdf_node_path: str = "") -> DesignSpace:
        """Create a design space from an HDF file.

        Args:
            file_path: The path to the HDF file.
            hdf_node_path: The path of the HDF node from which
                the database should be imported.
                If empty, the root node is considered.

        Returns:
            The design space defined in the file.
        """
        return _design_space_io.from_hdf(cls, file_path, hdf_node_path)

    def to_complex(self) -> None:
        """Cast the current value to complex."""
        self._current.to_complex()

    @classmethod
    def from_file(
        cls,
        file_path: StrPath,
        hdf_node_path: str = "",
        header: Iterable[str] = (),
        delimiter: str = "",
    ) -> DesignSpace:
        """Create a design space from a file.

        Args:
            file_path: The path to the file.
                If the extension starts with `"hdf"`,
                the file will be considered as an HDF file.
            hdf_node_path: The path of the HDF node from which
                the database should be imported.
                If empty, the root node is considered.
            header: The names of the fields saved in the CSV file.
                If empty, read them in the first row of the CSV file.
            delimiter: The string used to separate values for CSV files. If empty,
                any consecutive whitespaces act as delimiter.

        Returns:
            The design space defined in the file.
        """
        return _design_space_io.from_file(
            cls,
            file_path,
            hdf_node_path=hdf_node_path,
            header=header,
            delimiter=delimiter,
        )

    def to_file(
        self,
        file_path: StrPath,
        delimiter: str = " ",
        append: bool = False,
        fields: Sequence[str] = (),
    ) -> None:
        """Save the design space.

        Args:
            file_path: The file path to save the design space.
                If the extension starts with `"hdf"`,
                the design space will be saved in an HDF file.
            delimiter: The string used to separate values for CSV files.
            append: If `True`, appends the data in the HDF file.
            fields: The fields to be exported in the CSV fields.
                If empty, export all fields.
        """
        _design_space_io.to_file(
            self, file_path, delimiter=delimiter, append=append, fields=fields
        )

    def to_csv(
        self,
        output_file: StrPath,
        fields: Sequence[str] = (),
        delimiter: str = " ",
    ) -> None:
        """Export the design space to a CSV file.

        Args:
            output_file: The path to the file.
            fields: The fields to be exported.
                If empty, export all fields.
            delimiter: The string used to separate values.
        """
        _design_space_io.to_csv(self, output_file, fields=fields, delimiter=delimiter)

    @classmethod
    def from_csv(
        cls,
        file_path: StrPath,
        header: Iterable[str] = (),
        delimiter: str = "",
    ) -> DesignSpace:
        """Create a design space from a CSV file.

        Args:
            file_path: The path to the CSV file.
            header: The names of the fields saved in the file.
                If empty, read them in the file.
            delimiter: The string used to separate values. If empty, any consecutive
                whitespaces act as delimiter.

        Returns:
            The design space defined in the file.

        Raises:
            ValueError: If the file does not contain the minimal variables
                in its header.
        """
        return _design_space_io.from_csv(
            cls, file_path, header=header, delimiter=delimiter
        )

    def __repr__(self) -> str:
        return _view.render_string(self, use_html=False)

    def __str__(self) -> str:
        return _view.render_string(self, use_html=False, simplify=True)

    def _repr_html_(self) -> str:
        return _view.render_html(self)

    def project_into_bounds(
        self,
        x_vect: ndarray,
        normalized: bool = False,
    ) -> ndarray:
        """Project a vector onto the bounds, using a simple coordinate wise approach.

        Args:
            x_vect: The vector to be projected onto the bounds.
            normalized: If `True`, then the vector is assumed to be normalized.

        Returns:
            The projected vector.
        """
        return self._bounds.clip_to_bounds(x_vect, normalized=normalized)

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
        self._variables.rename(current_name, new_name)
        self._current.rename(current_name, new_name)

    def initialize_missing_current_values(self) -> None:
        """Initialize the current values of the design variables when missing.

        Use:

        - the center of the design space when the lower and upper bounds are finite,
        - the lower bounds when the upper bounds are infinite,
        - the upper bounds when the lower bounds are infinite,
        - zero when the lower and upper bounds are infinite.
        """
        self._current.initialize_missing()

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
            at the next normalization (or denormalization).
        """
        return self._variables.enable_integer_variables_normalization

    @enable_integer_variables_normalization.setter
    def enable_integer_variables_normalization(self, value: bool) -> None:
        if value != self._variables.enable_integer_variables_normalization:
            self._variables.enable_integer_variables_normalization = value
