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
"""Base class for spaces of variables."""

from __future__ import annotations

from abc import abstractmethod
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar
from typing import cast

from numpy import concatenate

from gemseo.space._core.codec import concatenate_values as _convert_dict_to_array
from gemseo.space._core.codec import split_full_value as _convert_array_to_dict
from gemseo.space._core.rendering import render_html
from gemseo.space._core.rendering import render_string
from gemseo.space._core.variables import Variables
from gemseo.space.variables_view import VariablesView
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta
from gemseo.util.string import _convert_camel_case_to_lower_case_words
from gemseo.util.string import convert_strings_to_iterable
from gemseo.util.string import pretty_str
from gemseo.util.string import repr_variable

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Self

    from numpy import ndarray
    from prettytable import PrettyTable

    from gemseo.space.variable import BaseVariable
    from gemseo.util.typing import IntegerArray

_VariablesT = TypeVar("_VariablesT", bound="Variables[Any]")
"""The type of the registry of the variables of the space."""

_VariablesViewT = TypeVar("_VariablesViewT", bound="VariablesView[Any]")
"""The type of the read-only view over the registry of the variables of the space."""


class BaseVariableSpace(
    Generic[_VariablesT, _VariablesViewT], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """Base class for spaces of variables.

    A variable space stores variables,
    together with their names, sizes and types,
    and provides the operations to manipulate them.
    What else defines a variable depends on the kind of space,
    e.g. bounds for a design space
    and probability distributions for a random space.

    The variables can be read through the
    [variables][gemseo.space.base.BaseVariableSpace.variables] view
    and can only be mutated by the methods of the space.

    Whatever its kind,
    a space offers a
    [reference_value][gemseo.space.base.BaseVariableSpace.reference_value],
    namely the single representative value per variable
    that a consumer of the space uses when it needs one,
    e.g. the current value of a design space
    and the mean of a random space.
    """

    name: str
    """The name of the space."""

    _variables: _VariablesT
    """The versioned variables."""

    __variables_view: _VariablesViewT
    """The read-only view over the versioned variables."""

    _variables_class: ClassVar[type[Variables[Any]]] = Variables
    """The class of the registry of the variables of the space."""

    _variables_view_class: ClassVar[type[VariablesView[Any]]] = VariablesView
    """The class of the read-only view over the registry of the variables."""

    def __init__(self, name: str = "") -> None:
        """
        Args:
            name: The name to be given to the space of variables.
                If empty, the space of variables is unnamed.
        """  # noqa: D205, D212
        self.name = name
        # A ClassVar cannot hold a type variable,
        # so bridge the declared registry class to the parameterized attribute here.
        self._variables = cast("_VariablesT", self._variables_class())
        self.__variables_view = cast(
            "_VariablesViewT", self._variables_view_class(self._variables)
        )

    @property
    def variables(self) -> _VariablesViewT:
        """The registry of the variables of the space (read-only).

        This view is live:
        it reflects the mutations of the space.

        The variables are mutated by the methods of the space,
        namely
        [add_variable][gemseo.space.base.BaseVariableSpace.add_variable],
        [remove_variable][gemseo.space.base.BaseVariableSpace.remove_variable],
        [rename_variable][gemseo.space.base.BaseVariableSpace.rename_variable],
        [filter][gemseo.space.base.BaseVariableSpace.filter]
        and
        [filter_dimensions][gemseo.space.base.BaseVariableSpace.filter_dimensions],
        plus
        [add_copula][gemseo.space.random.RandomSpace.add_copula]
        for a random space.
        """
        return self.__variables_view

    @abstractmethod
    def add_variable(self, name: str, *args: Any, **kwargs: Any) -> None:
        """Add a variable to the space.

        Args:
            name: The name of the variable.
            *args: The positional arguments defining the variable.
            **kwargs: The keyword arguments defining the variable.

        Raises:
            ValueError: When the variable name already exists.
        """

    def _add_variable(self, name: str, variable: BaseVariable) -> None:
        """Register a variable whose name does not already exist.

        Args:
            name: The name of the variable.
            variable: The variable.

        Raises:
            ValueError: When the variable name already exists.
        """
        if name in self._variables:
            msg = f"The variable {name!r} already exists."
            raise ValueError(msg)

        self._variables[name] = variable

    @property
    def dimension(self) -> int:
        """The dimension of the space, i.e., the sum of the variable sizes.

        Note:
            `len(space)` is the number of variables, not the dimension.
        """
        return self._variables.size

    def get_variables_indexes(
        self,
        variable_names: Collection[str],
        use_space_order: bool = True,
    ) -> IntegerArray:
        """Return the indexes of a full vector corresponding to variables names.

        Args:
            variable_names: The names of the variables.
            use_space_order: Whether to order the indexes
                according to the order of the variables names in the space.
                Otherwise,
                the indexes will be ordered
                in the same order as the variables names were required.

        Returns:
            The indexes of a full vector corresponding to the variables names.
        """
        if use_space_order:
            names = [name for name in self if name in variable_names]
        else:
            names = variable_names

        name_to_indices = self._variables.name_to_indices
        return concatenate([name_to_indices[name] for name in names])

    def get_indexed_variable_names(
        self, variable_names: str | Sequence[str] = ()
    ) -> list[str]:
        """Create the names of the components of variables.

        If the size of the variable is equal to 1,
        its name remains unaltered.
        Otherwise,
        it concatenates the name of the variable and the index of the component.

        Args:
            variable_names: The names of the variables.
                If `empty`, use all the variables.

        Returns:
            The names of the components of the variables.
        """
        names: Iterable[str] = (
            convert_strings_to_iterable(variable_names)
            if variable_names
            else self._variables
        )

        var_ind_names = []
        for variable_name in names:
            size = self._variables[variable_name].size
            var_ind_names.extend([
                repr_variable(variable_name, i, size) for i in range(size)
            ])

        return var_ind_names

    def convert_array_to_dict(self, x_vect: ndarray) -> dict[str, ndarray]:
        """Convert a full vector into a dictionary indexed by the variables names.

        Args:
            x_vect: A value of the space expressed as a NumPy array.

        Returns:
            The value of the space expressed as a dictionary of NumPy arrays.
        """
        return _convert_array_to_dict(x_vect, self._variables)

    def convert_dict_to_array(
        self,
        variable_values: Mapping[str, ndarray],
        variable_names: Iterable[str] = (),
    ) -> ndarray:
        """Convert a mapping of values into a NumPy array.

        Args:
            variable_values: The mapping of values.
            variable_names: The variables to be considered.
                If empty, consider all the variables.

        Returns:
            The values as a NumPy array.

        Notes:
            The data type of the returned NumPy array is
            the most general data type of the values of the mapping `variable_values`
            corresponding to the keys iterable from `variable_names`.
        """
        if not variable_names:
            variable_names = self._variables
        return _convert_dict_to_array(variable_values, variable_names)

    @property
    def _current_value(self) -> Mapping[str, ndarray | None]:
        """The current value of the space.

        Maps every variable to its current value,
        or to `None` when the variable has no value.
        It is empty for a space defining no current value,
        e.g. a random space,
        and setting it is then a no-op.
        """
        return {}

    @_current_value.setter
    def _current_value(self, value: Mapping[str, ndarray | None]) -> None:
        # A space defining no current value has nothing to write.
        return

    def _to_complex(self) -> None:
        """Cast the current value to complex.

        A space defining no current value has nothing to cast,
        e.g. a random space,
        so the base implementation does nothing.
        """

    @property
    @abstractmethod
    def reference_value(self) -> dict[str, ndarray]:
        """The reference value of the space.

        This value is what a consumer of the space uses
        when it needs a single representative value per variable,
        e.g. a formulation seeding the default input values
        of the top-level disciplines of a scenario.
        Each kind of space defines it in its own terms,
        as the map from a variable name to its reference value,
        possibly empty when the space defines no such value.
        """

    def check(self) -> None:
        """Check the state of the space.

        Raises:
            ValueError: If the space is empty.
        """
        if not self._variables:
            kind = _convert_camel_case_to_lower_case_words(self.__class__.__name__)
            msg = f"The {kind} is empty."
            raise ValueError(msg)

    def remove_variable(self, name: str) -> None:
        """Remove a variable from the space.

        Args:
            name: The name of the variable.
        """
        del self._variables[name]

    def rename_variable(self, current_name: str, new_name: str) -> None:
        """Rename a variable.

        Args:
            current_name: The current name of the variable.
            new_name: The new name of the variable.

        Raises:
            ValueError: When `new_name` is already the name of another variable.
        """
        self._variables.rename(current_name, new_name)

    def filter(self, keep_variables: str | Iterable[str], copy: bool = False) -> Self:
        """Filter the space to keep a subset of variables.

        Args:
            keep_variables: The names of the variables to be kept.
            copy: If `True`, then a copy of the space is filtered,
                otherwise the space itself is filtered.

        Returns:
            Either the filtered original space or a copy.
        """
        # Materialize the names, as they are read twice below
        # and an iterator would be exhausted by the validation,
        # leaving nothing to keep.
        names_to_keep = list(convert_strings_to_iterable(keep_variables))
        # Validate the requested names before removing anything,
        # so an unknown name does not leave the space partially emptied.
        for name in names_to_keep:
            # Validate via __getitem__ so an unknown name raises before mutating.
            self._variables[name]

        unique_names_to_keep = set(names_to_keep)
        space = deepcopy(self) if copy else self
        for name in list(self._variables):
            if name not in unique_names_to_keep:
                space.remove_variable(name)
        return space

    def filter_dimensions(self, name: str, dimensions: Sequence[int]) -> Self:
        """Filter the space to keep a subset of dimensions for a variable.

        Args:
            name: The name of the variable.
            dimensions: The dimensions of the variable to be kept,
                between $0$ and $d-1$
                where $d$ is the number of dimensions of the variable.

        Returns:
            The filtered space.

        Raises:
            ValueError: If a dimension is repeated,
                or if a dimension does not exist,
                or if no dimension is to be kept.
        """
        # Read the size through the registry so an unknown name raises before mutating.
        size = self._variables[name].size
        # Filtering keeps a subset of the dimensions,
        # so a dimension repeated would duplicate the component instead,
        # which the kinds of variable do not even agree on:
        # a scalar variable rejects such a request
        # while an interval variable honors it silently.
        if len(set(dimensions)) != len(dimensions):
            msg = f"The dimensions of variable {name!r} cannot be repeated."
            raise ValueError(msg)

        nonexistent_dimensions = {i for i in dimensions if not 0 <= i < size}
        if nonexistent_dimensions:
            plural = len(nonexistent_dimensions) > 1
            msg = (
                f"Dimension{'s' if plural else ''}"
                f" {pretty_str(nonexistent_dimensions)}"
                f" of variable '{name}' {'do' if plural else 'does'} not exist."
            )
            raise ValueError(msg)

        self._filter_dimensions(name, dimensions)
        return self

    def _filter_dimensions(self, name: str, dimensions: Sequence[int]) -> None:
        """Keep a subset of dimensions for a variable, once they are validated.

        Args:
            name: The name of the variable.
            dimensions: The dimensions of the variable to be kept.
        """
        self._variables.filter_components(name, dimensions)

    @abstractmethod
    def get_pretty_table(
        self,
        fields: Sequence[str] = (),
        with_index: bool = False,
        capitalize: bool = False,
    ) -> PrettyTable:
        """Build a tabular view of the space.

        Args:
            fields: The name of the fields to be exported.
                If empty, export all the fields.
            with_index: Whether to show index of names for arrays.
                This is ignored for scalars.
            capitalize: Whether to capitalize the field names
                and replace `"_"` by `" "`.

        Returns:
            A tabular view of the space.
        """

    def _render_footer(self) -> str:
        """Render the part of the representation that the tabular view cannot carry.

        A tabular view has one row per component of the space,
        so a piece of information relating several variables has no place in it.

        Returns:
            The footer of the representation of the space, empty by default.
        """
        return ""

    @abstractmethod
    def transform_vect(self, x_vect: ndarray) -> ndarray:
        """Map a point of the space to the unit hypercube.

        Args:
            x_vect: A point of the space.

        Returns:
            A vector with components in $[0,1]$.
        """

    @abstractmethod
    def untransform_vect(self, x_vect: ndarray, no_check: bool = False) -> ndarray:
        """Map a point of the unit hypercube to the space.

        Args:
            x_vect: A vector with components in $[0,1]$.
            no_check: Whether to check if the components are in $[0,1]$.

        Returns:
            A point of the space.
        """

    @contextmanager
    def _prepare_untransformation(
        self, check_boundedness: bool = False
    ) -> Iterator[None]:
        """Set the space up for the untransformation of unit vectors, then restore it.

        The mapping from the unit hypercube can require the space to be temporarily
        configured, e.g. the normalization of the integer variables of a design space.
        The base implementation does nothing.

        Args:
            check_boundedness: Whether to check
                that every component of the space can be untransformed.

        Yields:
            Nothing; the space is set up for the duration of the context.
        """
        yield

    def __repr__(self) -> str:
        return render_string(self, use_html=False)

    __str__ = __repr__

    def _repr_html_(self) -> str:
        return render_html(self)

    def __contains__(self, variable: str) -> bool:
        return variable in self._variables

    def __len__(self) -> int:
        return len(self._variables)

    def __iter__(self) -> Iterator[str]:
        return iter(self._variables)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        variables = self._variables
        other_variables = other._variables
        if variables.keys() != other_variables.keys():
            return False

        return all(variables[name] == other_variables[name] for name in self)
