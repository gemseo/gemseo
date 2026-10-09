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
"""Bounds for versioned variables."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

from numpy import abs as np_abs
from numpy import array
from numpy import clip
from numpy import equal
from numpy import full
from numpy import inf
from numpy import where
from numpy import zeros

from gemseo.space._core.codec import concatenate_values
from gemseo.space._core.registry_derived_data import RegistryDerivedData
from gemseo.space.variable import BaseNumericVariable
from gemseo.util._numpy import freeze_array

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from gemseo.space._design.variables import DesignVariables
    from gemseo.space.variable import BaseDeterministicVariable
    from gemseo.util.typing import BooleanArray
    from gemseo.util.typing import NumberArray


class Bounds(RegistryDerivedData):
    """Read/write access to the lower/upper bounds of versioned variables."""

    __full_lower_bound: NumberArray
    """The lower bound of the full vector."""

    __full_upper_bound: NumberArray
    """The upper bound of the full vector."""

    def __init__(self, variables: DesignVariables) -> None:
        """
        Args:
            variables: The variables.
        """  # noqa: D205, D212
        super().__init__(variables)
        self._register_guard(self._rebuild)
        self.__full_lower_bound = freeze_array(array([]))
        self.__full_upper_bound = freeze_array(array([]))

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the bounds from a pickled or copied state.

        Args:
            state: The state.
        """
        self.__dict__.update(state)
        self._register_guard(self._rebuild)

    def get_lower_bound(self, name: str) -> NumberArray:
        """Return the lower bound of a variable (read-only).

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite);
            this array is read-only.

        Raises:
            TypeError: If the variable has no bounds.
        """
        # The variable hands out a view of the frozen bound it stores.
        return self.__get_numeric_variable(name).lower_bound

    def get_upper_bound(self, name: str) -> NumberArray:
        """Return the upper bound of a variable (read-only).

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite);
            this array is read-only.

        Raises:
            TypeError: If the variable has no bounds.
        """
        return self.__get_numeric_variable(name).upper_bound

    def __get_numeric_variable(self, name: str) -> BaseNumericVariable:
        """Return a variable that has bounds.

        Args:
            name: The name of the variable.

        Returns:
            The variable.

        Raises:
            TypeError: If the variable has no bounds.
        """
        variable = self._variables[name]
        if not isinstance(variable, BaseNumericVariable):
            msg = f"The {variable.type} variable {name} has no bounds."
            raise TypeError(msg)

        return variable

    def set_lower_bound(
        self, name: str, lower_bound: complex | Iterable[complex]
    ) -> None:
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The lower bound of the variable.

        Raises:
            TypeError: If the variable has no bounds.
        """
        variable = self.__get_numeric_variable(name)
        self._variables[name] = variable.model_copy(update={"lower_bound": lower_bound})

    def set_upper_bound(
        self, name: str, upper_bound: complex | Iterable[complex]
    ) -> None:
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The upper bound of the variable.

        Raises:
            TypeError: If the variable has no bounds.
        """
        variable = self.__get_numeric_variable(name)
        self._variables[name] = variable.model_copy(update={"upper_bound": upper_bound})

    @staticmethod
    def __get_coordinate_bound(
        variable: BaseDeterministicVariable, select_lower_bound: bool
    ) -> NumberArray:
        """Return the bound of the coordinates of a variable.

        A variable without bounds, e.g. a categorical one,
        has coordinates ranging from zero to the number of its categories minus one.

        Args:
            variable: The variable.
            select_lower_bound: Whether to select the lower bound.
                Otherwise, select the upper bound.

        Returns:
            The bound of the coordinates of the variable.
        """
        if isinstance(variable, BaseNumericVariable):
            return variable.lower_bound if select_lower_bound else variable.upper_bound

        if select_lower_bound:
            return zeros(variable.size, dtype=variable.coordinate_type)

        return full(
            variable.size, len(variable.categories) - 1, dtype=variable.coordinate_type
        )

    def _rebuild(self) -> None:
        """Rebuild the bounds of the full vector.

        The bounds describe the coordinates of the full vector.
        """
        variables = self._variables
        self.__full_lower_bound = freeze_array(
            concatenate_values(
                {
                    name: self.__get_coordinate_bound(variable, True)
                    for name, variable in variables.items()
                },
                variables,
            )
        )
        self.__full_upper_bound = freeze_array(
            concatenate_values(
                {
                    name: self.__get_coordinate_bound(variable, False)
                    for name, variable in variables.items()
                },
                variables,
            )
        )

    @property
    def full_lower_bound(self) -> NumberArray:
        """The lower bound of the full vector (read-only)."""
        self._refresh()
        # A view, for the reason given in get_lower_bound.
        return self.__full_lower_bound.view()

    @property
    def full_upper_bound(self) -> NumberArray:
        """The upper bound of the full vector (read-only)."""
        self._refresh()
        return self.__full_upper_bound.view()

    @overload
    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[False] = False,
    ) -> NumberArray: ...

    @overload
    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[True] = True,
    ) -> dict[str, NumberArray]: ...

    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> NumberArray | dict[str, NumberArray]:
        """Return the lower bounds of variables.

        Args:
            names: The names of the variables.
                If empty, return the lower bounds of all the variables.
            as_dict: Whether to return a dictionary keyed by variable name.
                The variables without bounds, e.g. categorical ones,
                are then left out.
                Otherwise, return an array
                in which a variable without bounds has the range of its coordinates.

        Returns:
            The lower bounds of the variables;
            the arrays are read-only.
        """
        return self.__select(names, as_dict, True)

    @overload
    def get_upper_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[False] = False,
    ) -> NumberArray: ...

    @overload
    def get_upper_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[True] = True,
    ) -> dict[str, NumberArray]: ...

    def get_upper_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> NumberArray | dict[str, NumberArray]:
        """Return the upper bounds of variables.

        Args:
            names: The names of the variables.
                If empty, return the upper bounds of all the variables.
            as_dict: Whether to return a dictionary keyed by variable name.
                The variables without bounds, e.g. categorical ones,
                are then left out.
                Otherwise, return an array
                in which a variable without bounds has the range of its coordinates.

        Returns:
            The upper bounds of the variables;
            the arrays are read-only.
        """
        return self.__select(names, as_dict, False)

    def __select(
        self,
        names: Sequence[str],
        as_dict: bool,
        select_lower_bounds: bool,
    ) -> NumberArray | dict[str, NumberArray]:
        """Select the bounds of variables, building only the requested ones.

        Args:
            names: The names of the variables (empty means all).
            as_dict: Whether to return a dictionary keyed by variable name.
            select_lower_bounds: Whether to select the lower bounds.
                Otherwise, select the upper bounds.

        Returns:
            The selected bounds (read-only).
        """
        if not names:
            if not as_dict:
                # Fast path returns the cached bound of the full vector.
                if select_lower_bounds:
                    return self.full_lower_bound

                return self.full_upper_bound

            names = self._variables

        if as_dict:
            # A variable without bounds, e.g. a categorical one, is left out.
            return {
                name: self.__get_coordinate_bound(
                    self._variables[name], select_lower_bounds
                )
                for name in names
                if isinstance(self._variables[name], BaseNumericVariable)
            }

        # A variable without bounds, e.g. a categorical one,
        # has the range of its coordinates here,
        # so that the array is aligned with the design vector.
        name_to_bound = {
            name: self.__get_coordinate_bound(
                self._variables[name], select_lower_bounds
            )
            for name in names
        }

        # Freeze for consistency: every bound handed out is read-only.
        return freeze_array(concatenate_values(name_to_bound, names))

    def get_active_bounds_masks(
        self,
        name_to_value: Mapping[str, NumberArray],
        atol: float = 1e-8,
    ) -> tuple[dict[str, BooleanArray], dict[str, BooleanArray]]:
        """Compute the active lower-bound and upper-bound mask of a point.

        Args:
            name_to_value: The point keyed by variable name.
            atol: The absolute tolerance of comparison of a scalar with a bound.

        Returns:
            A map from a variable name to an active lower-bound mask,
            followed by a map from a variable name to an active upper-bound mask.
        """
        active_lower_bound: dict[str, BooleanArray] = {}
        active_upper_bound: dict[str, BooleanArray] = {}
        for name, variable in self._variables.items():
            if not isinstance(variable, BaseNumericVariable):
                # A variable without bounds has no active bound.
                no_active_bound = zeros(variable.size, dtype=bool)
                active_lower_bound[name] = no_active_bound
                active_upper_bound[name] = no_active_bound.copy()
                continue

            lower_bound = self.get_lower_bound(name)
            lower_bound = where(equal(lower_bound, None), -inf, lower_bound)
            upper_bound = self.get_upper_bound(name)
            upper_bound = where(equal(upper_bound, None), inf, upper_bound)
            value = name_to_value[name]
            active_lower_bound[name] = np_abs(value - lower_bound) <= atol
            active_upper_bound[name] = np_abs(value - upper_bound) <= atol
        return active_lower_bound, active_upper_bound

    def clip_to_bounds(
        self,
        full_value: NumberArray,
        normalized: bool = False,
    ) -> NumberArray:
        """Clip a full value to the bounds, component-wise.

        Args:
            full_value: The full value.
            normalized: Whether the full value is in the normalized space.

        Returns:
            The clipped full value.
        """
        if normalized:
            return clip(full_value, 0, 1)

        return clip(full_value, self.full_lower_bound, self.full_upper_bound)
