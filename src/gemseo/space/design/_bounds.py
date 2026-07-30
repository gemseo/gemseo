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
from numpy import inf
from numpy import where

from gemseo.space.design._codec import concatenate_values
from gemseo.space.design._registry_derived_data import RegistryDerivedData

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from numpy import ndarray

    from gemseo.space.design._variables import Variables


class Bounds(RegistryDerivedData):
    """Read/write access to the lower/upper bounds of versioned variables."""

    __full_lower_bound: ndarray
    """The lower bound of the full vector."""

    __full_upper_bound: ndarray
    """The upper bound of the full vector."""

    def __init__(self, variables: Variables) -> None:
        """
        Args:
            variables: The variables.
        """  # noqa: D205, D212
        super().__init__(variables)
        self._register_guard(self._rebuild)
        self.__full_lower_bound = array([])
        self.__full_lower_bound.setflags(write=False)
        self.__full_upper_bound = array([])
        self.__full_upper_bound.setflags(write=False)

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the bounds from a pickled or copied state.

        Args:
            state: The state.
        """
        self.__dict__.update(state)
        # NumPy does not preserve the writeable flag across pickling and copying,
        # so the restored full bounds are writeable;
        # reset the guard so that the next access rebuilds and refreezes them.
        self._register_guard(self._rebuild)

    def get_lower_bound(self, name: str) -> ndarray:
        """Return the lower bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The lower bound of the variable (possibly infinite);
            this array is read-only.
        """
        # Hand out a read-only view rather than the frozen array itself:
        # a view does not own its data,
        # so NumPy refuses to re-enable its writeable flag.
        return self._variables[name].lower_bound.view()

    def get_upper_bound(self, name: str) -> ndarray:
        """Return the upper bound of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The upper bound of the variable (possibly infinite);
            this array is read-only.
        """
        return self._variables[name].upper_bound.view()

    def set_lower_bound(
        self, name: str, lower_bound: complex | Iterable[complex]
    ) -> None:
        """Set the lower bound of a variable.

        Args:
            name: The name of the variable.
            lower_bound: The lower bound of the variable.
        """
        variable = self._variables[name]
        self._variables[name] = variable.model_copy(update={"lower_bound": lower_bound})

    def set_upper_bound(
        self, name: str, upper_bound: complex | Iterable[complex]
    ) -> None:
        """Set the upper bound of a variable.

        Args:
            name: The name of the variable.
            upper_bound: The upper bound of the variable.
        """
        variable = self._variables[name]
        self._variables[name] = variable.model_copy(update={"upper_bound": upper_bound})

    def _rebuild(self) -> None:
        """Rebuild the bounds of the full vector."""
        variables = self._variables
        self.__full_lower_bound = concatenate_values(
            {name: variable.lower_bound for name, variable in variables.items()},
            variables,
        )
        self.__full_lower_bound.setflags(write=False)
        self.__full_upper_bound = concatenate_values(
            {name: variable.upper_bound for name, variable in variables.items()},
            variables,
        )
        self.__full_upper_bound.setflags(write=False)

    @property
    def full_lower_bound(self) -> ndarray:
        """The lower bound of the full vector (read-only)."""
        self._refresh()
        # Hand out a read-only view rather than the cache itself,
        # so that no caller can reach the cached array.
        return self.__full_lower_bound.view()

    @property
    def full_upper_bound(self) -> ndarray:
        """The upper bound of the full vector (read-only)."""
        self._refresh()
        return self.__full_upper_bound.view()

    @overload
    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[False] = False,
    ) -> ndarray: ...

    @overload
    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[True] = True,
    ) -> dict[str, ndarray]: ...

    def get_lower_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the lower bounds of variables.

        Args:
            names: The names of the variables.
                If empty, return the lower bounds of all the variables.
            as_dict: Whether to return a dictionary keyed by variable name.
                Otherwise, return an array.

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
    ) -> ndarray: ...

    @overload
    def get_upper_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: Literal[True] = True,
    ) -> dict[str, ndarray]: ...

    def get_upper_bounds(
        self,
        names: Sequence[str] = (),
        as_dict: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the upper bounds of variables.

        Args:
            names: The names of the variables.
                If empty, return the upper bounds of all the variables.
            as_dict: Whether to return a dictionary keyed by variable name.
                Otherwise, return an array.

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
    ) -> ndarray | dict[str, ndarray]:
        """Select the bounds of variables, building only the requested ones.

        Args:
            names: The names of the variables (empty means all).
            as_dict: Whether to return a dictionary keyed by variable name.
            select_lower_bounds: Whether to select the lower bounds.
                Otherwise, select the upper bounds.

        Returns:
            The selected bounds.
        """
        if not names:
            if not as_dict:
                # Fast path returns the cached bound of the full vector.
                if select_lower_bounds:
                    return self.full_lower_bound

                return self.full_upper_bound

            names = self._variables

        get_bound = (
            self.get_lower_bound if select_lower_bounds else self.get_upper_bound
        )
        name_to_bound = {name: get_bound(name) for name in names}
        if as_dict:
            return name_to_bound

        full_bound = concatenate_values(name_to_bound, names)
        # Freeze for consistency with the cached full bounds and the per-variable ones:
        # every bound array handed out is read-only.
        # This array is a throwaway that nothing else reads,
        # so freezing it is enough: no view is needed.
        full_bound.setflags(write=False)
        return full_bound

    def get_active_bounds_masks(
        self,
        name_to_value: Mapping[str, ndarray],
        atol: float = 1e-8,
    ) -> tuple[dict[str, ndarray], dict[str, ndarray]]:
        """Compute the active lower-bound and upper-bound mask of a point.

        Args:
            name_to_value: The point keyed by variable name.
            atol: The absolute tolerance of comparison of a scalar with a bound.

        Returns:
            A map from a variable name to an active lower-bound mask,
            followed by a map from a variable name to an active upper-bound mask.
        """
        active_lower_bound: dict[str, ndarray] = {}
        active_upper_bound: dict[str, ndarray] = {}
        for name in self._variables:
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
        full_value: ndarray,
        normalized: bool = False,
    ) -> ndarray:
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
