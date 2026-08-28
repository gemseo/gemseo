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
"""Value for versioned variables."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy import array
from numpy import logical_or
from numpy import ndarray

from gemseo.optimization.result import OptimizationResult
from gemseo.space.design import _checking
from gemseo.space.design._codec import concatenate_values
from gemseo.space.design._codec import split_full_value
from gemseo.space.design._constants import BOUND_ATOL
from gemseo.space.design._registry_derived_data import RegistryDerivedData
from gemseo.util._numpy import COMPLEX128_DTYPE
from gemseo.util._numpy import FLOAT64_DTYPE
from gemseo.util._numpy import get_common_dtype as _compute_common_dtype
from gemseo.util.read_only_mapping import ReadOnlyMapping
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import dtype

    from gemseo.space.design._bounds import Bounds
    from gemseo.space.design._normalizer import Normalizer
    from gemseo.space.design._variables import Variables


class Value(RegistryDerivedData):
    """Value for versioned variables."""

    __bounds: Bounds
    """The bounds of the variables."""

    __normalizer: Normalizer
    """The normalizer."""

    __name_to_value: dict[str, ndarray]
    """The map from a variable name to a variable value."""

    __name_to_normalized_value: dict[str, ndarray]
    """The map from a variable name to a normalized variable value."""

    __full_value: ndarray
    """The derived full value."""

    __normalized_full_value: ndarray
    """The derived normalized full value."""

    __mutation_count: int
    """A counter bumped on every mutation of the values, for staleness keys."""

    __last_variables_version: int
    """The variables version seen at the last status refresh."""

    __has_value: bool
    """Whether every variable has a current value."""

    __common_dtype: dtype
    """The common dtype of the current values, driving normalization upcasting."""

    __name_to_value_view: ReadOnlyMapping[str, ndarray | None]
    """The read-only view on `__name_to_value`."""

    def __init__(
        self,
        variables: Variables,
        bounds: Bounds,
        normalizer: Normalizer,
    ) -> None:
        """
        Args:
            variables: The variables.
            bounds: The bounds.
            normalizer: The normalizer.
        """  # noqa: D205, D212
        super().__init__(variables)
        self.__bounds = bounds
        self.__normalizer = normalizer
        self.__name_to_value = {}
        self.__has_value = False
        self.__mutation_count = 0
        self.__last_variables_version = variables.version
        self.__common_dtype = FLOAT64_DTYPE
        self._register_guard(self._refresh_status, name="status")
        self._register_guard(self._refresh_common_dtype, name="common_dtype")
        self._register_guard(self._clear_derived, name="derived_arrays")
        self._clear_derived()
        self.__name_to_value_view = ReadOnlyMapping(self.__name_to_value)

    @property
    def name_to_value(self) -> ReadOnlyMapping[str, ndarray | None]:
        """The map from a variable name to a variable value (read-only).

        A variable value is `None` when the variable has no value.

        The stored values are reconciled with the variables on access, so that
        a value invalidated by a resize is never served.
        """
        self.__update_status()
        return self.__name_to_value_view

    def _get_version_key(self) -> object:
        """Return the (registry version, mutation count) key.

        Returns:
            The version key.
        """
        return (self._variables.version, self.__mutation_count)

    def _clear_derived(self) -> None:
        """Reset the derived array forms."""
        self.__full_value = array([])
        self.__name_to_normalized_value = {}
        self.__normalized_full_value = array([])

    def __clear_derived_if_stale(self) -> None:
        """Clear the derived array forms if the variables or values changed."""
        self._refresh("derived_arrays")

    def __update_status(self) -> None:
        """Refresh the `has_value` flag from `__name_to_value` if stale.

        The refresh is skipped when neither the variables (`version`)
        nor the values (`__mutation_count`) changed since the last call.
        """
        self._refresh("status")

    def _refresh_status(self) -> None:
        """Recompute the `has_value` flag from `__name_to_value`."""
        if self.__last_variables_version != self._variables.version:
            self.__last_variables_version = self._variables.version
            # A resized variable keeps its entry, marked as having no value,
            # so that every variable always has an entry in the current value.
            # Only the sizes are reconciled here: an entry without a matching
            # variable belongs to a rename or a removal in progress and is
            # handled by rename and pop.
            for name, value in self.__name_to_value.items():
                variable = self._variables.get(name)
                if (
                    value is not None
                    and variable is not None
                    and value.size != variable.size
                ):
                    self.__name_to_value[name] = None

        self.__has_value = (
            bool(self.__name_to_value)
            and self.__name_to_value.keys() == self._variables.keys()
            and all(
                value is not None and value.size == self._variables[name].size
                for name, value in self.__name_to_value.items()
            )
        )

    def __reconcile_before_write(self) -> None:
        """Reconcile the stored values with the variables before a mutation.

        A pending resize invalidation is applied against the variables version
        that precedes the mutation, so that a stale value left by an earlier
        resize is dropped while the values about to be written are untouched.
        A wrong-size value written afterwards is left in place for the
        membership check to reject.
        """
        self.__update_status()

    def __update_metadata(self) -> None:
        """Refresh `has_value` and clear derived data if a value is set."""
        self.__mutation_count += 1
        self.__update_status()
        if self.__has_value:
            self._clear_derived()

    @property
    def has_value(self) -> bool:
        """Whether every variable has a current value."""
        self.__update_status()
        return self.__has_value

    @property
    def common_dtype(self) -> dtype:
        """The common dtype of the current variable values.

        The dtype is derived and refreshed only when the variables or the
        values changed. Without a complete current value, the default float
        dtype is returned.
        """
        self._refresh("common_dtype")
        return self.__common_dtype

    def _refresh_common_dtype(self) -> None:
        """Recompute the common dtype of the current variable values."""
        self.__update_status()
        self.__common_dtype = (
            _compute_common_dtype(self.__name_to_value.values())
            if self.__has_value
            else FLOAT64_DTYPE
        )

    def set(
        self,
        value: ndarray | Mapping[str, ndarray | None] | OptimizationResult,
    ) -> None:
        """Set the current value.

        Args:
            value: Either the current full value,
                the map from a variable name to a current variable value
                (or `None` when the variable has no value)
                or an optimization result.

        Raises:
            ValueError: If the value has a wrong dimension.
            TypeError: If the value is not a mapping, array, or
                [OptimizationResult]
                [gemseo.optimization.result.OptimizationResult].
        """
        if isinstance(value, Mapping):
            new_name_to_value = {k: v for k, v in value.items() if k in self._variables}
        elif isinstance(value, ndarray):
            if value.size != self._variables.size:
                msg = (
                    "Invalid current_x, "
                    f"dimension mismatch: {self._variables.size} "
                    f"!= {value.size}."
                )
                raise ValueError(msg)

            new_name_to_value = split_full_value(value, self._variables)
        elif isinstance(value, OptimizationResult):
            if value.x_opt.size != self._variables.size:
                msg = (
                    "Invalid x_opt, "
                    f"dimension mismatch: {self._variables.size} "
                    f"!= {value.x_opt.size}."
                )
                raise ValueError(msg)

            new_name_to_value = split_full_value(value.x_opt, self._variables)
        else:
            msg = (
                "The current design value should be either an array, "
                "a dictionary of arrays "
                "or an optimization result; "
                f"got {type(value)} instead."
            )
            raise TypeError(msg)

        self.__reconcile_before_write()
        self.__name_to_value = new_name_to_value
        self.__name_to_value_view = ReadOnlyMapping(self.__name_to_value)
        for name, val in self.__name_to_value.items():
            if val is not None:
                self.__name_to_value[name] = self._variables[name].cast(val)

        self.__update_metadata()

    def set_variable(self, name: str, value: ndarray | None) -> None:
        """Set the current value of a single variable.

        Args:
            name: The name of the variable.
            value: The current value of the variable,
                or `None` when the variable has no value.

        Raises:
            ValueError: If a component of the value falls outside the domain
                of the kind of the variable.
        """
        # Validate via __getitem__ so an unknown name raises before mutating.
        variable = self._variables[name]
        self.__reconcile_before_write()
        if value is not None:
            # Reject before casting: casting to the NumPy type of the variable,
            # e.g. an integer variable, would otherwise silently truncate an
            # out-of-domain value instead of rejecting it.
            _checking.check_domain(self._variables, name, value)
        # A variable with no value keeps its None marker; a genuine value is cast
        # so that the caller does not keep a hand on the stored array.
        self.__name_to_value[name] = None if value is None else variable.cast(value)
        self.__update_metadata()

    def pop(self, name: str) -> None:
        """Remove a variable's current value if present.

        Args:
            name: The name of the variable.
        """
        self.__reconcile_before_write()
        if name in self.__name_to_value:
            del self.__name_to_value[name]
        self.__update_metadata()

    def rename(self, current_name: str, new_name: str) -> None:
        """Rename a variable.

        Args:
            current_name: The original name.
            new_name: The new name.
        """
        self.__reconcile_before_write()
        self.__name_to_value[new_name] = self.__name_to_value.pop(current_name, None)
        self.__update_metadata()

    def to_complex(self) -> None:
        """Cast all current variable values to `COMPLEX128_DTYPE`."""
        self.__reconcile_before_write()
        for name, val in self.__name_to_value.items():
            # A variable with no value keeps its None marker: casting it would
            # yield a zero-dimensional NaN array passing for a genuine value.
            if val is not None:
                self.__name_to_value[name] = array(val, dtype=COMPLEX128_DTYPE)

        self.__mutation_count += 1
        self._clear_derived()

    def initialize_missing(self) -> None:
        """Initialize the current values that are missing.

        Use the center of the bounds when both are finite, otherwise the
        finite bound, otherwise zero.
        """
        # A value invalidated by a resize must read as missing here,
        # otherwise the variable would keep its stale wrong-size value.
        self.__update_status()
        for name, variable in self._variables.items():
            if self.__name_to_value.get(name) is not None:
                continue

            self.set_variable(name, variable.compute_default_value())

    def check_value(self, name: str) -> None:
        """Verify that the current value of a variable is within the bounds.

        Args:
            name: The name of the variable.

        Raises:
            ValueError: If the current value falls outside the bounds.
        """
        lower_bound = self.__bounds.get_lower_bound(name)
        upper_bound = self.__bounds.get_upper_bound(name)
        # A value invalidated by a resize is not checked against the new bounds.
        self.__update_status()
        current_value = self.__name_to_value.get(name)
        if current_value is None:
            return

        indices = logical_or(
            current_value < lower_bound - BOUND_ATOL,
            current_value > upper_bound + BOUND_ATOL,
        ).nonzero()[0]
        for index in indices:
            msg = (
                f"The current value of variable {name!r} "
                f"({current_value[index]}) is "
                f"not between the lower bound {lower_bound[index]} "
                f"and the upper bound {upper_bound[index]}."
            )
            raise ValueError(msg)

    def __is_missing(self, name: str) -> bool:
        """Check whether a variable has no current value or a wrong-size one.

        Args:
            name: The name of the variable.

        Returns:
            Whether the variable is missing its current value.
        """
        value = self.__name_to_value.get(name)
        return value is None or value.size != self._variables[name].size

    def get(
        self,
        names: Sequence[str] | None = None,
        complex_to_real: bool = False,
        as_dict: bool = False,
        normalize: bool = False,
    ) -> ndarray | dict[str, ndarray]:
        """Return the values of variables.

        Args:
            names: The names of the variables.
                If `None`, return the values of all variables.
            complex_to_real: Whether to cast complex numbers to real.
            as_dict: Whether to return a dictionary.
            normalize: Whether to normalize the values into `[0,1]`.

        Returns:
            The values of the variables.
        """
        if names is not None and not names:
            return {} if as_dict else array([])

        return_all = names is None or set(names) == self._variables.keys()

        self.__update_status()
        if not self.__has_value:
            if return_all and as_dict and not normalize:
                return {
                    name: value
                    for name, value in self.__name_to_value.items()
                    if not self.__is_missing(name)
                }

            if return_all or normalize:
                missing = {name for name in self._variables if self.__is_missing(name)}
                msg = (
                    "There is no current value for the design variables: "
                    f"{pretty_str(missing)}."
                )
                if not normalize:
                    raise KeyError(msg)

                msg = (
                    "The current value of a design space cannot be normalized "
                    f"when some variables have no current value. {msg}"
                )
                raise KeyError(msg)

        if normalize:
            self.__compute_normalization_values()

        if (names is None or list(names) == list(self._variables)) and not as_dict:
            return self.__format_full_value(
                self.__normalized_full_value if normalize else self.__get_array(),
                complex_to_real,
            )

        if return_all and as_dict:
            return self.__format_values(
                self.__name_to_normalized_value if normalize else self.__name_to_value,
                complex_to_real,
            )

        unknown = set(names) - set(self._variables)
        if unknown:
            msg = f"There are no such variables named: {pretty_str(unknown)}."
            raise ValueError(msg)

        if not normalize:
            missing = {name for name in names if self.__is_missing(name)}
            if missing:
                msg = (
                    "There is no current value for the design variables: "
                    f"{pretty_str(missing)}."
                )
                raise KeyError(msg)

        source = self.__name_to_normalized_value if normalize else self.__name_to_value
        current_value = {name: source[name] for name in names}

        if as_dict:
            return self.__format_values(current_value, complex_to_real)

        return self.__format_full_value(
            concatenate_values(current_value, names),
            complex_to_real,
        )

    def __get_array(self) -> ndarray:
        """Return the current full value, populating the derived value.

        Returns:
            The current full value.
        """
        self.__clear_derived_if_stale()
        if not len(self.__full_value):
            self.__full_value = concatenate_values(
                self.__name_to_value, self._variables
            )
        return self.__full_value

    def __compute_normalization_values(self) -> None:
        """Populate the derived normalized current-value forms."""
        self.__clear_derived_if_stale()
        if len(self.__normalized_full_value):
            return

        self.__normalized_full_value = self.__normalizer.normalize(
            self.__get_array(),
            self.common_dtype,
        )
        self.__name_to_normalized_value = split_full_value(
            self.__normalized_full_value, self._variables
        )
        for (
            name,
            to_normalize,
        ) in self._variables.name_to_normalization_mask.items():
            if not to_normalize.any():
                # A value left unnormalized keeps the NumPy type of its variable.
                self.__name_to_normalized_value[name] = self._variables[name].cast(
                    self.__name_to_normalized_value[name]
                )

    @staticmethod
    def __format_values(
        name_to_value: dict[str, ndarray], complex_to_real: bool
    ) -> dict[str, ndarray]:
        """Cast variable values to real numbers if requested.

        Args:
            name_to_value: The map from a variable name to a variable value.
            complex_to_real: Whether to cast complex numbers to real.

        Returns:
            The map from a variable name to a formatted variable value.
        """
        if complex_to_real:
            return {name: value.real for name, value in name_to_value.items()}
        return name_to_value

    @staticmethod
    def __format_full_value(value: ndarray, complex_to_real: bool) -> ndarray:
        """Cast a full value to real if requested.

        Args:
            value: The full value.
            complex_to_real: Whether to cast the complex components to real.

        Returns:
            The formatted full value.
        """
        if complex_to_real:
            return value.real
        return value
