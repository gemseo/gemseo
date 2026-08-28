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
"""Membership and consistency checks for a design space."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Complex
from typing import TYPE_CHECKING
from typing import Any

from numpy import equal
from numpy import isnan
from numpy import ndarray
from numpy import vectorize

from gemseo.space._variable import format_components
from gemseo.space.design._constants import BOUND_ATOL
from gemseo.util.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from numpy import dtype

    from gemseo.space.design._bounds import Bounds
    from gemseo.space.design._variables import Variables


def _is_numeric(value: Any) -> bool:
    """Check that a value is numeric.

    Args:
        value: The value to be checked.

    Returns:
        Whether the value is numeric.
    """
    return value is None or isinstance(value, Complex)


def _is_not_nan(value: ndarray) -> bool:
    """Check that a value is not a nan.

    Args:
        value: The value to be checked.

    Returns:
        Whether the value is not a nan.
    """
    return (value is None) or ~isnan(value)


def check_addable_value(
    variables: Variables,
    value: ndarray,
    name: str,
) -> bool:
    """Check that the value of a variable is valid before adding it.

    Args:
        variables: The variables.
        value: The value to be checked.
        name: The name of the variable.

    Returns:
        Whether the value of the variable is valid.

    Raises:
        ValueError: Either if the array is not one-dimensional,
            if the value is not numerizable,
            if the value is nan
            or if a component falls outside the domain
            of the kind of the variable.
    """
    all_indices = set(range(len(value)))
    # OK if the variable value is one-dimensional
    if value.ndim > 1:
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

    test = vectorize(_is_numeric)(value)
    indices = all_indices - set(test.nonzero()[0])
    if indices:
        plural = len(indices) > 1
        msg = (
            f"The following value{'s' if plural else ''} of variable '{name}' "
            f"{'are' if plural else 'is'} "
            "neither None nor complex and cannot be cast to float: "
            f"{format_components(value, indices)}."
        )
        raise ValueError(msg)

    test = vectorize(_is_not_nan)(value)
    indices = all_indices - set(test.nonzero()[0])
    if indices:
        plural = len(indices) > 1
        msg = (
            f"The following value{'s' if plural else ''} of variable '{name}' "
            f"{'are' if plural else 'is'} neither None nor "
            f"{'numbers' if plural else 'a number'}: "
            f"{format_components(value, indices)}."
        )
        raise ValueError(msg)

    # Check if some components are outside the domain of the kind of the variable.
    variable = variables[name]
    indices = variable.find_components_outside_domain(value)
    if indices:
        plural = len(indices) > 1
        msg = (
            f"The following value{'s' if plural else ''} of variable '{name}' "
            f"{'are' if plural else 'is'} neither None nor {variable.type} "
            f"while variable '{name}' is of type {variable.type}: "
            f"{format_components(value, indices)}."
        )
        raise ValueError(msg)

    return True


def check_out_array(
    out: ndarray,
    dtype_: dtype,
    shape: tuple[int, ...],
) -> None:
    """Check that an array can store a result exactly.

    An array supplied by a caller can neither be converted nor resized,
    so a mismatch is an error rather than something to accommodate.

    Args:
        out: The array to store the result.
        dtype_: The dtype of the result.
        shape: The shape of the result.

    Raises:
        ValueError: When the shape or the dtype of `out` is not that of the result.
    """
    if out.shape != shape:
        msg = f"Expected an out array of shape {shape}; got {out.shape}."
        raise ValueError(msg)

    if out.dtype != dtype_:
        msg = f"Expected an out array of dtype {dtype_}; got {out.dtype}."
        raise ValueError(msg)


def check_membership(
    variables: Variables,
    bounds: Bounds,
    value: Mapping[str, ndarray | None] | ndarray,
    names: Sequence[str] = (),
) -> None:
    """Check whether a value satisfies the bounds and the domains of the kinds.

    Args:
        variables: The variables.
        bounds: The bounds.
        value: Either the full value
            or the map from a variable name to a variable value
            (a variable value of `None` is skipped).
        names: The names of the variables.
            If empty, use all the variables.

    Raises:
        ValueError: If the dimension of the values is wrong,
            the values fall outside the bounds,
            or a component falls outside the domain of the kind of the variable.
        TypeError: If `value` is neither an array nor a mapping.
    """
    if isinstance(value, Mapping):
        _check_membership_dict(variables, value, names)
        return

    if isinstance(value, ndarray):
        if (shape := value.shape)[-1] != (size := variables.size):
            msg = f"Expected an array of shape (..., {size}); got {shape}."
            raise ValueError(msg)

        if names:
            name_to_size = {name: variables[name].size for name in names}
            _check_membership_dict(
                variables,
                split_array_to_dict_of_arrays(value, name_to_size, names),
                names,
            )
        else:
            _check_membership_array(bounds, value)

        return

    msg = (
        "The input vector should be an array or a dictionary; "
        f"got a {type(value)} instead."
    )
    raise TypeError(msg)


def check(variables: Variables, current_value_checker: Callable[[], None]) -> None:
    """Check the consistency of the design space state.

    Args:
        variables: The variables.
        current_value_checker: A zero-argument callable
            that performs the current-value consistency check
            (typically `DesignSpace.__check_current_names`).

    Raises:
        ValueError: If the design space is empty.
    """
    if not variables:
        msg = "The design space is empty."
        raise ValueError(msg)

    current_value_checker()


def _check_membership_array(bounds: Bounds, full_value: ndarray) -> None:
    """Check that the full value stays within the bounds.

    Args:
        bounds: The bounds.
        full_value: The full value.

    Raises:
        ValueError: When the values are outside the bounds up to a tolerance.
    """
    if full_value.ndim > 1:
        for value_i in full_value:
            _check_membership_array(bounds, value_i)
        return

    lower_bound = bounds.full_lower_bound
    upper_bound = bounds.full_upper_bound
    violated_components = (full_value < lower_bound - BOUND_ATOL).nonzero()[0]
    if len(violated_components):
        value_ = full_value[violated_components]
        lower_bound_ = lower_bound[violated_components]
        msg = (
            f"The components {violated_components} of the given array ({value_}) "
            f"are lower than the lower bound ({lower_bound_}) "
            f"by {lower_bound_ - value_}."
        )
        raise ValueError(msg)

    violated_components = (full_value > upper_bound + BOUND_ATOL).nonzero()[0]
    if len(violated_components):
        value_ = full_value[violated_components]
        upper_bound_ = upper_bound[violated_components]
        msg = (
            f"The components {violated_components} of the given array ({value_}) "
            f"are greater than the upper bound ({upper_bound_}) "
            f"by {value_ - upper_bound_}."
        )
        raise ValueError(msg)


def _check_index_in_domain(
    variable: Any,
    name: str,
    index: int,
    value_i: Any,
    out_of_domain_indices: set[int],
) -> None:
    """Check that a component of a value lies within the domain of a variable kind.

    Args:
        variable: The variable.
        name: The name of the variable.
        index: The index of the component.
        value_i: The value of the component.
        out_of_domain_indices: The indices of the components outside the domain
            of the kind of the variable.

    Raises:
        ValueError: If the component falls outside the domain of the kind
            of the variable.
    """
    if index in out_of_domain_indices:
        msg = (
            f"The variable {name} is of type {variable.type}; "
            f"got {name}[{index}] = {value_i}."
        )
        raise ValueError(msg)


def check_domain(variables: Variables, name: str, value: ndarray) -> None:
    """Check that a value lies within the domain of the kind of a variable.

    A value whose size does not match the variable is left unchecked here,
    the size mismatch being handled elsewhere.

    Args:
        variables: The variables.
        name: The name of the variable.
        value: The value of the variable.

    Raises:
        ValueError: If a component falls outside the domain of the kind
            of the variable.
    """
    variable = variables[name]
    if value.size != variable.size:
        return

    out_of_domain_indices = variable.find_components_outside_domain(value.real)
    for i in sorted(out_of_domain_indices):
        _check_index_in_domain(variable, name, i, value[i].real, out_of_domain_indices)


def _check_membership_dict(
    variables: Variables,
    name_to_value: Mapping[str, ndarray | None],
    names: Sequence[str],
) -> None:
    """Check that a per-variable mapping stays within the per-variable bounds.

    Args:
        variables: The variables.
        name_to_value: The map from a variable name to a variable value.
        names: The names of the variables.
            If empty, use all the variables.

    Raises:
        ValueError: If the dimension of an array is wrong,
            the values are outside the bounds,
            or a component falls outside the domain of the kind of the variable.
    """
    names = names or variables
    for name in names:
        variable = variables[name]
        value = name_to_value[name]
        if value is None:
            continue

        if value.size != variable.size:
            msg = (
                f"The variable {name} of size {variable.size} "
                f"cannot be set with an array of size {value.size}."
            )
            raise ValueError(msg)

        out_of_domain_indices = variable.find_components_outside_domain(value.real)
        for i in range(variable.size):
            value_i = value[i].real
            lower_bound = variable.lower_bound[i]
            if value_i < lower_bound - BOUND_ATOL:
                msg = (
                    f"The component {name}[{i}] of the given array ({value_i}) "
                    f"is lower than the lower bound ({lower_bound}) "
                    f"by {lower_bound - value_i:.1e}."
                )
                raise ValueError(msg)

            upper_bound = variable.upper_bound[i]
            if upper_bound + BOUND_ATOL < value_i:
                msg = (
                    f"The component {name}[{i}] of the given array ({value_i}) "
                    f"is greater than the upper bound ({upper_bound}) "
                    f"by {value_i - upper_bound:.1e}."
                )
                raise ValueError(msg)

            _check_index_in_domain(variable, name, i, value_i, out_of_domain_indices)
