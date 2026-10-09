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
from typing import Final

from numpy import equal
from numpy import isnan
from numpy import ndarray
from numpy import vectorize

from gemseo.space._design.constants import bound_atol
from gemseo.space.variable import BaseNumericVariable
from gemseo.space.variable import VariableType
from gemseo.space.variable._formatting import format_components
from gemseo.util.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.space._design.bounds import Bounds
    from gemseo.space._design.variables import DesignVariables
    from gemseo.space.variable import BaseDeterministicVariable
    from gemseo.util.typing import NumberArray

_types_checked_per_variable: Final[frozenset[VariableType]] = frozenset({
    VariableType.CATALOG,
    VariableType.CATEGORICAL,
    VariableType.DISCRETE,
})
"""The types of the variables whose domain the bounds do not describe.

The bounds of a discrete variable are derived from its choices,
those of a catalog variable from its catalog,
and a categorical variable has none,
so the vectorized bound comparison cannot tell a non-candidate value
lying inside the bounds from a candidate one;
a design space holding such a variable asks each variable about its own domain.
"""


def is_numeric(value: Any) -> bool:
    """Check that a value is numeric.

    Args:
        value: The value to be checked.

    Returns:
        Whether the value is numeric.
    """
    return value is None or isinstance(value, Complex)


def is_not_nan(value: NumberArray) -> bool:
    """Check that a value is not a nan.

    Args:
        value: The value to be checked.

    Returns:
        Whether the value is not a nan.
    """
    return (value is None) or ~isnan(value)


def check_addable_value(
    variables: DesignVariables,
    value: NumberArray,
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

    test = vectorize(is_numeric)(value)
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

    test = vectorize(is_not_nan)(value)
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
        # The wording of the failure belongs to the kind of the variable.
        msg = variable._get_out_of_domain_message(name, value, indices)
        raise ValueError(msg)

    return True


def check_membership(
    variables: DesignVariables,
    bounds: Bounds,
    value: Mapping[str, NumberArray | None] | NumberArray,
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
        check_membership_dict(variables, value, names)
        return

    if isinstance(value, ndarray):
        if (shape := value.shape)[-1] != (size := variables.size):
            msg = f"Expected an array of shape (..., {size}); got {shape}."
            raise ValueError(msg)

        if not names and any(
            variable.type in _types_checked_per_variable
            for variable in variables.values()
        ):
            # Fall back to the per-variable path,
            # which asks each variable about its own domain.
            if value.ndim > 1:
                if are_full_values_surely_valid(variables, bounds, value):
                    # The values are all valid:
                    # a failure is reported by the loop below,
                    # which names the variable and the component at fault.
                    return

                for value_i in value:
                    check_membership(variables, bounds, value_i)
                return

            names = list(variables)

        if names:
            name_to_size = {name: variables[name].size for name in names}
            check_membership_dict(
                variables,
                split_array_to_dict_of_arrays(value, name_to_size, names),
                names,
            )
        else:
            check_membership_array(bounds, value)

        return

    msg = (
        "The input vector should be an array or a dictionary; "
        f"got a {type(value)} instead."
    )
    raise TypeError(msg)


def are_full_values_surely_valid(
    variables: DesignVariables, bounds: Bounds, full_values: NumberArray
) -> bool:
    """Check quickly that full values stay within the bounds and the categories.

    This vectorized test only handles design spaces
    whose variables are real, integer or categorical and whose values are real,
    and is conservative:
    when it returns `False`,
    the values may still be valid,
    and the caller must then check them variable by variable.

    Args:
        variables: The variables.
        bounds: The bounds.
        full_values: The full values,
            the components being along the last axis.

    Returns:
        Whether the values are valid for sure.
    """
    if full_values.dtype.kind not in "iuf" or variables.has_variables_of_type(
        VariableType.DISCRETE
    ):
        return False

    # The integer and categorical components must be whole numbers.
    whole_number_indices = [
        index
        for name, variable in variables.items()
        if variable.type in {VariableType.CATEGORICAL, VariableType.INTEGER}
        for index in variables.name_to_indices[name]
    ]
    return bool(
        (full_values[..., whole_number_indices] % 1 == 0).all()
        and (full_values >= bounds.full_lower_bound - bound_atol).all()
        and (full_values <= bounds.full_upper_bound + bound_atol).all()
    )


def check_membership_array(bounds: Bounds, full_value: NumberArray) -> None:
    """Check that the full value stays within the bounds.

    Args:
        bounds: The bounds.
        full_value: The full value.

    Raises:
        ValueError: When the values are outside the bounds up to a tolerance.
    """
    if full_value.ndim > 1:
        for value_i in full_value:
            check_membership_array(bounds, value_i)
        return

    lower_bound = bounds.full_lower_bound
    upper_bound = bounds.full_upper_bound
    violated_components = (full_value < lower_bound - bound_atol).nonzero()[0]
    if len(violated_components):
        value_ = full_value[violated_components]
        lower_bound_ = lower_bound[violated_components]
        msg = (
            f"The components {violated_components} of the given array ({value_}) "
            f"are lower than the lower bound ({lower_bound_}) "
            f"by {lower_bound_ - value_}."
        )
        raise ValueError(msg)

    violated_components = (full_value > upper_bound + bound_atol).nonzero()[0]
    if len(violated_components):
        value_ = full_value[violated_components]
        upper_bound_ = upper_bound[violated_components]
        msg = (
            f"The components {violated_components} of the given array ({value_}) "
            f"are greater than the upper bound ({upper_bound_}) "
            f"by {value_ - upper_bound_}."
        )
        raise ValueError(msg)


def check_index_in_domain(
    variable: BaseDeterministicVariable,
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
        # The wording of the failure belongs to the kind of the variable.
        msg = variable._get_out_of_domain_component_message(name, index, value_i)
        raise ValueError(msg)


def check_domain(variables: DesignVariables, name: str, value: NumberArray) -> None:
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

    # Take the real part of the array, not of a component:
    # a component of an object array can be a non-numeric value,
    # e.g. a label, which has no real part.
    real_value = value.real
    out_of_domain_indices = variable.find_components_outside_domain(real_value)
    for i in sorted(out_of_domain_indices):
        check_index_in_domain(variable, name, i, real_value[i], out_of_domain_indices)


def check_membership_dict(
    variables: DesignVariables,
    name_to_value: Mapping[str, NumberArray | None],
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
        is_numeric_variable = isinstance(variable, BaseNumericVariable)
        for i in range(variable.size):
            value_i = value[i].real
            if not is_numeric_variable:
                # A variable without bounds is checked against its domain only.
                check_index_in_domain(variable, name, i, value_i, out_of_domain_indices)
                continue

            lower_bound = variable.lower_bound[i]
            if value_i < lower_bound - bound_atol:
                msg = (
                    f"The component {name}[{i}] of the given array ({value_i}) "
                    f"is lower than the lower bound ({lower_bound}) "
                    f"by {lower_bound - value_i:.1e}."
                )
                raise ValueError(msg)

            upper_bound = variable.upper_bound[i]
            if upper_bound + bound_atol < value_i:
                msg = (
                    f"The component {name}[{i}] of the given array ({value_i}) "
                    f"is greater than the upper bound ({upper_bound}) "
                    f"by {value_i - upper_bound:.1e}."
                )
                raise ValueError(msg)

            check_index_in_domain(variable, name, i, value_i, out_of_domain_indices)
