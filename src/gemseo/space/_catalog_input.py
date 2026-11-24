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
"""Helpers for the discipline inputs fed by catalog variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space.variable.catalog import CatalogVariable
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Container
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.space.design import DesignSpace
    from gemseo.space.variable.base import BaseVariable


def get_fed_input_names(
    variable_name: str, variable: BaseVariable, input_names: Container[str]
) -> tuple[str, ...]:
    """Return the names of the discipline inputs fed by a design variable.

    A variable feeds the input of the same name.
    A catalog variable also feeds the inputs named after its numerical properties.

    Args:
        variable_name: The name of the design variable.
        variable: The design variable.
        input_names: The names of the discipline inputs.

    Returns:
        The names of the fed inputs, each once:
        the variable name first, then the properties in the catalog order.
    """
    names = [variable_name] if variable_name in input_names else []
    if isinstance(variable, CatalogVariable):
        # TODO: performance: use set operations if this loop is too slow;
        # they must keep the catalog order.
        names.extend(
            name
            for name in variable.catalog.numerical_property_names
            if name in input_names and name != variable_name
        )

    return tuple(names)


def get_input_to_variable_names(
    variables: Mapping[str, BaseVariable], input_names: Container[str]
) -> dict[str, str]:
    """Return the name of the design variable feeding each discipline input.

    Args:
        variables: The design variables, by name.
        input_names: The names of the discipline inputs.

    Returns:
        The name of the variable feeding each fed input, by input name,
        in the order of the variables, then of the inputs they feed.

    Raises:
        ValueError: When an input is fed by several variables.
    """
    input_to_variable_names: dict[str, list[str]] = {}
    for name, variable in variables.items():
        for input_name in get_fed_input_names(name, variable, input_names):
            input_to_variable_names.setdefault(input_name, []).append(name)

    conflicts = [
        f"the input '{input_name}' is fed by the variables "
        f"{pretty_repr(names, sort=False)}"
        for input_name, names in input_to_variable_names.items()
        if len(names) > 1
    ]
    if conflicts:
        msg = (
            "A discipline input must be fed by a single design variable; "
            f"{'; '.join(conflicts)}."
        )
        raise ValueError(msg)

    return {
        input_name: names[0] for input_name, names in input_to_variable_names.items()
    }


def get_feeding_variable_names(
    variables: Mapping[str, BaseVariable], input_names: Container[str]
) -> tuple[str, ...]:
    """Return the names of the design variables that feed discipline inputs.

    Args:
        variables: The design variables, by name.
        input_names: The names of the discipline inputs.

    Returns:
        The names of the variables feeding at least one input.

    Raises:
        ValueError: When an input is fed by several variables.
    """
    feeding_names = set(get_input_to_variable_names(variables, input_names).values())
    return tuple(name for name in variables if name in feeding_names)


def get_catalog_input_values(
    space: DesignSpace, input_names: Container[str]
) -> dict[str, ndarray]:
    """Return the values of the catalog properties that are discipline inputs.

    Args:
        space: The design space.
        input_names: The names of the discipline inputs.

    Returns:
        The 1D array of values of each property that is an input, by property name,
        at the current alternative of the catalog variables.
        Empty if a variable has no current value,
        or if no numerical property of a catalog variable is an input.

    Raises:
        ValueError: When an input is fed by several variables.
    """
    input_to_variable_names = get_input_to_variable_names(space.variables, input_names)
    if not space.has_current_value:
        return {}

    values: dict[str, ndarray] = {}
    positions = space.get_current_value(as_dict=True)
    for input_name, name in input_to_variable_names.items():
        # An input named after its variable is fed by the position of the variable;
        # the other inputs are fed by the properties of a catalog variable.
        if input_name == name:
            continue

        catalog = space.variables[name].catalog  # type: ignore[attr-defined]
        # The space may be cast to complex numbers, e.g. for the complex step.
        indices = positions[name].real.astype(int)
        values[input_name] = catalog.properties[input_name][indices]

    return values
