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
"""Pretty-table rendering for a design space."""

from __future__ import annotations

from typing import TYPE_CHECKING

from prettytable import PrettyTable

from gemseo.space._design.constants import table_names
from gemseo.util.string import _format_value_in_pretty_table_16

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.space.design import DesignSpace


def get_pretty_table(
    design_space: DesignSpace,
    fields: Sequence[str] = (),
    with_index: bool = False,
    capitalize: bool = False,
) -> PrettyTable:
    """Build a tabular view of a design space.

    Args:
        design_space: The design space.
        fields: The name of the fields to be exported.
            If empty, export all the fields.
        with_index: Whether to show indices of components for arrays.
        capitalize: Whether to capitalize the field names
            and replace `"_"` by `" "`.

    Returns:
        The tabular view of the design space.
    """
    if not fields:
        fields = table_names

    if capitalize:
        field_names = [field.capitalize().replace("_", " ") for field in fields]
    else:
        field_names = list(fields)

    table = PrettyTable(field_names)
    table.custom_format = _format_value_in_pretty_table_16
    for name, variable in design_space.variables.items():
        value = design_space._current_value.get(name)
        name_template = f"{name}"
        if with_index and variable.size > 1:
            name_template += "[{index}]"
        for i in range(variable.size):
            # Strip the imaginary part of a complex-step perturbation.
            value_i = None if value is None else value[i].real

            data = {
                "name": name_template.format(name=name, index=i),
                "value": value_i,
                "lower_bound": variable.lower_bound[i],
                "upper_bound": variable.upper_bound[i],
                "type": variable.type,
            }

            table.add_row([data[key] for key in fields])

    for name in ("Name", "Type") if capitalize else ("name", "type"):
        table.align[name] = "l"
    return table
