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
"""Pretty-table and string rendering for a design space."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Final

from prettytable import PrettyTable

from gemseo.space.design._constants import _TABLE_NAMES
from gemseo.util.repr_html import REPR_HTML_WRAPPER
from gemseo.util.string import _format_value_in_pretty_table_16

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.space.design import DesignSpace


CAMEL_CASE_REGEX: Final[re.Pattern] = re.compile(r"[A-Z][^A-Z]*")


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
        fields = _TABLE_NAMES

    if capitalize:
        field_names = [field.capitalize().replace("_", " ") for field in fields]
    else:
        field_names = list(fields)

    table = PrettyTable(field_names)
    table.custom_format = _format_value_in_pretty_table_16
    for name, variable in design_space._variables.items():
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


def render_string(
    design_space: DesignSpace,
    use_html: bool,
    title: str = "",
    simplify: bool = False,
) -> str:
    """Render a design space as a string.

    Args:
        design_space: The design space.
        use_html: Whether the output is HTML.
        title: The title of the table. If empty, use the class name.
        simplify: Whether to return a simplified representation.
            Only honored by subclasses whose `get_pretty_table` supports it.

    Returns:
        The string representation of the design space.
    """
    if not title:
        title = " ".join(
            CAMEL_CASE_REGEX.findall(design_space.__class__.__name__)
        ).lower()
    title = title.capitalize()
    post_title = ": " if design_space.name else ":"
    new_line = "<br/>" if use_html else "\n"
    pretty_table = design_space.get_pretty_table(
        with_index=True, capitalize=True, simplify=simplify
    )
    method = "get_html_string" if use_html else "get_string"
    table = getattr(pretty_table, method)()
    return f"{title}{post_title}{design_space.name}{new_line}{table}"


def render_html(design_space: DesignSpace) -> str:
    """Render a design space as embedded HTML (for Jupyter `_repr_html_`).

    Args:
        design_space: The design space.

    Returns:
        The HTML representation of the design space.
    """
    return REPR_HTML_WRAPPER.format(render_string(design_space, use_html=True))
