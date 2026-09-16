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
"""String rendering for a space of variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.util.repr_html import repr_html_wrapper
from gemseo.util.string import _convert_camel_case_to_lower_case_words

if TYPE_CHECKING:
    from gemseo.space.base import BaseVariableSpace


def render_string(
    space: BaseVariableSpace,
    use_html: bool,
) -> str:
    """Render a space of variables as a string.

    Args:
        space: The space of variables.
        use_html: Whether the output is HTML.

    Returns:
        The string representation of the space of variables.
    """
    title = _convert_camel_case_to_lower_case_words(type(space).__name__).capitalize()
    post_title = ": " if space.name else ":"
    new_line = "<br/>" if use_html else "\n"
    pretty_table = space.get_pretty_table(with_index=True, capitalize=True)
    method = "get_html_string" if use_html else "get_string"
    table = getattr(pretty_table, method)()
    header = f"{title}{post_title}{space.name}{new_line}"
    footer = space._render_footer()
    if footer:
        return f"{header}{table}{new_line}{footer}"

    return f"{header}{table}"


def render_html(space: BaseVariableSpace) -> str:
    """Render a space of variables as embedded HTML (for Jupyter `_repr_html_`).

    Args:
        space: The space of variables.

    Returns:
        The HTML representation of the space of variables.
    """
    return repr_html_wrapper.format(render_string(space, use_html=True))
