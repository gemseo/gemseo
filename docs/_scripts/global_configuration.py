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
"""Insert the list of the global configuration options in its user guide page.

The list is built at build time from the fields of
[GlobalConfiguration][gemseo.util.global_configuration.GlobalConfiguration],
so that it cannot drift from the code.
It replaces the marker [_marker][] of the page [_page_path][].

The defaults are read from the fields, not from `gemseo.configuration`:
the latter is built from the environment variables and the `.env` file
of the machine building the documentation.
"""

from __future__ import annotations

from enum import Enum
from logging import getLevelName
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

import mkdocs_gen_files
from pydantic import BaseModel

from gemseo.util.global_configuration import GlobalConfiguration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic.fields import FieldInfo

_page_path: Final[str] = "user_guide/user_experience/global_configuration.md"
"""The path of the page, relative to the documentation directory."""

_marker: Final[str] = "<!-- global-configuration-options -->"
"""The marker of the page replaced by the list."""

_nested_field_names: Final[tuple[str, ...]] = ("logging", "directory_manager")
"""The names of the fields whose own fields are listed instead of them."""

_env_prefix: Final[str] = GlobalConfiguration.model_config["env_prefix"]
"""The prefix of the environment variables."""


def _iter_rows() -> Iterator[tuple[str, FieldInfo]]:
    """Iterate over the options of the global configuration.

    Yields:
        The dotted name of an option and its field.
    """
    for name, field in GlobalConfiguration.model_fields.items():
        if name in _nested_field_names:
            for sub_name, sub_field in field.annotation.model_fields.items():
                yield f"{name}.{sub_name}", sub_field
        else:
            yield name, field


def _format_default(value: Any) -> str:
    """Format a default value which is not a Pydantic model.

    Args:
        value: The default value.

    Returns:
        The default value formatted as Markdown.
    """
    if isinstance(value, Enum):
        value = value.value
    elif isinstance(value, PurePath):
        value = value.as_posix()

    if isinstance(value, str):
        return f'`"{value}"`'

    return f"`{value}`"


def _format_description(description: str | None) -> str:
    """Format the description of an option.

    Args:
        description: The description of the field.

    Returns:
        The first paragraph of the description on a single line.
    """
    first_paragraph = (description or "").strip().split("\n\n")[0]
    return " ".join(first_paragraph.split())


def _build_list() -> str:
    """Build the list of the options.

    Returns:
        The list formatted as Markdown.
    """
    lines = []
    for name, field in _iter_rows():
        env_var_name = _env_prefix + name.replace(".", "_").upper()
        default = field.default
        if name == "logging.level":
            # Show the level by its name, the form an environment variable accepts.
            default = getLevelName(default)

        lines.extend((
            f"- `{name}`: {_format_description(field.description)}",
            f"    - Environment variable: `{env_var_name}`",
        ))
        if isinstance(default, BaseModel):
            lines.append("    - Default:")
            lines.extend(
                f"        - `{sub_name}`: {_format_default(getattr(default, sub_name))}"
                for sub_name in type(default).model_fields
            )
        else:
            lines.append(f"    - Default: {_format_default(default)}")

    return "\n".join(lines)


with mkdocs_gen_files.open(_page_path, "r") as page_file:
    _content = page_file.read()

if _marker not in _content:
    msg = f"The page {_page_path} does not contain the marker {_marker}."
    raise ValueError(msg)

with mkdocs_gen_files.open(_page_path, "w") as page_file:
    page_file.write(_content.replace(_marker, _build_list()))

mkdocs_gen_files.set_edit_path(_page_path, _page_path)
