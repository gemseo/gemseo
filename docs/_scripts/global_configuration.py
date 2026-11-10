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
"""Write the list of the global configuration options of its user guide page.

The list is built before the build (see ``prebuild.py``) from the fields of
[GlobalConfiguration][gemseo.util.global_configuration.GlobalConfiguration],
so that it cannot drift from the code.
It is written to [_output_path][],
which the page includes with a snippet.

The defaults are read from the fields, not from `gemseo.configuration`:
the latter is built from the environment variables and the `.env` file
of the machine building the documentation.
"""

from __future__ import annotations

from enum import Enum
from logging import getLevelName
from pathlib import Path
from pathlib import PurePath
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from pydantic import BaseModel

from gemseo.util.global_configuration import GlobalConfiguration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from pydantic.fields import FieldInfo

_output_path: Final[Path] = Path("_docs") / "global_configuration.md"
"""The path of the list, relative to the root of the repository."""

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


_output_path.parent.mkdir(exist_ok=True)
_output_path.write_text(_build_list() + "\n", encoding="utf-8")
