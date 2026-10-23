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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Parse source code to extract information."""

from __future__ import annotations

import inspect
import logging
import re
from inspect import getfullargspec
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def get_options_doc(
    function: Callable[..., Any],
) -> dict[str, str]:
    """Get the documentation of a function.

    Args:
        function: The function to retrieve the doc from.

    Returns:
        The descriptions of the options.
    """
    # the docstring has all the leading and common white spaces removed
    docstring = inspect.getdoc(function)

    if docstring is None:
        msg = f"Empty doc for {function}."
        raise ValueError(msg)

    return parse_google(docstring, len(getfullargspec(function)[0]))


def get_callable_argument_defaults(
    callable_: Callable[..., Any],
) -> dict[str, Any]:
    """Return the default values of the kwargs of a callable.

    Args:
        callable_: The callable.

    Returns:
        The defaults if any, an empty dictionary otherwise.
    """
    full_arg_specs = getfullargspec(callable_)
    defaults = full_arg_specs.defaults
    if defaults is None:
        return {}
    args = full_arg_specs.args
    if "self" in args:
        args.remove("self")
    n_defaults = len(defaults)
    return {args[-n_defaults:][i]: defaults[i] for i in range(n_defaults)}


# regex pattern for finding the arguments section of a Google docstring
# docstring-inheritance replaces the section title "Args" with "Parameters"
re_pattern_args_section: Final[re.Pattern[str]] = re.compile(
    r"(?:Args)\s*:\s*\n(.*?)(?:\n\n\S|$)", flags=re.DOTALL
)

# regex pattern for finding the arguments names and description of a Google docstring
re_pattern_args: Final[re.Pattern[str]] = re.compile(
    r"\**(\w+)\s*:\s*(.*?)(?:$|(?=\n\**\w+\s*:))", flags=re.DOTALL
)

re_pattern_return: Final[re.Pattern[str]] = re.compile(r"Returns:\s*(.*)", re.DOTALL)


def get_return_description(f: Callable[[Any], Any]) -> str | list[str]:
    """Return the description of the returned object in the Returns block.

    Args:
        f: A callable.

    Returns:
        The description.
    """
    try:
        return re_pattern_return.findall(f.__doc__)[0].strip()
    except TypeError:
        return ""


def parse_google(docstring: str, n_arguments: int = 0) -> dict[str, str]:
    """Parse a Google docstring.

    Args:
        docstring: The docstring to be parsed.
        n_arguments: The number of arguments of the function.

    Returns:
        The parsed docstring with the function arguments names bound to their
        descriptions.
    """
    args_sections = re_pattern_args_section.findall(docstring)

    if len(args_sections) != 1:
        if n_arguments:
            logger.warning("The Args section is missing.")
        return {}

    # remove leading common blank spaces
    args_section = inspect.cleandoc(args_sections[0])

    parsed_doc = {}

    for name, desc in re_pattern_args.findall(args_section):
        # remove multiple blank spaces
        parsed_doc[name] = re.sub(
            r"\n ", "\n", re.sub(r"[\r\t\f\v ]+", " ", desc).strip()
        )

    return parsed_doc
