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
import re
from inspect import getfullargspec
from typing import Any
from typing import Callable


def get_options_doc(
    method: Callable,
) -> dict[str, str]:
    """Get the documentation of a method.

    Args:
        method: The method to retrieve the doc from.

    Returns:
        The descriptions of the options.
    """
    # the docstring has all the leading and common white spaces removed
    docstring = inspect.getdoc(method)

    if docstring is None:
        raise ValueError(f"Empty doc for {method}")

    for parse in (parse_google, parse_rest):
        parsed_docstring = parse(docstring)
        if parsed_docstring:
            return parsed_docstring

    raise ValueError(
        "The docstring of the arguments is malformed: "
        "please use Google style docstrings"
    )


# TODO: API: remove.
def get_default_option_values(
    cls: type,
) -> dict[str, str]:
    """Return the default values of the kwargs of a class constructor.

    Args:
        cls: The class.

    Returns:
        The defaults.
    """
    return get_callable_argument_defaults(cls.__init__)


def get_callable_argument_defaults(
    callable_: Callable,
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


def parse_rest(
    docstring: str,
) -> dict[str, str]:
    """Parse a reST docstring.

    Args:
        docstring: The docstring to be parsed.

    Returns:
        The parsed docstring.
    """
    pattern = r":param ([\*\w]+): (.*?)(?:(?=:param)|(?=:return)|\Z)"
    param_re = re.compile(pattern, re.S)
    strings = param_re.findall(docstring)
    parsed_doc = {txt[0]: txt[1].replace(" " * 4, "").rstrip("\n") for txt in strings}
    return {
        name: description.replace("  ", " ").replace("\n", " ").replace("  ", "\n\n")
        for name, description in parsed_doc.items()
    }


# regex pattern for finding the arguments section of a Google docstring
# docstring-inheritance replaces the section title "Args" with "Parameters"
RE_PATTERN_ARGS_SECTION = re.compile(
    r"(?:Args|Parameters)\s*:\s*\n(.*?)(?:\n\n\S|$)", flags=re.DOTALL
)

# regex pattern for finding the arguments names and description of a Google docstring
RE_PATTERN_ARGS = re.compile(
    r"\**(\w+)\s*:\s*(.*?)(?:$|(?=\n\**\w+\s*:))", flags=re.DOTALL
)


def parse_google(docstring: str) -> dict[str, str]:
    """Parse a Google docstring.

    Args:
        docstring: The docstring to be parsed.

    Returns:
        The parsed docstring with the function arguments names bound to their
        descriptions.
    """
    args_sections = RE_PATTERN_ARGS_SECTION.findall(docstring)

    if len(args_sections) != 1:
        # This is not a Google docstring
        return {}

    # remove leading common blank spaces
    args_section = inspect.cleandoc(args_sections[0])

    parsed_doc = {}

    for name, desc in RE_PATTERN_ARGS.findall(args_section):
        # remove multiple blank spaces
        parsed_doc[name] = re.sub(
            r"\n ", "\n", re.sub(r"[\r\t\f\v ]+", " ", desc).strip()
        )

    return parsed_doc
