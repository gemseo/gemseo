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
from __future__ import annotations

import logging

import pytest

from gemseo.utils.source_parsing import get_callable_argument_defaults
from gemseo.utils.source_parsing import get_options_doc
from gemseo.utils.source_parsing import get_return_description
from gemseo.utils.source_parsing import parse_google


def function_with_google_docstring(arg1, arg2) -> None:
    """Compute the sum of two elements.

    Args:
        arg1: The first element.
        arg2: The second element.

    Returns:
        The sum of the elements.
    """


class ClassWithGoogleDocstring:
    """A class doing nothing."""

    def __init__(self, arg1=0.0, arg2=1.0) -> None:
        """Args:
        arg1: The first argument.
        arg2: The second argument.
        """


def test_get_default_option_values() -> None:
    """Check the function getting the default values of the __init__'s options."""
    assert get_callable_argument_defaults(ClassWithGoogleDocstring.__init__) == {
        "arg1": 0.0,
        "arg2": 1.0,
    }


def test_get_options_doc() -> None:
    """Check the function getting the documentation of the options of a function."""
    assert get_options_doc(function_with_google_docstring) == {
        "arg1": "The first element.",
        "arg2": "The second element.",
    }


DOCSTRING = """
Args:
    arg1: A one-line description: with colon.
    arg2: A multi-line
        description.
    *arg3: A description with a first paragraph.

        And a second one.
    **arg4: A kwargs.

Section title:
    Section description.
"""


def test_google() -> None:
    """Test that the Google docstrings are correctly parsed."""
    parsed_docstring = parse_google(DOCSTRING)
    assert parsed_docstring == {
        "arg1": "A one-line description: with colon.",
        "arg2": "A multi-line\ndescription.",
        "arg3": "A description with a first paragraph.\n\nAnd a second one.",
        "arg4": "A kwargs.",
    }


def test_parsing_function_without_args_section(caplog) -> None:
    """Test parsing a function without Args section."""

    def function() -> None:
        """Function without and without Args section."""

    assert get_options_doc(function) == {}
    assert not caplog.record_tuples

    def function(x) -> None:
        """Function with an argument and without Args section."""

    assert get_options_doc(function) == {}
    _, level, message = caplog.record_tuples[0]
    assert level == logging.WARNING
    assert message == "The Args section is missing."


def test_no_docstring():
    """Test parsing a function without docstring."""

    def foo() -> None: ...

    with pytest.raises(
        ValueError,
        match=r"Empty doc for <function test_no_docstring.<locals>.foo at .*\.",
    ):
        get_options_doc(foo)


def test_return_docstring():
    """Get the docstring of the return object."""
    docstring = get_return_description(function_with_google_docstring)
    assert docstring == "The sum of the elements."

    def foo() -> None: ...

    docstring = get_return_description(foo)
    assert docstring == ""
