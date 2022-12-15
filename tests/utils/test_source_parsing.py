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

import pytest
from gemseo.utils.source_parsing import get_default_options_values
from gemseo.utils.source_parsing import get_options_doc
from gemseo.utils.source_parsing import parse_google
from gemseo.utils.source_parsing import parse_rest


def function_with_google_docstring(arg1, arg2):
    """Compute the sum of two elements.

    Args:
        arg1: The first element.
        arg2: The second element.

    Returns:
        The sum of the elements.
    """


class ClassWithGoogleDocstring:
    """A class doing nothing."""

    def __init__(self, arg1=0.0, arg2=1.0):
        """
        Args:
            arg1: The first argument.
            arg2: The second argument.
        """


def test_get_default_options_values():
    """Check the function getting the default values of the __init__'s options."""
    assert get_default_options_values(ClassWithGoogleDocstring) == {
        "arg1": 0.0,
        "arg2": 1.0,
    }


def test_get_options_doc():
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

REST_DOCSTRING = """
:param arg1: A one-line description.
:param arg2: A multi-line
    description.
:param arg3: A description with a first paragraph.

    And a second one.
"""


def test_google():
    """Test that the Google docstrings are correctly parsed."""
    parsed_docstring = parse_google(DOCSTRING)
    assert parsed_docstring == {
        "arg1": "A one-line description: with colon.",
        "arg2": "A multi-line\ndescription.",
        "arg3": "A description with a first paragraph.\n\nAnd a second one.",
        "arg4": "A kwargs.",
    }


def test_rest():
    """Test that the reST docstrings are correctly parsed."""
    assert parse_rest(REST_DOCSTRING) == {
        "arg1": "A one-line description.",
        "arg2": "A multi-line description.",
        "arg3": "A description with a first paragraph.\n\nAnd a second one.",
    }


def test_google_without_parameters_block():
    """Test that the arguments docstring cannot be parsed without 'Args' section."""
    assert not parse_google(DOCSTRING.replace("Args", "Foo"))


def function_with_malformed_docstring(x):
    """Function.

    Foo:
        x: Description.
    """


def test_parsing_with_malformed_docstring():
    """Test an invalid docstring."""
    with pytest.raises(
        ValueError,
        match=(
            "The docstring of the arguments is malformed: "
            "please use Google style docstrings"
        ),
    ):
        get_options_doc(function_with_malformed_docstring)
