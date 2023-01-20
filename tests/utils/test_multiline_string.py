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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_repr
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable


def test_message():
    src = [str(index + 1) for index in range(7)]
    msg = MultiLineString()
    msg.add(src[0])
    msg.indent()
    msg.add(src[1] + "{}", "a")
    msg.indent()
    msg.add(src[2] + "{}" * 2, "a", "b")
    msg.add(src[3])
    msg.dedent()
    msg.add(src[4])
    msg.dedent()
    msg.add(src[5])
    msg.dedent()
    msg.add(src[6])
    expected = ["1", "   2a", "      3ab", "      4", "   5", "6", "7"]
    expected = "\n".join(expected)
    assert str(msg) == expected


def test_message_with_offset():
    with MultiLineString.offset():
        msg = MultiLineString()
        msg.add("foo")
    assert str(msg) == MultiLineString.INDENTATION + "foo"
    msg = MultiLineString()
    msg.add("bar")
    assert str(msg) == "bar"


class A:
    def __repr__(self):
        return "foo"

    def __str__(self):
        return "bar"


@pytest.mark.parametrize(
    "obj,delimiter,key_value_separator,sort,expected",
    [
        (A(), None, None, None, "foo"),
        ({"b": 1, "a": "a"}, None, None, None, "a='a', b=1"),
        ({"b": 1, "a": "a"}, None, None, False, "b=1, a='a'"),
        ({"b": 1, "a": "a"}, "!", None, None, "a='a'!b=1"),
        ({"b": 1, "a": "a"}, None, ":", None, "a:'a', b:1"),
        ([1, "a", 2], None, None, None, "'a', 1, 2"),
        ([1, "a", 2], None, None, False, "1, 'a', 2"),
        ([1, "a", 2], "!", None, None, "'a'!1!2"),
    ],
)
def test_pretty_repr(obj, delimiter, expected, sort, key_value_separator):
    """Check the function pretty_repr."""
    kwargs = {}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    if key_value_separator is not None:
        kwargs["key_value_separator"] = key_value_separator
    if sort is not None:
        kwargs["sort"] = sort
    assert pretty_repr(obj, **kwargs) == expected


@pytest.mark.parametrize(
    "obj,delimiter,key_value_separator,sort,expected",
    [
        (A(), None, None, None, "bar"),
        ({"b": 1, "a": "a"}, None, None, None, "a=a, b=1"),
        ({"b": 1, "a": "a"}, None, None, False, "b=1, a=a"),
        ({"b": 1, "a": "a"}, "!", None, None, "a=a!b=1"),
        ({"b": 1, "a": "a"}, None, ":", None, "a:a, b:1"),
        ([1, "a", 2], None, None, None, "1, 2, a"),
        ([1, "a", 2], None, None, False, "1, a, 2"),
        ([1, "a", 2], "!", None, None, "1!2!a"),
    ],
)
def test_pretty_str(obj, delimiter, key_value_separator, sort, expected):
    """Check the function pretty_str."""
    kwargs = {}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    if key_value_separator is not None:
        kwargs["key_value_separator"] = key_value_separator
    if sort is not None:
        kwargs["sort"] = sort
    assert pretty_str(obj, **kwargs) == expected


def test_replace():
    msg = MultiLineString()
    msg.add("123")
    msg.add("4526")
    expected = "\n".join(("13", "456"))
    repl = msg.replace("2", "")
    assert str(repl) == expected

    msg = MultiLineString()
    msg.add("123")
    repl = msg.replace("5", "9")
    assert "123" == str(repl)


def test_add():
    msg = MultiLineString()
    msg.add("123")
    msg2 = MultiLineString()
    msg2.add("456")

    expected = "\n".join(("123", "456"))
    assert str(msg + msg2) == expected
    assert str(msg + "456") == expected


def test_repr_variable_default_settings():
    """Check repr_variable() with default settings."""
    assert repr_variable("x", 0) == "x[0]"


@pytest.mark.parametrize("size,expected", [(0, "x[0]"), (1, "x"), (2, "x[0]")])
def test_repr_variable_custom_settings(size, expected):
    """Check repr_variable() with custom settings."""
    assert repr_variable("x", 0, size=size) == expected
