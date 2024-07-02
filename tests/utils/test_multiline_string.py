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

from gemseo.utils.repr_html import REPR_HTML_WRAPPER
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import filter_names
from gemseo.utils.string_tools import get_name_and_component
from gemseo.utils.string_tools import get_variables_with_components
from gemseo.utils.string_tools import pretty_repr
from gemseo.utils.string_tools import pretty_str
from gemseo.utils.string_tools import repr_variable


def test_message() -> None:
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


def test_message_with_offset() -> None:
    with MultiLineString.offset():
        msg = MultiLineString()
        msg.add("foo")
    assert str(msg) == MultiLineString.INDENTATION + "foo"
    msg = MultiLineString()
    msg.add("bar")
    assert str(msg) == "bar"


class A:
    def __repr__(self) -> str:
        return "foo"

    def __str__(self) -> str:
        return "bar"


@pytest.mark.parametrize(
    ("obj", "delimiter", "key_value_separator", "sort", "expected"),
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
def test_pretty_repr(obj, delimiter, expected, sort, key_value_separator) -> None:
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
    ("obj", "delimiter", "key_value_separator", "sort", "expected"),
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
def test_pretty_str(obj, delimiter, key_value_separator, sort, expected) -> None:
    """Check the function pretty_str."""
    kwargs = {}
    if delimiter is not None:
        kwargs["delimiter"] = delimiter
    if key_value_separator is not None:
        kwargs["key_value_separator"] = key_value_separator
    if sort is not None:
        kwargs["sort"] = sort
    assert pretty_str(obj, **kwargs) == expected


def test_use_and() -> None:
    """Check the option use_and of pretty_repr and pretty_str."""
    assert pretty_str(["b", "c", "a"]) == "a, b, c"
    assert pretty_str(["a", "c", "b"], use_and=True) == "a, b and c"
    assert pretty_repr(["b", "c", "a"]) == "'a', 'b', 'c'"
    assert pretty_repr(["a", "c", "b"], use_and=True) == "'a', 'b' and 'c'"


def test_replace() -> None:
    msg = MultiLineString()
    msg.add("123")
    msg.add("4526")
    expected = "13\n456"
    repl = msg.replace("2", "")
    assert str(repl) == expected

    msg = MultiLineString()
    msg.add("123")
    repl = msg.replace("5", "9")
    assert str(repl) == "123"


def test_add() -> None:
    msg = MultiLineString()
    msg.add("123")
    msg2 = MultiLineString()
    msg2.add("456")

    expected = "123\n456"
    assert str(msg + msg2) == expected
    assert str(msg + "456") == expected


def test_repr_variable_default_settings() -> None:
    """Check repr_variable() with default settings."""
    assert repr_variable("x", 0) == "x[0]"


@pytest.mark.parametrize(("size", "expected"), [(0, "x[0]"), (1, "x"), (2, "x[0]")])
def test_repr_variable_custom_settings(size, expected) -> None:
    """Check repr_variable() with custom settings."""
    assert repr_variable("x", 0, size=size) == expected


@pytest.mark.parametrize(("index", "expected"), [(0, "x[0]"), (1, "[1]")])
def test_repr_variable_simplify(index, expected) -> None:
    """Check repr_variable() with argument simplify."""
    assert repr_variable("x", index, simplify=True) == expected


def test_repr_html() -> None:
    """Check MultiLineString._repr_html_."""
    mls = MultiLineString()
    mls.add("a")
    mls.add("b")
    mls.indent()
    mls.add("c")
    mls.add("d")
    mls.indent()
    mls.add("e")
    mls.dedent()
    mls.add("f")
    mls.indent()
    mls.add("h")
    assert mls._repr_html_() == REPR_HTML_WRAPPER.format(
        "a<br/>"
        "b<br/>"
        "<ul>"
        "<li>c</li>"
        "<li>d"
        "<ul>"
        "<li>e</li>"
        "</ul>"
        "</li>"
        "<li>f"
        "<ul>"
        "<li>h</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


@pytest.mark.parametrize(
    ("variable", "expected"),
    [("foo", ("foo", 0)), (("foo", 0), ("foo", 0)), (("foo", 1), ("foo", 1))],
)
def test_get_name_and_component(variable, expected):
    """Check get_name_and_component()."""
    assert get_name_and_component(variable) == expected


@pytest.mark.parametrize(
    ("names_to_keep", "expected"),
    [((), ["a", "b", "c"]), (("a", "b"), ["a", "b"]), (("b", "a"), ["a", "b"])],
)
def test_filter_names(names_to_keep, expected):
    """Check filter_names()."""
    assert filter_names(["a", "b", "c"], names_to_keep=names_to_keep) == expected


@pytest.mark.parametrize(
    ("variables", "expected"),
    [
        ("x", [("x", 0), ("x", 1)]),
        (("x", 0), [("x", 0)]),
        ((("x", 0), "y"), [("x", 0), ("y", 0), ("y", 1), ("y", 2)]),
        (("y", ("x", 0)), [("y", 0), ("y", 1), ("y", 2), ("x", 0)]),
        ((("y", 1), ("x", 0)), [("y", 1), ("x", 0)]),
    ],
)
def test_rewrite_variables_with_components(variables, expected):
    """Check rewrite_variables_with_components()."""
    assert (
        list(get_variables_with_components(variables, {"x": 2, "a": 1, "y": 3}))
        == expected
    )
