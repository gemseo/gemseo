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
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.string_tools import MultiLineString
from gemseo.utils.string_tools import pretty_repr


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


def test_pretty_repr():
    assert pretty_repr({"a": 1, "b": 2}) == "a=1, b=2"
    assert pretty_repr({"a": 1, "b": "a"}) == "a=1, b='a'"
    assert pretty_repr([1, "a", 2]) == "1, a, 2"
    assert pretty_repr([1, "a", 2], delimiter=", ") == "1, a, 2"
    assert (
        pretty_repr(MDODiscipline) == "<class 'gemseo.core.discipline.MDODiscipline'>"
    )


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
