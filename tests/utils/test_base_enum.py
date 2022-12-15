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
#        :author:  Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for BaseEnum."""
from __future__ import annotations

import re

import pytest
from gemseo.utils.base_enum import BaseEnum
from gemseo.utils.base_enum import CallableEnum
from gemseo.utils.base_enum import CamelCaseEnum
from gemseo.utils.base_enum import get_names


class MyEnum(BaseEnum):
    ELEM_2 = 1
    ELEM_1 = 0
    ELEM_3 = 2


class MyEnum2(BaseEnum):
    ELEM_1 = 0


def test_str():
    """Check MetaEnum.str."""
    assert str(MyEnum) == "['ELEM_1', 'ELEM_2', 'ELEM_3']"


def test_repr():
    """Check MetaEnum.repr."""
    assert repr(MyEnum) == "MyEnum: ['ELEM_1', 'ELEM_2', 'ELEM_3']"


def test_base_enum1():
    """Test the existence of an Enum member in an Enum."""
    assert MyEnum.ELEM_1 in MyEnum


def test_getitem():
    """Test the __getitem__ class method."""
    assert MyEnum["ELEM_1"] == MyEnum.ELEM_1
    assert MyEnum[MyEnum.ELEM_1] == MyEnum.ELEM_1


def test_get_member_from_name():
    """Test the get_member_from_name class method."""
    assert MyEnum.get_member_from_name("ELEM_1") == MyEnum.ELEM_1
    assert MyEnum.get_member_from_name(MyEnum.ELEM_1) == MyEnum.ELEM_1


def test_get_member_from_name_incorrect_enum():
    """Test that providing an incorrect Enum will raise an Exception."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            "The type of value is ['ELEM_1'] but MyEnum or str are expected."
        ),
    ):
        MyEnum.get_member_from_name(MyEnum2.ELEM_1)


def test_base_enum2():
    """Test the existence of a string in an Enum."""
    assert "ELEM_1" in MyEnum
    assert "ELEM_4" not in MyEnum


def test_base_enum3():
    """Test the equality between an Enum member and a string."""
    assert MyEnum.ELEM_1 == "ELEM_1"


def test_base_enum4():
    """Test the inequality between an Enum member and a string."""
    assert MyEnum.ELEM_2 != "ELEM_1"


def test_base_enum5():
    """Test the equality of an Enum member with itself."""
    assert MyEnum.ELEM_3 == MyEnum.ELEM_3


def test_base_enum6():
    """Test the cast to str of an Enum element."""
    assert str(MyEnum.ELEM_2) == "ELEM_2"


def test_base_enum_not_eq():
    """Test the inequality of an Enum member with a different type."""
    assert MyEnum.ELEM_2 != MyEnum2.ELEM_1


def test_meta_enum_not_in():
    """Test that a member from another Enum is not in an Enum."""
    assert MyEnum.ELEM_2 not in MyEnum2


def test_get_names():
    """Test get_names()."""
    assert get_names(MyEnum) == ["ELEM_1", "ELEM_2", "ELEM_3"]


def test_camel_case_enum():
    """Verify the representation of a camel case Enum."""
    Enum = CamelCaseEnum("Enum", "FOO_BAR FOO")  # noqa: N806
    assert str(Enum.FOO_BAR) == "FooBar"
    assert str(Enum.FOO) == "Foo"


def test_callable_enum():
    """Check that a CallableEnum is callable."""

    def f(a, b=1):
        return a + b

    Function = CallableEnum("Function", {"f": f})  # noqa: N806
    assert Function.f(1) == 2
    assert Function.f(1, 2) == 3
