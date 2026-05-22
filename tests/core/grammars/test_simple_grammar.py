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
from __future__ import annotations

import pytest

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.simple import SimpleGrammar
from gemseo.core.grammars.simpler import SimplerGrammar
from gemseo.utils.testing.helpers import assert_exception


@pytest.fixture(params=(SimpleGrammar, SimplerGrammar))
def grammar_class(request):
    """Iterate over the simple grammars."""
    return request.param


@pytest.mark.parametrize("required_names", [[], ["name"]])
def test_init(grammar_class, required_names) -> None:
    """Verify init with non-empty inputs."""
    grammar = grammar_class(
        "g", name_to_type={"name": str}, required_names=required_names
    )
    assert grammar.keys() == {"name"}
    assert list(grammar.values()) == [str]
    assert set(grammar.required_names) == set(required_names)
    assert not grammar.defaults


def test_init_errors(grammar_class, snapshot) -> None:
    """Verify init errors."""
    with assert_exception(TypeError, snapshot):
        grammar_class("g", name_to_type={"name": 0})

    with assert_exception(KeyError, snapshot):
        grammar_class("g", name_to_type={"name": str}, required_names=["foo"])


def test_getitem(grammar_class) -> None:
    """Verify getitem."""
    grammar = grammar_class("g", name_to_type={"name": str})
    assert grammar["name"] is str


def test_update_error(grammar_class, snapshot) -> None:
    """Verify update error."""
    grammar = grammar_class("g1")

    with assert_exception(TypeError, snapshot):
        grammar.update_from_types({"name": 0})


@pytest.mark.parametrize(
    ("name_to_type", "data"),
    [
        # None values element means any type.
        ({"name": None}, {"name": {}}),
    ],
)
def test_validate(grammar_class, name_to_type, data) -> None:
    """Verify validate."""
    grammar = grammar_class("g", name_to_type=name_to_type)
    grammar.validate(data)


@pytest.mark.parametrize(
    ("data", "error_msg"),
    [
        (
            {"name1": 0, "name2": ""},
            r"Bad type for name2: <class 'str'> instead of <class 'int'>.",
        ),
    ],
)
@pytest.mark.parametrize("raise_exception", [True, False])
def test_validate_error(data, error_msg, raise_exception, caplog, snapshot) -> None:
    """Verify that validate raises the expected errors."""
    grammar = SimpleGrammar(
        "g", name_to_type={"name1": None, "name2": int}, required_names=["name1"]
    )

    if raise_exception:
        with assert_exception(InvalidDataError, snapshot):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


def test_update_with_merge_error(grammar_class, snapshot):
    """Verify that any update method raises when merging."""
    grammar = grammar_class("g")

    for method_name in (
        "update",
        "update_from_names",
        "update_from_types",
        "update_from_data",
    ):
        with assert_exception(ValueError, snapshot):
            getattr(grammar, method_name)({"name": bool}, merge=True)
