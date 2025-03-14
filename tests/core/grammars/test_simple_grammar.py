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
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.core.grammars.simpler_grammar import SimplerGrammar


@pytest.fixture(params=(SimpleGrammar, SimplerGrammar))
def grammar_class(request):
    """Iterate over the simple grammars."""
    return request.param


@pytest.mark.parametrize("required_names", [[], ["name"]])
def test_init(grammar_class, required_names) -> None:
    """Verify init with non-empty inputs."""
    grammar = grammar_class(
        "g", names_to_types={"name": str}, required_names=required_names
    )
    assert grammar.keys() == {"name"}
    assert list(grammar.values()) == [str]
    assert set(grammar.required_names) == set(required_names)
    assert not grammar.defaults


def test_init_errors(grammar_class) -> None:
    """Verify init errors."""
    msg = "The element name must be a type or None: it is 0."
    with pytest.raises(TypeError, match=msg):
        grammar_class("g", names_to_types={"name": 0})

    msg = "The name 'foo' is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        grammar_class("g", names_to_types={"name": str}, required_names=["foo"])


def test_getitem(grammar_class) -> None:
    """Verify getitem."""
    grammar = grammar_class("g", names_to_types={"name": str})
    assert grammar["name"] is str


def test_update_error(grammar_class) -> None:
    """Verify update error."""
    grammar = grammar_class("g1")

    msg = "The element name must be a type or None: it is 0."
    with pytest.raises(TypeError, match=msg):
        grammar.update_from_types({"name": 0})


@pytest.mark.parametrize(
    ("names_to_types", "data"),
    [
        # None values element means any type.
        ({"name": None}, {"name": {}}),
    ],
)
def test_validate(grammar_class, names_to_types, data) -> None:
    """Verify validate."""
    grammar = grammar_class("g", names_to_types=names_to_types)
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
def test_validate_error(data, error_msg, raise_exception, caplog) -> None:
    """Verify that validate raises the expected errors."""
    grammar = SimpleGrammar(
        "g", names_to_types={"name1": None, "name2": int}, required_names=["name1"]
    )

    if raise_exception:
        with pytest.raises(InvalidDataError, match=error_msg):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


def test_update_with_merge_error(grammar_class):
    """Verify that any update method raises when merging."""
    grammar = grammar_class("g")
    match = rf"Merge is not supported for {grammar_class.__name__}."

    for method_name in (
        "update",
        "update_from_names",
        "update_from_types",
        "update_from_data",
    ):
        with pytest.raises(ValueError, match=match):
            getattr(grammar, method_name)({"name": bool}, merge=True)
