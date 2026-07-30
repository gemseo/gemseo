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

from collections.abc import Mapping

import pytest

from gemseo.core.data_converter.simple import SimpleGrammarDataConverter
from gemseo.core.grammar.error import InvalidDataError
from gemseo.core.grammar.simple import SimpleGrammar
from gemseo.core.grammar.simpler import SimplerGrammar
from gemseo.util.testing.helper import assert_exception


@pytest.fixture(params=(SimpleGrammar, SimplerGrammar))
def grammar_class(request):
    """Iterate over the simple grammars."""
    return request.param


class _ClassConverterGrammar(SimpleGrammar):
    """A SimpleGrammar subclass whose converter is set as a class, not a name."""

    DATA_CONVERTER_CLASS = SimpleGrammarDataConverter


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


def test_init_with_defaults(grammar_class) -> None:
    """Verify that elements with a default are not marked required."""
    grammar = grammar_class(
        "g", name_to_type={"x": int, "y": str}, defaults={"y": "foo"}
    )
    assert set(grammar.required_names) == {"x"}
    assert dict(grammar.defaults) == {"y": "foo"}


def test_init_required_names_overrides_defaults(grammar_class) -> None:
    """Verify that explicit `required_names` overrides the defaults-based logic."""
    grammar = grammar_class(
        "g",
        name_to_type={"x": int, "y": str},
        required_names=["x", "y"],
        defaults={"y": "foo"},
    )
    assert set(grammar.required_names) == {"x", "y"}
    assert dict(grammar.defaults) == {"y": "foo"}


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


@pytest.mark.parametrize("raise_exception", [True, False])
def test_validate_bad_type(raise_exception, caplog, snapshot) -> None:
    """SimpleGrammar reports type mismatches; SimplerGrammar skips them."""
    grammar = SimpleGrammar(
        "g", name_to_type={"name1": None, "name2": int}, required_names=["name1"]
    )
    data = {"name1": 0, "name2": ""}

    if raise_exception:
        with assert_exception(InvalidDataError, snapshot):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"


def test_simpler_grammar_skips_type_checks(caplog) -> None:
    """SimplerGrammar accepts type mismatches silently."""
    grammar = SimplerGrammar(
        "g", name_to_type={"name1": None, "name2": int}, required_names=["name1"]
    )
    grammar.validate({"name1": 0, "name2": ""})
    assert not caplog.records


def test_to_simple_grammar_returns_a_copy(grammar_class) -> None:
    """Verify that `to_simple_grammar` returns an independent copy."""
    grammar = grammar_class("g", name_to_type={"x": int})
    converted = grammar.to_simple_grammar()
    assert converted is not grammar
    converted.update_from_types({"y": float})
    assert "y" in converted
    assert "y" not in grammar


def test_get_name_to_type_returns_a_copy(grammar_class) -> None:
    """Verify that `_get_name_to_type` returns a copy of the name-to-type mapping."""
    grammar = grammar_class("g", name_to_type={"x": int})
    mapping = grammar._get_name_to_type()
    assert mapping == {"x": int}
    mapping["y"] = float
    assert "y" not in grammar


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


def test_data_converter_class_not_a_string() -> None:
    """Verify that DATA_CONVERTER_CLASS may be set to a class instead of a name."""
    grammar = _ClassConverterGrammar("g")
    assert isinstance(grammar.data_converter, SimpleGrammarDataConverter)


def test_update_from_types_generalizes_dict_to_mapping(grammar_class) -> None:
    """Verify that a dict type is generalized to collections.abc.Mapping."""
    grammar = grammar_class("g")
    grammar.update_from_types({"x": dict})
    assert grammar["x"] is Mapping


def test_schema_without_required_names(grammar_class) -> None:
    """Verify that the schema has no "required" key when nothing is required."""
    grammar = grammar_class("g", name_to_type={"x": int}, required_names=[])
    schema = grammar.schema
    assert "required" not in schema
    assert "x" in schema["properties"]
