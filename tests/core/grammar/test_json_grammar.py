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
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
from pathlib import Path

import pytest
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy import ndarray
from pydantic import BaseModel

from gemseo.core.grammar.error import InvalidDataError
from gemseo.core.grammar.json import JSONGrammar
from gemseo.util.testing.helper import assert_exception

DATA_PATH = Path(__file__).parent / "data"


def new_grammar(file_path: Path | None) -> JSONGrammar:
    """Create a grammar."""
    grammar = JSONGrammar("g")
    if file_path is not None:
        grammar.update_from_file(file_path)
    return grammar


def test_to_simple_grammar_empty() -> None:
    """Verify that converting an empty `JSONGrammar` to a `SimpleGrammar` works.

    The legacy implementation called `.items()` on the result of
    `schema.get("properties")` without a default, which raised on an empty
    grammar.
    """
    grammar = JSONGrammar("g")
    simple_grammar = grammar.to_simple_grammar()
    assert dict(simple_grammar) == {}


def test_to_simple_grammar_any_of_property() -> None:
    """Verify that `anyOf` properties degrade to the catch-all `None` type."""
    grammar = JSONGrammar("g")
    grammar.update_from_schema({
        "type": "object",
        "properties": {"x": {"anyOf": [{"type": "integer"}, {"type": "string"}]}},
    })
    simple_grammar = grammar.to_simple_grammar()
    assert simple_grammar["x"] is None


def test_update_from_schema_rejects_namespace_separator(snapshot) -> None:
    """Verify that `update_from_schema` rejects property names with the separator."""
    grammar = JSONGrammar("g")
    schema = {
        "type": "object",
        "properties": {"x:y": {"type": "integer"}},
    }
    with assert_exception(ValueError, snapshot):
        grammar.update_from_schema(schema)


def test_init_with_file_error(snapshot) -> None:
    """Verify that init raises the expected errors."""
    with assert_exception(FileNotFoundError, snapshot):
        JSONGrammar("g", file_path="foo")


def test_update_from_file_missing(snapshot) -> None:
    """Verify that update_from_file raises on a nonexistent file."""
    grammar = JSONGrammar("g")
    with assert_exception(FileNotFoundError, snapshot):
        grammar.update_from_file("does_not_exist.json")


def test_roundtrip_empty_defaults(tmp_wd) -> None:
    """to_file/update_from_file roundtrip when defaults is empty."""
    grammar = JSONGrammar("g")
    grammar.update_from_data({"x": 1})
    assert not grammar.defaults

    path = Path("g.json")
    grammar.to_file(path)

    reloaded = JSONGrammar("g", file_path=path)
    assert set(reloaded) == set(grammar)
    assert set(reloaded.required_names) == set(grammar.required_names)
    assert not reloaded.defaults


def test_unpickle_leaves_no_stray_defaults_attribute() -> None:
    """Verify that unpickling does not leave a raw defaults instance attribute."""
    grammar = JSONGrammar("g")
    grammar.update_from_names(["x"])
    grammar.defaults["x"] = [1.0]

    unpickled = pickle.loads(pickle.dumps(grammar))

    assert "defaults" not in vars(unpickled)
    assert dict(unpickled.defaults) == {"x": [1.0]}


def test_update_from_schema_keeps_other_descriptions() -> None:
    """Verify that update_from_schema does not reset the other descriptions."""
    grammar = JSONGrammar("g")
    grammar.update_from_schema({
        "type": "object",
        "properties": {"name1": {"type": "integer", "description": "Old."}},
    })
    grammar.descriptions["name1"] = "Manual."

    grammar.update_from_schema({
        "type": "object",
        "properties": {"name2": {"type": "integer", "description": "New."}},
    })

    assert grammar.descriptions == {"name1": "Manual.", "name2": "New."}


def test_update_from_file_with_non_ascii(tmp_wd) -> None:
    """Verify that a schema file is read as UTF-8 whatever the locale."""
    grammar = JSONGrammar("g")
    grammar.update_from_names(["x"])
    grammar.descriptions["x"] = "Découpage en éléments finis."

    path = Path("g.json")
    grammar.to_file(path)

    reloaded = JSONGrammar("g", file_path=path)
    assert reloaded.descriptions == dict(grammar.descriptions)


def test_init_with_file() -> None:
    """Verify initializing with a file."""
    grammar = new_grammar(DATA_PATH / "grammar_2.json")
    assert grammar
    assert grammar.keys() == {"name1", "name2"}
    assert grammar.required_names == {"name1"}
    assert grammar.descriptions == {"name2": "The description of name2."}


def test_init_with_file_and_descriptions() -> None:
    """Verify initializing with a file and descriptions."""
    descriptions = {"name1": "name1 description", "name2": "name2 description"}
    grammar = JSONGrammar(
        "g",
        file_path=DATA_PATH / "grammar_3.json",
        descriptions=descriptions,
    )
    assert grammar.descriptions == descriptions
    assert grammar.keys() == {"name1", "name2"}
    assert grammar.required_names == {"name1"}
    assert grammar.schema["properties"]["name1"]["description"] == "name1 description"
    for item in grammar.schema["properties"]["name2"]["anyOf"]:
        # We would expect
        # assert item["description"] == "name2 description"
        # instead of
        assert item["description"] == descriptions["name2"]


def test_getitem() -> None:
    """Verify getting an item."""
    grammar = new_grammar(DATA_PATH / "grammar_2.json")
    assert (int, float, float64, int64) == grammar["name1"]._active_strategies[
        0
    ].PYTHON_TYPES


@pytest.mark.parametrize(
    "file_path1",
    [
        None,
        DATA_PATH / "grammar_2.json",
        DATA_PATH / "grammar_3.json",
    ],
)
@pytest.mark.parametrize(
    "file_path2",
    [
        DATA_PATH / "grammar_2.json",
        DATA_PATH / "grammar_3.json",
    ],
)
def test_update_and_update_from_file(file_path1, file_path2) -> None:
    """Verify update and update_from_file."""
    g1 = new_grammar(file_path1)
    g1_names_before = g1.keys()
    g1_required_names_before = set(g1.required_names)
    g2 = new_grammar(file_path2)

    g1.update_from_file(file_path2)

    assert g1.defaults.keys() == g2.defaults.keys()
    assert set(g1) == g1_names_before | set(g2)
    assert set(g1.required_names) == g1_required_names_before | set(g2.required_names)


def test_update_error(snapshot) -> None:
    """Verify update error."""
    grammar = JSONGrammar("g")

    with assert_exception(TypeError, snapshot):
        grammar.update(True)


def test_update_excludes_name_absent_from_source() -> None:
    """Verify that excluding a name absent from the source grammar is a no-op."""
    source = JSONGrammar("g1")
    source.update_from_names(["x"])

    target = JSONGrammar("g2")
    target.update(source, excluded_names={"absent"})

    assert set(target) == {"x"}


@pytest.mark.parametrize(
    "data",
    [
        {"number": 1.1},
        {"number": 1},
        {"number": 1 + 1j},
        {"number": complex128(3)},
        {"string": "foo"},
        {"string": Path("foo")},
        {"1d_array": array([1, 2])},
        {"2d_array": array([[1, 2], [3, 4]])},
        {
            "list_of_dict_of_1D_arrays": [
                {"x": array([1.0, 2.0, 1.0])},
                {"x": array([1.0, 2.0, 0.0])},
            ]
        },
        {"dict_of_2d_arrays": {"x": array([[1.0, 2.0, 1.0], [1.0, 2.0, 0.0]])}},
    ],
)
def test_validate(data) -> None:
    """Verify validate."""
    grammar = new_grammar(file_path=DATA_PATH / "grammar_5.json")
    data["mandatory"] = True
    grammar.validate(data)


@pytest.mark.parametrize("raise_exception", [True, False])
def test_validate_error(raise_exception, caplog, snapshot) -> None:
    """Verify that validate raises the expected errors."""
    grammar = new_grammar(DATA_PATH / "grammar_2.json")
    data = {"name1": 0, "name2": ""}

    if raise_exception:
        with assert_exception(InvalidDataError, snapshot):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"


@pytest.mark.parametrize("raise_exception", [True, False])
def test_validate_error_required_at_root(raise_exception, caplog, snapshot) -> None:
    """Verify that a root-level "data must contain" error is not duplicated.

    `BaseGrammar.validate` already reports missing required *elements* (checked
    against `required_names`) via a dedicated "Missing required names" message,
    so `JSONGrammar._validate` intentionally skips the validator's own detail when
    its message is exactly rooted at "data" (as opposed to a nested path such as
    "data.x"). Such a root-level error can only come from the validator itself
    when the schema's top-level `required` is defined through a `$ref` to a
    `definitions` entry, since the grammar's own top-level `required` is always
    known through `required_names` and is stripped before compiling the validator.
    """
    grammar = JSONGrammar("g")
    grammar.update_from_schema({
        "definitions": {
            "sub": {
                "type": "object",
                "properties": {"a": {"type": "integer"}},
                "required": ["a"],
            },
        },
        "$ref": "#/definitions/sub",
    })

    if raise_exception:
        with assert_exception(InvalidDataError, snapshot):
            grammar.validate({})
    else:
        grammar.validate({}, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"


def test_convert_to_simple_grammar_not_convertible_type() -> None:
    """Verify grammar conversion with non-convertible type."""
    g1 = new_grammar(DATA_PATH / "grammar_1.json")
    g2 = g1.to_simple_grammar()
    assert g2["name"] is None


@pytest.mark.parametrize(
    "descriptions",
    [
        {},
        {"name1": "name1 description"},
        {"name1": "name1 description", "name2": "name2 description"},
    ],
)
def test_set_descriptions(descriptions) -> None:
    """Verify setting descriptions."""
    grammar = JSONGrammar(
        "g",
        file_path=DATA_PATH / "grammar_3.json",
    )
    grammar.set_descriptions(descriptions)

    descriptions_ = {"name2": "The description of name2."}
    descriptions_.update(descriptions)
    assert grammar.descriptions == descriptions_

    if "name1" in descriptions:
        assert (
            grammar.schema["properties"]["name1"]["description"] == "name1 description"
        )
    else:
        assert "description" not in grammar.schema["properties"]["name1"]

    for item in grammar.schema["properties"]["name2"]["anyOf"]:
        assert item["description"] == descriptions_["name2"]


@pytest.mark.parametrize(
    ("file_path", "schema"),
    [
        (None, {"$schema": "http://json-schema.org/schema#"}),
        (
            DATA_PATH / "grammar_3.json",
            {
                "$schema": "http://json-schema.org/draft-04/schema",
                "additionalProperties": False,
                "properties": {
                    "name1": {"type": "integer"},
                    "name2": {
                        "anyOf": [
                            {
                                "description": "The description of name2.",
                                "type": "string",
                            },
                            {
                                "description": "The description of name2.",
                                "type": "integer",
                            },
                        ],
                    },
                },
                "required": ["name1"],
                "type": "object",
            },
        ),
    ],
)
def test_schema(file_path, schema) -> None:
    """Verify schema getter."""
    grammar = JSONGrammar("g", file_path=file_path)
    assert grammar.schema == schema


EXPECTED_JSON = """
{
  "$schema": "http://json-schema.org/draft-04/schema",
  "additionalProperties": false,
  "type": "object",
  "properties": {
    "name": {
      "type": "object"
    }
  },
  "required": [
    "name"
  ]
}
""".strip()


@pytest.mark.parametrize("path", [None, "g.json"])
def test_write(path, tmp_wd) -> None:
    """Verify write."""
    grammar = JSONGrammar("g", file_path=DATA_PATH / "grammar_1.json")
    grammar.to_file(path)
    assert Path("g.json").read_text() == EXPECTED_JSON


def test_to_json(tmp_wd) -> None:
    """Verify to_json."""
    grammar = JSONGrammar("g", file_path=DATA_PATH / "grammar_1.json")
    assert grammar.to_json(indent=2) == EXPECTED_JSON


def test_to_json_error_does_not_leak_required_names() -> None:
    """Verify that a failing to_json does not leave stale required names behind."""
    grammar = JSONGrammar("g")
    grammar.update_from_names(["x"])
    grammar.defaults["x"] = object()

    with pytest.raises(TypeError):
        grammar.to_json()

    del grammar.defaults["x"]
    grammar.required_names.remove("x")
    assert "required" not in grammar.schema


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("foo", "foo"),
        (Path("foo"), "foo"),
        (3 + 1j, 3),
        (array([1, 2]), [1, 2]),
        (array([[1, 2], [3, 4]]), [[1, 2], [3, 4]]),
        (complex128(3), 3),
        (
            [1, array([1, 2]), "foo", [3, array([3, 4])]],
            [1, [1, 2], "foo", [3, [3, 4]]],
        ),
        (
            {"x": array([1, 2]), "y": ["foo", {"z": array([2, 3])}]},
            {"x": [1, 2], "y": ["foo", {"z": [2, 3]}]},
        ),
    ],
)
def test_default_value_is_json_castable(value, expected) -> None:
    """Defaults of any value are cast to a JSON-interpretable representation."""
    grammar = JSONGrammar("g")
    grammar.update_from_names(["x"])
    grammar.defaults["x"] = value
    assert grammar.schema["properties"]["x"]["default"] == expected


@pytest.mark.parametrize("type_", [float, complex])
@pytest.mark.parametrize("value", [1.0, 1.0 + 1.0j])
def test_to_simple_grammar_float_complex(type_, value):
    """Check that a JSONGrammar.to_simple_grammar() can validate float and complex."""
    data = {"x": value}

    json_grammar = JSONGrammar("g_json")
    json_grammar.update_from_types({"x": type_})
    json_grammar.validate(data)

    simple_grammar = json_grammar.to_simple_grammar()
    simple_grammar.validate(data)

    # Warning:
    # This SimpleGrammar validates complex data with imaginary part when type_ is float
    # because JSONGrammar cannot distinguish between float and complex
    # and thus creates the SimpleGrammar with the most generic type, which is complex.


def test_update_from_types():
    """Verify that a JSONGrammar can be updated from types."""
    grammar = JSONGrammar("g")
    grammar.update_from_types({
        "ndarray": ndarray,
        "list": list,
        "tuple": tuple,
        "str": str,
        "int": int,
        "bool": bool,
        "complex": complex,
        "Complex": complex,
        "float": float,
        "None": None,
    })

    assert grammar.schema["properties"] == {
        "None": {},
        "bool": {
            "type": "boolean",
        },
        "Complex": {
            "type": "number",
        },
        "complex": {
            "type": "number",
        },
        "float": {
            "type": "number",
        },
        "int": {
            "type": "integer",
        },
        "list": {
            "type": "array",
        },
        "ndarray": {
            "type": "array",
            "items": {"type": "number"},
        },
        "str": {
            "type": "string",
        },
        "tuple": {
            "type": "array",
        },
    }


def test_update_from_types_error(snapshot):
    """Verify error when updated from a bad type."""
    grammar = JSONGrammar("g")
    with assert_exception(TypeError, snapshot):
        grammar.update_from_types({"x": set})


def test_repr_with_nested_property() -> None:
    """Verify that repr recurses into nested property schemas, e.g. array items."""
    grammar = JSONGrammar("g")
    grammar.update_from_types({"x": ndarray})
    representation = repr(grammar)
    assert "Type:" in representation
    assert "Items:" in representation


def test_default_value_is_json_castable_for_pydantic_model() -> None:
    """Defaults that are pydantic models are cast to their model_dump() dict."""

    class _Model(BaseModel):
        a: int = 1
        b: float = 2.0

    grammar = JSONGrammar("g")
    grammar.update_from_names(["x"])
    grammar.defaults["x"] = _Model()
    assert grammar.schema["properties"]["x"]["default"] == {"a": 1, "b": 2.0}
