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

from pathlib import Path

import pytest
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import int64
from numpy import ndarray

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.json_grammar import JSONGrammar

DATA_PATH = Path(__file__).parent / "data"


def new_grammar(file_path: Path | None) -> JSONGrammar:
    """Create a grammar."""
    grammar = JSONGrammar("g")
    if file_path is not None:
        grammar.update_from_file(file_path)
    return grammar


def assert_reset_dependencies(grammar: JSONGrammar) -> None:
    """Verify that the dependencies has been reset."""
    assert grammar._JSONGrammar__validator is None
    assert grammar._JSONGrammar__schema == {}


def test_init_with_file_error() -> None:
    """Verify that init raises the expected errors."""
    path = "foo"
    msg = f"Cannot update the grammar from non existing file: {path}."
    with pytest.raises(FileNotFoundError, match=msg):
        JSONGrammar("g", file_path="foo")


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
        None,
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

    if file_path2 is None:
        # Nothing to be done
        return

    g1.update_from_file(file_path2)

    assert g1.defaults.keys() == g2.defaults.keys()
    assert set(g1) == g1_names_before | set(g2)
    assert set(g1.required_names) == g1_required_names_before | set(g2.required_names)
    assert_reset_dependencies(g1)


def test_update_error() -> None:
    """Verify update error."""
    grammar = JSONGrammar("g")

    msg = "A JSONGrammar cannot be updated from a grammar of type: <class 'bool'>"
    with pytest.raises(TypeError, match=msg):
        grammar.update(True)


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
@pytest.mark.parametrize(
    ("data", "error_msg"),
    [
        (
            {"name1": 0, "name2": ""},
            r"error: data.name2 must be array",
        ),
    ],
)
def test_validate_error(raise_exception, data, error_msg, caplog) -> None:
    """Verify that validate raises the expected errors."""
    grammar = new_grammar(DATA_PATH / "grammar_2.json")

    if raise_exception:
        with pytest.raises(InvalidDataError, match=error_msg):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


def test_convert_to_simple_grammar_not_convertible_type() -> None:
    """Verify grammar conversion with non-convertible type."""
    g1 = new_grammar(DATA_PATH / "grammar_1.json")
    g2 = g1.to_simple_grammar()
    assert g2["name"] is None


def test_convert_to_simple_grammar_warnings(caplog) -> None:
    """Verify grammar conversion warnings."""
    g1 = new_grammar(
        DATA_PATH / "grammar_conversion_to_simple_grammar_warn_for_array.json"
    )
    g2 = g1.to_simple_grammar()
    assert len(g2) == 1
    assert g2["name"] == ndarray
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.messages[0] == (
        "Unsupported type 'string' in JSONGrammar 'g' for property 'name' in "
        "conversion to SimpleGrammar."
    )
    assert caplog.records[1].levelname == "WARNING"
    assert caplog.messages[1] == (
        "Unsupported feature 'contains' in JSONGrammar 'g' for property 'name' in "
        "conversion to SimpleGrammar."
    )


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
def test_cast(value, expected) -> None:
    """Check the method casting any value to a JSON-interpretable one."""
    assert JSONGrammar._JSONGrammar__cast_value(value) == expected


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
