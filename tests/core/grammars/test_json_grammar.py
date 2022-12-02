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
from gemseo.core.discipline_data import Data
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from numpy import ndarray

DATA_PATH = Path(__file__).parent / "data"


def new_grammar(schema_path: Path | None) -> JSONGrammar:
    """Create a grammar."""
    g = JSONGrammar("g")
    if schema_path is not None:
        g.update_from_file(schema_path)
    return g


def assert_reset_dependencies(grammar: JSONGrammar) -> None:
    """Verify that the dependencies has been reset."""
    assert grammar._JSONGrammar__validator is None
    assert grammar._JSONGrammar__schema == {}


def test_init_error():
    """Verify that init raises the expected errors."""
    msg = "The grammar name cannot be empty."
    with pytest.raises(ValueError, match=msg):
        JSONGrammar("")

    path = "foo"
    msg = f"Cannot update the grammar from non existing file: {path}."
    with pytest.raises(FileNotFoundError, match=msg):
        JSONGrammar("g", schema_path="foo")


def test_init_with_name():
    """Verify init defaults."""
    g = JSONGrammar("g")
    assert g.name == "g"
    assert not g


def test_init_with_file():
    """Verify initializing with a file."""
    g = new_grammar(DATA_PATH / "grammar_2.json")
    assert g
    assert list(g.keys()) == ["name1", "name2"]
    assert g.required_names == {"name1"}


def test_init_with_file_and_descriptions():
    """Verify initializing with a file and descriptions."""
    descriptions = {"name1": "name1 description", "name2": "name2 description"}
    g = JSONGrammar(
        "g",
        schema_path=DATA_PATH / "grammar_3.json",
        descriptions=descriptions,
    )

    assert g
    assert list(g.keys()) == ["name1", "name2"]
    assert g.required_names == {"name1"}
    assert g.schema["properties"]["name1"]["description"] == "name1 description"
    for item in g.schema["properties"]["name2"]["anyOf"]:
        assert item["description"] == "name2 description"


def test_delitem_error():
    """Verify that removing a non-existing item raises."""
    g = JSONGrammar("g")
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        del g["foo"]


def test_delitem():
    """Verify removing an item."""
    g = new_grammar(DATA_PATH / "grammar_2.json")
    del g["name1"]
    assert "name1" not in g
    assert "name1" not in g.required_names
    assert "name2" in g
    assert_reset_dependencies(g)


def test_getitem_error():
    """Verify that getting a non-existing item raises."""
    g = JSONGrammar("g")
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        g["foo"]


def test_getitem():
    """Verify getting an item."""
    g = new_grammar(DATA_PATH / "grammar_2.json")
    assert g["name1"]._active_strategies[0].PYTHON_TYPES == (int, float)


@pytest.mark.parametrize(
    "schema_path,length",
    [
        (None, 0),
        (DATA_PATH / "grammar_2.json", 2),
    ],
)
def test_len(schema_path, length):
    """Verify computing the length."""
    g = new_grammar(schema_path)
    assert len(g) == length


@pytest.mark.parametrize(
    "schema_path,names",
    [
        (None, []),
        (DATA_PATH / "grammar_2.json", ["name1", "name2"]),
    ],
)
def test_iter(schema_path, names):
    """Verify iterating."""
    g = new_grammar(schema_path)
    assert list(iter(g)) == names


@pytest.mark.parametrize(
    "schema_path,names",
    [
        (None, []),
        (DATA_PATH / "grammar_2.json", ["name1", "name2"]),
    ],
)
def test_names(schema_path, names):
    """Verify names getter."""
    g = new_grammar(schema_path)
    assert list(g.names) == names


exclude_names = pytest.mark.parametrize(
    "exclude_names",
    [
        None,
        [],
        ["name1"],
        ["name2"],
    ],
)


@pytest.mark.parametrize(
    "schema_path1",
    [
        None,
        DATA_PATH / "grammar_2.json",
        DATA_PATH / "grammar_3.json",
    ],
)
@pytest.mark.parametrize(
    "schema_path2",
    [
        None,
        DATA_PATH / "grammar_2.json",
        DATA_PATH / "grammar_3.json",
    ],
)
@pytest.mark.parametrize("method_is_update", [True, False])
@exclude_names
def test_update_and_update_from_file(
    schema_path1, schema_path2, method_is_update, exclude_names
):
    """Verify update and update_from_file."""
    g1 = new_grammar(schema_path1)
    g1_names_before = set(g1.keys())
    g1_required_names_before = set(g1.required_names)
    g2 = new_grammar(schema_path2)

    if method_is_update:
        g1.update(g2, exclude_names=exclude_names)
    elif schema_path2 is None:
        # Nothing to be done
        return
    else:
        g1.update_from_file(schema_path2)

    if exclude_names is None or not method_is_update:
        exclude_names = set()
    else:
        exclude_names = set(exclude_names)

    assert set(g1) == g1_names_before | (set(g2) - exclude_names)
    assert set(g1.required_names) == g1_required_names_before | (
        set(g2.required_names) - exclude_names
    )
    assert_reset_dependencies(g1)


@pytest.mark.parametrize(
    "schema_path",
    [
        None,
        DATA_PATH / "grammar_2.json",
    ],
)
def test_clear(schema_path):
    """Verify clear."""
    g = new_grammar(schema_path)
    g.clear()
    assert not g
    assert not g.required_names
    assert_reset_dependencies(g)


@pytest.mark.parametrize(
    "schema_path,repr_",
    [
        (
            None,
            """
Grammar name: g, schema: {
  "$schema": "http://json-schema.org/schema#"
}
""",
        ),
        (
            DATA_PATH / "grammar_2.json",
            """
Grammar name: g, schema: {
  "$schema": "http://json-schema.org/draft-04/schema",
  "additionalProperties": false,
  "type": "object",
  "properties": {
    "name1": {
      "type": "integer"
    },
    "name2": {
      "type": "array",
      "items": {
        "type": "number"
      }
    }
  },
  "required": [
    "name1"
  ]
}
""",
        ),
    ],
)
def test_repr(schema_path, repr_):
    """Verify repr."""
    g = new_grammar(schema_path)
    assert repr(g) == repr_.strip()


@pytest.mark.parametrize(
    "schema_path,data_sets",
    (
        # Empty grammar: everything validates.
        (None, ({"name": 0},)),
        (
            DATA_PATH / "grammar_3.json",
            (
                {"name1": 1},
                {"name1": 1, "name2": "bar"},
                {"name1": 1, "name2": 0},
            ),
        ),
    ),
)
def test_validate(schema_path, data_sets):
    """Verify validate."""
    g = JSONGrammar("g", schema_path=schema_path)
    for data in data_sets:
        g.validate(data)


@pytest.mark.parametrize("raise_exception", [True, False])
@pytest.mark.parametrize(
    "data,error_msg",
    [
        ({}, r"Missing required names: name1."),
        (
            {"name1": 0, "name2": ""},
            r"error: data.name2 must be array",
        ),
    ],
)
def test_validate_error(raise_exception, data, error_msg, caplog):
    """Verify that validate raises the expected errors."""
    g = new_grammar(DATA_PATH / "grammar_2.json")

    if raise_exception:
        with pytest.raises(InvalidDataException, match=error_msg):
            g.validate(data)
    else:
        g.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


@pytest.mark.parametrize(
    "schema_path",
    [
        None,
        DATA_PATH / "grammar_2.json",
    ],
)
@pytest.mark.parametrize(
    "names",
    [
        [],
        ["name1"],
        ["name2"],
    ],
)
@exclude_names
def test_update(schema_path, names, exclude_names):
    """Verify update with names."""
    g = new_grammar(schema_path)
    names_before = set(g.keys())
    required_names_before = set(g.required_names)

    g.update(names, exclude_names=exclude_names)

    if exclude_names is None:
        exclude_names = set()
    else:
        exclude_names = set(exclude_names)

    assert_reset_dependencies(g)
    assert set(g) == names_before | (set(names) - exclude_names)
    assert g.required_names == required_names_before | (set(names) - exclude_names)

    if not set(names) - set(exclude_names):
        return

    name = names[0]
    property = g.schema["properties"][name]

    if name == "name1" and schema_path:
        assert property == {
            "anyOf": [
                {"type": "integer"},
                {"type": "array", "items": {"type": "number"}},
            ]
        }

    if name == "name2" or not schema_path:
        assert property == {"type": "array", "items": {"type": "number"}}


@pytest.mark.parametrize(
    "data,expected_type",
    [
        ({}, "integer"),
        ({"name1": 0}, "integer"),
        ({"name1": 0.0}, "number"),
        ({"name1": ""}, "string"),
        ({"name1": True}, "boolean"),
        ({"name1": ndarray([0])}, "array"),
        ({"name1": {"name2": 0}}, "object"),
    ],
)
def test_update_from_data_with_empty(data, expected_type):
    """Verify update_from_data from an empty grammar."""
    g = _test_update_from_data(None, data)

    if not g:
        return

    assert g.schema["properties"]["name1"]["type"] == expected_type


@pytest.mark.parametrize(
    "data,expected_type",
    [
        ({}, "integer"),
        ({"name1": 0}, "integer"),
        ({"name1": 0.0}, "number"),
        ({"name1": ""}, ["integer", "string"]),
        ({"name1": True}, ["boolean", "integer"]),
        ({"name1": ndarray([0])}, ["array", "integer"]),
        ({"name1": {"name2": 0}}, "object"),
    ],
)
def test_update_from_data_with_non_empty(data, expected_type):
    """Verify update_from_data from a non empty grammar."""
    g = _test_update_from_data(DATA_PATH / "grammar_2.json", data)

    if isinstance(data.get("name1"), dict):
        assert g.schema["properties"]["name1"] == {
            "anyOf": [
                {"type": "integer"},
                {
                    "type": "object",
                    "properties": {"name2": {"type": "integer"}},
                    "required": ["name2"],
                },
            ]
        }
    else:
        assert g.schema["properties"]["name1"]["type"] == expected_type


def _test_update_from_data(schema_path: Path | None, data: Data):
    """Helper function for testing update_from_data."""
    g = new_grammar(schema_path)
    names_before = set(g.keys())
    required_names_before = set(g.required_names)

    g.update_from_data(data)

    assert_reset_dependencies(g)
    assert set(g) == names_before | set(data)
    assert g.required_names == required_names_before | set(data)

    return g


def test_is_array_error():
    """Verify that is_array error."""
    g = JSONGrammar("g")
    msg = "The name foo is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        g.is_array("foo")


def test_is_array():
    """Verify is_array."""
    g = new_grammar(DATA_PATH / "grammar_2.json")
    assert not g.is_array("name1")
    assert g.is_array("name2")


def test_restrict_to_error():
    """Verify that raises the expected error."""
    g = JSONGrammar("g")
    msg = "The name foo is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        g.restrict_to(["foo"])


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["name1"],
        ["name1", "name2"],
    ],
)
def test_restrict_to(names):
    """Verify restrict_to."""
    g = new_grammar(DATA_PATH / "grammar_2.json")
    required_names_before = set(g.required_names)
    g.restrict_to(names)
    assert set(g) == set(names)
    assert g.required_names == required_names_before & set(names)
    assert_reset_dependencies(g)


@pytest.mark.parametrize(
    "schema_path",
    [
        None,
        DATA_PATH / "grammar_2.json",
    ],
)
def test_convert_to_simple_grammar(schema_path):
    """Verify grammar conversion."""
    g1 = new_grammar(schema_path)
    g2 = g1.convert_to_simple_grammar()
    assert set(g1) == set(g2)
    assert g1.required_names == g2.required_names
    assert isinstance(g2, SimpleGrammar)


def test_convert_to_simple_grammar_not_convertible_type():
    """Verify grammar conversion with non convertible type."""
    g1 = new_grammar(DATA_PATH / "grammar_1.json")
    g2 = g1.convert_to_simple_grammar()
    assert g2["name"] is None


def test_convert_to_simple_grammar_warnings(caplog):
    """Verify grammar conversion warnings."""
    g1 = new_grammar(
        DATA_PATH / "grammar_conversion_to_simple_grammar_warn_for_array.json"
    )
    g2 = g1.convert_to_simple_grammar()
    assert len(g2) == 1
    assert g2["name"] == ndarray
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.messages[0] == (
        "Unsupported type 'string' in JSONGrammar 'g' for property 'name' in "
        "conversion to simple grammar."
    )
    assert caplog.records[1].levelname == "WARNING"
    assert caplog.messages[1] == (
        "Unsupported feature 'contains' in JSONGrammar 'g' for property 'name' in "
        "conversion to simple grammar."
    )


@pytest.mark.parametrize(
    "schema_path,names",
    [
        (None, set()),
        (DATA_PATH / "grammar_2.json", {"name1"}),
    ],
)
def test_required_names(schema_path, names):
    """Verify required_names."""
    g = new_grammar(schema_path)
    assert g.required_names == names


@pytest.mark.parametrize(
    "descriptions",
    (
        {},
        {"name1": "name1 description"},
        {"name1": "name1 description", "name2": "name2 description"},
    ),
)
def test_set_descriptions(descriptions):
    """Verify setting descriptions."""
    g = JSONGrammar(
        "g",
        schema_path=DATA_PATH / "grammar_3.json",
    )
    g.set_descriptions(descriptions)

    if "name1" in descriptions:
        assert g.schema["properties"]["name1"]["description"] == "name1 description"
    else:
        assert "description" not in g.schema["properties"]["name1"]

    if "name2" in descriptions:
        for item in g.schema["properties"]["name2"]["anyOf"]:
            assert item["description"] == "name2 description"
    else:
        assert "description" not in g.schema["properties"]["name2"]


@pytest.mark.parametrize(
    "schema_path,schema",
    [
        (None, {"$schema": "http://json-schema.org/schema#"}),
        (
            DATA_PATH / "grammar_3.json",
            {
                "$schema": "http://json-schema.org/draft-04/schema",
                "additionalProperties": False,
                "properties": {
                    "name1": {"type": "integer"},
                    "name2": {"type": ["integer", "string"]},
                },
                "required": ["name1"],
                "type": "object",
            },
        ),
    ],
)
def test_schema(schema_path, schema):
    """Verify schema getter."""
    g = JSONGrammar("g", schema_path=schema_path)
    assert g.schema == schema


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


@pytest.mark.parametrize("path", (None, "g.json"))
def test_write(path, tmp_wd):
    """Verify write."""
    g = JSONGrammar("g", schema_path=DATA_PATH / "grammar_1.json")
    g.write(path)
    assert Path("g.json").read_text() == EXPECTED_JSON


def test_to_json(tmp_wd):
    """Verify to_json."""
    g = JSONGrammar("g", schema_path=DATA_PATH / "grammar_1.json")
    assert g.to_json(indent=2) == EXPECTED_JSON


def test_rename():
    """Verify rename."""
    g = JSONGrammar("g")
    g.update(["name1", "name2"])

    g.rename_element("name1", "n:name1")
    g.rename_element("name2", "n:name2")

    assert sorted(g.required_names) == ["n:name1", "n:name2"]
    assert "name1" not in g
    assert sorted(g.names) == ["n:name1", "n:name2"]


def test_update_from_error():
    g = JSONGrammar("g")
    g2 = SimpleGrammar("g")
    with pytest.raises(
        TypeError, match="A JSONGrammar cannot be updated from a grammar"
    ):
        g.update(g2)


@pytest.mark.parametrize(
    "var, check_is_numeric_array, expected",
    [
        pytest.param(
            "IDONTEXIST",
            False,
            None,
            marks=pytest.mark.xfail(raises=KeyError, math="is not in the grammar"),
        ),
        ("not_array_var", False, False),
        ("array_number_var", False, True),
        ("array_str_var", False, True),
        ("not_array_var", True, False),
        ("array_number_var", True, True),
        ("array_wo_type", True, True),
        ("array_str_var", True, False),
        ("array_array_number_var", True, True),
        ("array_array_str_var", True, False),
    ],
)
def test_is_type_array_errors(var, check_is_numeric_array, expected):
    fpath = DATA_PATH / "grammar_test6.json"
    gram = JSONGrammar(name="toto", schema_path=fpath)
    assert gram.is_array(var, check_is_numeric_array) == expected
