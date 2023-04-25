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

from numbers import Number
from pathlib import Path

import pytest
from gemseo.core.discipline_data import Data
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from numpy import array
from numpy import ndarray

DATA_PATH = Path(__file__).parent / "data"


def new_grammar(file_path: Path | None) -> JSONGrammar:
    """Create a grammar."""
    g = JSONGrammar("g")
    if file_path is not None:
        g.update_from_file(file_path)
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
        JSONGrammar("g", file_path="foo")


def test_init_with_name():
    """Verify init defaults."""
    g = JSONGrammar("g")
    assert g.name == "g"
    assert not g
    assert not g.defaults


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
        file_path=DATA_PATH / "grammar_3.json",
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
    "file_path,length",
    [
        (None, 0),
        (DATA_PATH / "grammar_2.json", 2),
    ],
)
def test_len(file_path, length):
    """Verify computing the length."""
    g = new_grammar(file_path)
    assert len(g) == length


@pytest.mark.parametrize(
    "file_path,names",
    [
        (None, []),
        (DATA_PATH / "grammar_2.json", ["name1", "name2"]),
    ],
)
def test_iter(file_path, names):
    """Verify iterating."""
    g = new_grammar(file_path)
    assert list(iter(g)) == names


@pytest.mark.parametrize(
    "file_path,names",
    [
        (None, []),
        (DATA_PATH / "grammar_2.json", ["name1", "name2"]),
    ],
)
def test_names(file_path, names):
    """Verify names getter."""
    g = new_grammar(file_path)
    assert list(g.names) == names


exclude_names = pytest.mark.parametrize(
    "exclude_names",
    [
        (),
        ["name1"],
        ["name2"],
    ],
)


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
@pytest.mark.parametrize("method_is_update", [True, False])
@exclude_names
def test_update_and_update_from_file(
    file_path1, file_path2, method_is_update, exclude_names
):
    """Verify update and update_from_file."""
    g1 = new_grammar(file_path1)
    g1_names_before = set(g1.keys())
    g1_required_names_before = set(g1.required_names)
    g2 = new_grammar(file_path2)

    if method_is_update:
        for name in g2.names:
            g2.defaults[name] = array([0.0])
        g1.update(g2, exclude_names=exclude_names)
    elif file_path2 is None:
        # Nothing to be done
        return
    else:
        g1.update_from_file(file_path2)

    if not method_is_update:
        exclude_names = set()
    else:
        exclude_names = set(exclude_names)

    assert set(g1.defaults.keys()) == set(g2.defaults.keys()) - exclude_names

    assert set(g1) == g1_names_before | (set(g2) - exclude_names)
    assert set(g1.required_names) == g1_required_names_before | (
        set(g2.required_names) - exclude_names
    )
    assert_reset_dependencies(g1)


def test_update_error():
    """Verify update error."""
    g = JSONGrammar("g")

    msg = "A JSONGrammar cannot be updated from a grammar of type: <class 'NoneType'>"
    with pytest.raises(TypeError, match=msg):
        g.update(None)


@pytest.mark.parametrize(
    "file_path",
    [
        None,
        DATA_PATH / "grammar_2.json",
    ],
)
def test_clear(file_path):
    """Verify clear."""
    g = new_grammar(file_path)
    g.clear()
    assert not g
    assert not g.required_names
    assert_reset_dependencies(g)


@pytest.mark.parametrize(
    "file_path,repr_",
    [
        (
            None,
            """
Grammar name: g
   Required elements:
   Optional elements:
""",
        ),
        (
            DATA_PATH / "grammar_2.json",
            """
Grammar name: g
   Required elements:
      name1: {'type': 'integer'}
   Optional elements:
      name2: {'type': 'array', 'items': {'type': 'number'}}
         default: foo
""",
        ),
    ],
)
def test_repr(file_path, repr_):
    """Verify repr."""
    g = new_grammar(file_path)
    if g:
        g.defaults["name2"] = "foo"
    assert repr(g) == repr_.strip()


@pytest.mark.parametrize(
    "file_path,data_sets",
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
def test_validate(file_path, data_sets):
    """Verify validate."""
    g = JSONGrammar("g", file_path=file_path)
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
        with pytest.raises(InvalidDataError, match=error_msg):
            g.validate(data)
    else:
        g.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


@pytest.mark.parametrize(
    "file_path",
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
@pytest.mark.parametrize("merge", (True, False))
def test_update_from_names(file_path, names, merge):
    """Verify update with names."""
    g = new_grammar(file_path)
    names_before = set(g.keys())
    required_names_before = set(g.required_names)

    g.update_from_names(names, merge=merge)

    assert_reset_dependencies(g)
    assert set(g) == names_before | set(names)
    assert g.required_names == required_names_before | set(names)

    if not names:
        return

    name = names[0]
    property = g.schema["properties"][name]

    if name == "name1" and file_path:
        if merge:
            assert property == {
                "anyOf": [
                    {"type": "integer"},
                    {"type": "array", "items": {"type": "number"}},
                ]
            }
        else:
            assert property == {"type": "array", "items": {"type": "number"}}

    if name == "name2" or not file_path:
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
@pytest.mark.parametrize("merge", (True, False))
def test_update_from_data_with_empty(data, expected_type, merge):
    """Verify update_from_data from an empty grammar."""
    g = _test_update_from_data(None, data, merge)

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
        ({"name1": True}, ["integer", "boolean"]),
        ({"name1": ndarray([0])}, ["integer", "array"]),
        ({"name1": {"name2": 0}}, "object"),
    ],
)
@pytest.mark.parametrize("merge", (True, False))
def test_update_from_data_with_non_empty(data, expected_type, merge):
    """Verify update_from_data from a non empty grammar."""
    g = _test_update_from_data(DATA_PATH / "grammar_2.json", data, merge)

    if isinstance(data.get("name1"), dict):
        if merge:
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
            assert g.schema["properties"]["name1"] == {
                "type": "object",
                "properties": {"name2": {"type": "integer"}},
                "required": ["name2"],
            }
    else:
        if not merge and not isinstance(expected_type, str):
            expected_type_ = expected_type[1]
        else:
            if isinstance(expected_type, str):
                expected_type_ = expected_type
            else:
                # genson sorts the types
                expected_type_ = sorted(expected_type)
        assert g.schema["properties"]["name1"]["type"] == expected_type_


def _test_update_from_data(file_path: Path | None, data: Data, merge):
    """Helper function for testing update_from_data."""
    g = new_grammar(file_path)
    names_before = set(g.keys())
    required_names_before = set(g.required_names)

    g.update_from_data(data, merge)

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

    for name in g.names:
        g.defaults[name] = array([2.0])

    required_names_before = set(g.required_names)
    g.restrict_to(names)
    assert set(g) == set(names)
    assert g.required_names == required_names_before & set(names)
    assert_reset_dependencies(g)

    for name in names:
        assert g.defaults[name] == 2.0

    assert len(g.defaults) == len(names)


@pytest.mark.parametrize(
    "file_path",
    [
        None,
        DATA_PATH / "grammar_2.json",
    ],
)
def test_convert_to_simple_grammar(file_path):
    """Verify grammar conversion."""
    g1 = new_grammar(file_path)
    g2 = g1.to_simple_grammar()
    assert set(g1) == set(g2)
    assert g1.required_names == g2.required_names
    assert isinstance(g2, SimpleGrammar)


def test_convert_to_simple_grammar_not_convertible_type():
    """Verify grammar conversion with non-convertible type."""
    g1 = new_grammar(DATA_PATH / "grammar_1.json")
    g2 = g1.to_simple_grammar()
    assert g2["name"] is None


def test_convert_to_simple_grammar_warnings(caplog):
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
        "conversion to simple grammar."
    )
    assert caplog.records[1].levelname == "WARNING"
    assert caplog.messages[1] == (
        "Unsupported feature 'contains' in JSONGrammar 'g' for property 'name' in "
        "conversion to simple grammar."
    )


@pytest.mark.parametrize(
    "file_path,names",
    [
        (None, set()),
        (DATA_PATH / "grammar_2.json", {"name1"}),
    ],
)
def test_required_names(file_path, names):
    """Verify required_names."""
    g = new_grammar(file_path)
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
        file_path=DATA_PATH / "grammar_3.json",
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
    "file_path,schema",
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
def test_schema(file_path, schema):
    """Verify schema getter."""
    g = JSONGrammar("g", file_path=file_path)
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
    g = JSONGrammar("g", file_path=DATA_PATH / "grammar_1.json")
    g.to_file(path)
    assert Path("g.json").read_text() == EXPECTED_JSON


def test_to_json(tmp_wd):
    """Verify to_json."""
    g = JSONGrammar("g", file_path=DATA_PATH / "grammar_1.json")
    assert g.to_json(indent=2) == EXPECTED_JSON


def test_rename():
    """Verify rename."""
    g = JSONGrammar("g")
    g.update_from_names(["name1", "name2"])
    g.defaults["name1"] = 1.0
    g.required_names.remove("name2")

    g.rename_element("name1", "new_name1")
    g.rename_element("name2", "new_name2")

    assert sorted(g.required_names) == ["new_name1"]
    assert "name1" not in g
    assert sorted(g.names) == ["new_name1", "new_name2"]

    assert g.defaults == {"new_name1": 1.0}


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
    gram = JSONGrammar(name="toto", file_path=fpath)
    assert gram.is_array(var, check_is_numeric_array) == expected


def test_copy():
    """Verify copy."""
    g = JSONGrammar("g")
    g.update_from_names(["name"])
    g.defaults["name"] = 1.0
    g_copy = g.copy()
    # Contrary to the simple grammar, the items values are not shared because the
    # schema builder is deeply copied.
    assert g_copy.defaults["name"] is g.defaults["name"]
    assert list(g_copy.required_names)[0] is list(g.required_names)[0]


@pytest.mark.parametrize(
    "data",
    [
        [1.0, "s"],
        [[1, 2], ["a", "b"], (1, 2), ("a", "b")],
        [array([1, 2]), [1.0, 2.0]],
        [False, True, 1],
    ],
)
def test_update_from_types(data):
    """Test the consistency between the python types and the data validation."""
    data_dict = {str(i): value for i, value in enumerate(data)}
    names_to_types = {name: type(value) for name, value in data_dict.items()}
    grammar = JSONGrammar("test")
    grammar.update_from_types(names_to_types)
    grammar.validate(data_dict)


def test_update_from_types_two_elements():
    """Tests an update from types with two elements of different types."""
    grammar = JSONGrammar("test")
    grammar.update_from_types({"i": int, "x": float})
    assert grammar.required_names == {"i", "x"}
    assert grammar.schema["properties"] == {
        "i": {"type": "integer"},
        "x": {"type": "number"},
    }


def test_empty_types():
    """Test update_from_types with empty payload."""
    grammar = JSONGrammar("test")
    grammar.update_from_types({})
    assert not grammar


@pytest.mark.parametrize(
    "py_type, json_type",
    [
        [int, "integer"],
        [float, "number"],
        [ndarray, "array"],
        [list, "array"],
        [str, "string"],
        [bool, "boolean"],
        [Number, "number"],
    ],
)
def test_update_from_types_basic(py_type, json_type):
    """Tests with all supported basic types."""
    names_to_types = {"name": py_type}
    grammar = JSONGrammar("test")
    grammar.update_from_types(names_to_types)
    assert grammar.required_names == {"name"}
    assert grammar.schema["properties"] == {
        "name": {"type": json_type},
    }


def test_from_types_unsupported():
    grammar = JSONGrammar("test")
    with pytest.raises(
        KeyError, match="Unsupported python type for a JSON Grammar: <class 'complex'>"
    ):
        grammar.update_from_types({"x": complex})
