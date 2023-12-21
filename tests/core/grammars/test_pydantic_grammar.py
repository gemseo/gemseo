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

import re
from enum import Enum
from enum import auto
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Callable
from typing import Union

import pytest
from numpy import array
from numpy import ndarray
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.pydantic_grammar import ModelType
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.pydantic_ndarray import NDArrayPydantic
from gemseo.core.grammars.pydantic_ndarray import _NDArrayPydantic
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.testing.helpers import do_not_raise

from .pydantic_models import get_model1
from .pydantic_models import get_model2
from .pydantic_models import get_model3

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest

    from gemseo.core.discipline_data import Data

DATA_PATH = Path(__file__).parent / "data"


class ModelID(Enum):
    """Enumeration for selecting test models."""

    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()


model1 = pytest.fixture(get_model1)
model2 = pytest.fixture(get_model2)
model3 = pytest.fixture(get_model3)


@pytest.fixture()
def model(
    request: SubRequest, model1: ModelType, model2: ModelType
) -> ModelType | None:
    """Return a pydantic model.

    This fixture can be optionally parametrized.
    """
    if request.param is None:
        return None

    if request.param == ModelID.ONE:
        return model1

    if request.param == ModelID.TWO:
        return model2
    return None


def test_init_error():
    """Verify that init raises the expected errors."""
    msg = "The grammar name cannot be empty."
    with pytest.raises(ValueError, match=msg):
        PydanticGrammar("")


def test_init_with_name():
    """Verify init defaults."""
    g = PydanticGrammar("g")
    assert g.name == "g"
    assert not g


def test_init_with_model(model1):
    """Verify initializing with a file."""
    g = PydanticGrammar("g", model=model1)
    assert g
    assert list(g.keys()) == ["name1", "name2"]
    assert g.required_names == {"name1"}


def test_delitem_error():
    """Verify that removing a non-existing item raises."""
    g = PydanticGrammar("g")
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        del g["foo"]


def test_delitem(model1):
    """Verify removing an item."""
    g = PydanticGrammar("g", model=model1)
    del g["name1"]
    assert "name1" not in g
    assert "name1" not in g.required_names
    assert "name2" in g


def test_getitem_error():
    """Verify that getting a non-existing item raises."""
    g = PydanticGrammar("g")
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        g["foo"]


def test_getitem(model1):
    """Verify getting an item."""
    g = PydanticGrammar("g", model=model1)
    assert_equal_types(g["name1"], int)


@pytest.mark.parametrize(
    ("model", "length"),
    [
        (None, 0),
        (ModelID.ONE, 2),
    ],
    indirect=["model"],
)
def test_len(model, length):
    """Verify computing the length."""
    g = PydanticGrammar("g", model=model)
    assert len(g) == length


@pytest.mark.parametrize(
    ("model", "names"),
    [
        (None, []),
        (ModelID.ONE, ["name1", "name2"]),
    ],
    indirect=["model"],
)
def test_iter(model, names):
    """Verify iterating."""
    g = PydanticGrammar("g", model=model)
    assert list(iter(g)) == names


@pytest.mark.parametrize(
    ("model", "names"),
    [
        (None, []),
        (ModelID.ONE, ["name1", "name2"]),
    ],
    indirect=["model"],
)
def test_names(model, names):
    """Verify names getter."""
    g = PydanticGrammar("g", model=model)
    assert list(g.names) == names


# This is for using indirect in parametrize.
model_1 = model
model_2 = model


def assert_equal_types(field_1: FieldInfo, obj_2: FieldInfo | type) -> None:
    """Assert that 2 pydantic fields have the same types.

    Args:
        field_1: A field.
        obj_2: Another field or type.

    Raises:
        AssertionError: If the types are different.
    """
    type_2 = obj_2.annotation if isinstance(obj_2, FieldInfo) else obj_2
    assert field_1.annotation == type_2


@pytest.mark.parametrize(
    "model_1",
    [
        None,
        ModelID.ONE,
        ModelID.TWO,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "model_2",
    [
        None,
        ModelID.ONE,
        ModelID.TWO,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "exclude_names",
    [
        [],
        ["name1"],
        ["name2"],
    ],
)
def test_update(model_1, model_2, exclude_names):
    """Verify update."""
    g1 = PydanticGrammar("g", model=model_1)
    g1_names_before = g1.keys()
    g1_before = dict(g1)
    g1_required_names_before = set(g1.required_names)
    g1_defaults_before = g1.defaults.copy()
    g2 = PydanticGrammar("g", model=model_2)

    g1.update(g2, exclude_names=exclude_names)

    exclude_names = set() if exclude_names is None else set(exclude_names)

    assert set(g1) == g1_names_before | (g2.keys() - exclude_names)
    assert set(g1.required_names) == g1_required_names_before | (
        set(g2.required_names) - exclude_names
    )
    assert g1.defaults.keys() == g1_defaults_before.keys() | (
        g2.defaults.keys() - exclude_names
    )

    for name, type_ in g1.items():
        if name in g2.keys() - exclude_names:
            assert_equal_types(type_, g2[name])
            assert g1.defaults.get(name) == g2.defaults.get(name)
        else:
            assert_equal_types(type_, g1_before[name])
            assert g1.defaults.get(name) == g1_defaults_before.get(name)


@pytest.mark.parametrize(
    "model",
    [
        None,
        ModelID.ONE,
    ],
    indirect=True,
)
def test_clear(model):
    """Verify clear."""
    g = PydanticGrammar("g", model=model)
    g.clear()
    assert not g
    assert not g.required_names
    assert not g.defaults


@pytest.mark.parametrize(
    ("model", "repr_"),
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
            ModelID.ONE,
            """
Grammar name: g
   Required elements:
      name1:
         Type: <class 'int'>
   Optional elements:
      name2:
         Type: gemseo.core.grammars.pydantic_ndarray._NDArrayPydantic[typing.Any, numpy.dtype[int]]
         Default: [0]
""",  # noqa: E501
        ),
    ],
    indirect=["model"],
)
def test_repr(model, repr_):
    """Verify repr."""
    g = PydanticGrammar("g", model=model)
    assert repr(g) == repr_.strip()


@pytest.mark.parametrize(
    ("model", "data_sets"),
    [  # Empty grammar: everything validates.
        (None, ({"name": 0},)),
        (
            ModelID.TWO,
            (
                {"name1": 1},
                {"name1": 1, "name2": "bar"},
                {"name1": 1, "name2": 0},
            ),
        ),
    ],
    indirect=["model"],
)
def test_validate(model, data_sets):
    """Verify validate."""
    g = PydanticGrammar("g", model=model)
    for data in data_sets:
        g.validate(data)


def test_validate_with_rebuild(model1):
    """Verify validate with rebuild."""
    g = PydanticGrammar("g", model=model1)
    data = {"name1": 1}

    # Deleting.
    del g["name2"]
    g.validate(data)

    # Renaming.
    g.rename_element("name1", "name")
    g.validate({"name": 1})

    # Updating.
    class Model(BaseModel):
        foo: str

    g.update(PydanticGrammar("foo", model=Model))
    g.validate({"name": 1, "foo": ""})

    # Updating from names.
    g.update_from_names(["bar"])
    g.validate({"name": 1, "foo": "", "bar": array([])})

    # Updating from types.
    g.update_from_types({"baz": bool})
    g.validate({"name": 1, "foo": "", "bar": array([]), "baz": True})

    # Restricting.
    g.restrict_to(["name"])
    g.validate({"name": 1})


@pytest.mark.parametrize(
    ("raise_exception", "exception_tester"),
    [(True, pytest.raises), (False, do_not_raise)],
)
@pytest.mark.parametrize(
    ("data", "error_msg"),
    [
        (
            {},
            "Missing required names: name1.",
        ),
        (
            {"name1": 0.1, "name2": array([0])},
            """
1 validation error for Model
name1
  Input should be a valid integer [type=int_type, input_value=0.1, input_type=float]
""",  # noqa:E501
        ),
        (
            {"name1": 0, "name2": True},
            """
1 validation error for Model
name2
  Input should be an instance of ndarray [type=is_instance_of, input_value=True, input_type=bool]
""",  # noqa:E501
        ),
        (
            {"name1": 0, "name2": array([0.0])},
            """
1 validation error for Model
name2
  Value error, Input dtype should be <class 'int'>: got the dtype <class 'numpy.float64'>
""",  # noqa:E501
        ),
    ],
)
def test_validate_error(
    raise_exception, exception_tester, data, error_msg, model1, caplog
):
    """Verify that validate raises the expected errors."""
    g = PydanticGrammar("g", model=model1)

    with exception_tester(InvalidDataError, match=re.escape(error_msg.strip())):
        g.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"
    assert error_msg.strip() in caplog.text.strip()


@pytest.mark.parametrize(
    "model",
    [
        None,
        ModelID.ONE,
        ModelID.TWO,
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "names",
    [
        [],
        ["name1"],
        ["name2"],
    ],
)
def test_update_from_names(model, names):
    """Verify update from names."""
    g = PydanticGrammar("g", model=model)
    names_before = g.keys()
    g_before = dict(g)
    defaults_before = g.defaults.copy()
    required_names_before = set(g.required_names)

    g.update_from_names(names)

    assert set(g) == names_before | set(names)
    assert g.required_names == required_names_before | set(names)

    assert g.defaults == defaults_before

    for name, type_ in g.items():
        if name in names:
            assert_equal_types(type_, NDArrayPydantic)
        else:
            assert_equal_types(type_, g_before[name])


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        ({}, None),
        ({"name1": int}, int),
        ({"name1": float}, Union[int, float]),
        ({"name1": str}, Union[int, str]),
        ({"name1": bool}, Union[int, bool]),
        ({"name1": NDArrayPydantic}, Union[int, NDArrayPydantic]),
        ({"name1": dict}, Union[int, dict]),
        ({"name2": int}, Union[NDArrayPydantic[int], int]),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        ModelID.ONE,
    ],
    indirect=True,
)
def test_update_from_types_with_merge(model, data, expected_type):
    """Verify update_from_types."""

    def action(g: PydanticGrammar, data: Data) -> None:
        g.update_from_types(data, merge=True)

    _test_update_from(model, data, expected_type, action)


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        ({}, None),
        ({"name1": int}, int),
        ({"name1": float}, float),
        ({"name1": str}, str),
        ({"name1": bool}, bool),
        ({"name1": NDArrayPydantic}, NDArrayPydantic),
        ({"name1": dict}, dict),
    ],
)
def test_update_from_types_from_empty(data, expected_type):
    """Verify update_from_types."""

    def action(g: PydanticGrammar, data: Data) -> None:
        g.update_from_types(data, merge=True)

    _test_update_from(None, data, expected_type, action)


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        ({}, None),
        ({"name1": int}, int),
        ({"name1": float}, float),
        ({"name1": str}, str),
        ({"name1": bool}, bool),
        ({"name1": NDArrayPydantic}, NDArrayPydantic),
        ({"name1": dict}, dict),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        None,
        ModelID.ONE,
    ],
    indirect=True,
)
def test_update_from_types(model, data, expected_type):
    """Verify update_from_types."""

    def action(g: PydanticGrammar, data: Data) -> None:
        g.update_from_types(data)

    _test_update_from(model, data, expected_type, action)


@pytest.mark.parametrize(
    ("data", "expected_type"),
    [
        ({}, None),
        ({"name1": 0}, int),
        ({"name1": 0.0}, float),
        ({"name1": ""}, str),
        ({"name1": True}, bool),
        ({"name1": ndarray([0])}, _NDArrayPydantic),
        ({"name1": {"name2": 0}}, dict),
    ],
)
@pytest.mark.parametrize(
    "model",
    [
        None,
        ModelID.ONE,
    ],
    indirect=True,
)
def test_update_from_data(model, data, expected_type):
    """Verify update_from_data."""

    def action(g: PydanticGrammar, data: Data) -> None:
        g.update_from_data(data)

    _test_update_from(model, data, expected_type, action)


def _test_update_from(
    model: ModelType | None,
    data: Data,
    expected_type: type,
    action: Callable[[PydanticGrammar, Data], None],
):
    """Helper function for testing update_from_data."""
    g = PydanticGrammar("g", model=model)
    g_before = dict(g)
    required_names_before = set(g.required_names)
    defaults_before = g.defaults.copy()

    action(g, data)

    assert set(g) == g_before.keys() | set(data)
    assert g.required_names == required_names_before | set(data)
    assert g.defaults == defaults_before

    if not data:
        return

    for name, type_ in g.items():
        if name in data:
            assert_equal_types(type_, expected_type)
        else:
            assert_equal_types(type_, g_before[name])


def test_is_array_error():
    """Verify that is_array error."""
    g = PydanticGrammar("g")
    msg = "The name foo is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        g.is_array("foo")


def test_is_array(model3):
    """Verify is_array."""
    g = PydanticGrammar("g", model=model3)

    for name in ("an_int", "a_float", "a_bool"):
        assert not g.is_array(name)

    for name in (
        "an_int_ndarray",
        "a_float_ndarray",
        "a_bool_ndarray",
        "an_int_list",
        "a_float_list",
        "a_bool_list",
    ):
        assert g.is_array(name)

    for name in (
        "an_int_ndarray",
        "a_float_ndarray",
    ):
        assert g.is_array(name, numeric_only=True)

    for name in (
        "an_int_list",
        "a_float_list",
        "a_bool_ndarray",
        "a_bool_list",
    ):
        assert not g.is_array(name, numeric_only=True)


def test_restrict_to_error():
    """Verify that raises the expected error."""
    g = PydanticGrammar("g")
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
def test_restrict_to(names, model1):
    """Verify restrict_to."""
    g = PydanticGrammar("g", model=model1)
    required_names_before = set(g.required_names)
    defaults_before = g.defaults.copy()
    g.restrict_to(names)
    names = set(names)
    assert set(g) == names
    assert g.required_names == required_names_before & names
    assert g.defaults.keys() == defaults_before.keys() & names
    for name, value in g.defaults.items():
        assert value == defaults_before[name]


@pytest.mark.parametrize(
    ("model", "types"),
    [
        (None, []),
        (ModelID.ONE, [int, _NDArrayPydantic]),
        (ModelID.TWO, [int, None]),
    ],
    indirect=["model"],
)
def test_convert_to_simple_grammar(model, types):
    """Verify grammar conversion."""
    g1 = PydanticGrammar("g", model=model)
    g2 = g1.to_simple_grammar()
    assert set(g1) == set(g2)
    assert g1.required_names == g2.required_names
    assert isinstance(g2, SimpleGrammar)
    for type_, ref_type in zip(g2.values(), types):
        assert type_ == ref_type


def test_convert_to_simple_grammar_warnings(model2, caplog):
    """Verify grammar conversion warnings."""
    g1 = PydanticGrammar("g", model=model2)
    g1.to_simple_grammar()
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.messages[0] == (
        "Unsupported type 'typing.Union' in PydanticGrammar 'g' for field 'name2' in "
        "conversion to SimpleGrammar."
    )


@pytest.mark.parametrize(
    ("model", "names"),
    [
        (None, set()),
        (ModelID.ONE, {"name1"}),
    ],
    indirect=["model"],
)
def test_required_names(model, names):
    """Verify required_names."""
    g = PydanticGrammar("g", model=model)
    assert g.required_names == names


@pytest.mark.parametrize(
    "descriptions",
    [
        {},
        {"name1": "name1 description"},
        {"name1": "name1 description", "name2": "name2 description"},
    ],
)
def test_set_descriptions(descriptions, model2):
    """Verify setting descriptions."""
    g = PydanticGrammar("g", model=model2)
    g.set_descriptions(descriptions)

    for name in g:
        if name in descriptions:
            assert g.schema["properties"][name]["description"] == descriptions[name]
        else:
            assert "description" not in g.schema["properties"][name]


def test_set_descriptions_no_rebuild(model2):
    """Verify setting descriptions that does nothing."""
    g = PydanticGrammar("g", model=model2)
    g.set_descriptions({"dummy": "description"})
    assert "dummy" not in g


@pytest.mark.parametrize(
    ("model", "schema"),
    [
        (None, {"properties": {}, "title": "Model", "type": "object"}),
        (
            ModelID.TWO,
            {
                "properties": {
                    "name1": {"title": "Name1", "type": "integer"},
                    "name2": {
                        "anyOf": [{"type": "integer"}, {"type": "string"}],
                        "default": 0,
                        "title": "Name2",
                    },
                },
                "required": ["name1"],
                "title": "Model",
                "type": "object",
            },
        ),
    ],
    indirect=["model"],
)
def test_schema(model, schema):
    """Verify schema getter."""
    g = PydanticGrammar("g", model=model)
    assert g.schema == schema


def test_rename():
    """Verify rename."""
    g = PydanticGrammar("g")
    g.update_from_names(["name1", "name2"])
    g.defaults["name1"] = 0

    g.rename_element("name1", "n:name1")

    assert set(g.required_names) == {"n:name1", "name2"}
    assert set(g) == {"n:name1", "name2"}
    assert g.defaults.keys() == {"n:name1"}


def test_copy():
    """Verify copy."""
    g = PydanticGrammar("g")
    g.update_from_names(["name"])
    g.defaults["name"] = 1.0
    g_copy = g.copy()
    assert g_copy.defaults["name"] is g.defaults["name"]
    assert next(iter(g_copy.required_names)) is next(iter(g.required_names))


@pytest.mark.parametrize(
    ("model", "defaults"),
    [
        (ModelID.ONE, {"name2": array([0])}),
        (ModelID.TWO, {"name2": 0}),
        (ModelID.THREE, {}),
        (ModelID.FOUR, {}),
        (None, {}),
    ],
    indirect=["model"],
)
def test_defaults_instantiation(model, defaults):
    """Verify defaults after instantiation."""
    g = PydanticGrammar("g", model=model)
    assert g.defaults == defaults
