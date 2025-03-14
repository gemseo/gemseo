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

import pickle
import re
from enum import Enum
from enum import auto
from typing import TYPE_CHECKING

import pytest
from numpy import array
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.utils.testing.helpers import do_not_raise

from .pydantic_models import get_model1
from .pydantic_models import get_model2
from .pydantic_models import get_model3

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest

    from gemseo.core.grammars.pydantic_grammar import ModelType


class ModelID(Enum):
    """Enumeration for selecting test models."""

    ONE = auto()
    TWO = auto()
    THREE = auto()
    FOUR = auto()


model1 = pytest.fixture(get_model1)
model2 = pytest.fixture(get_model2)
model3 = pytest.fixture(get_model3)


@pytest.fixture
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


def test_init_with_model(model1) -> None:
    """Verify initializing with a Pydantic model."""
    grammar = PydanticGrammar("g", model=model1)
    assert grammar
    assert grammar.keys() == {"name1", "name2"}
    assert grammar.required_names == {"name1"}
    assert grammar.descriptions == {"name2": "Description of name2."}


def test_getitem(model1) -> None:
    """Verify getting an item."""
    grammar = PydanticGrammar("g", model=model1)
    assert_equal_types(grammar["name1"], int)


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
    ("model", "data_sets"),
    [
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
def test_validate(model, data_sets) -> None:
    """Verify validate with dict and DisciplineData."""
    grammar = PydanticGrammar("g", model=model)
    for data in data_sets:
        grammar.validate(DisciplineData(data))
        grammar.validate(data)


def test_validate_with_rebuild(model1) -> None:
    """Verify validate with rebuild."""
    grammar = PydanticGrammar("g", model=model1)
    data = {"name1": 1}

    # Deleting.
    del grammar["name2"]
    grammar.validate(data)

    # Renaming.
    grammar.rename_element("name1", "name")
    grammar.validate({"name": 1})

    # Updating.
    class Model(BaseModel):
        foo: str

    grammar.update(PydanticGrammar("foo", model=Model))
    grammar.validate({"name": 1, "foo": ""})

    # Updating from names.
    grammar.update_from_names(["bar"])
    grammar.validate({"name": 1, "foo": "", "bar": array([])})

    # Updating from types.
    grammar.update_from_types({"baz": bool})
    grammar.validate({"name": 1, "foo": "", "bar": array([]), "baz": True})

    # Restricting.
    grammar.restrict_to(["name"])
    grammar.validate({"name": 1})


@pytest.mark.parametrize(
    ("raise_exception", "exception_tester"),
    [(True, pytest.raises), (False, do_not_raise)],
)
@pytest.mark.parametrize(
    ("data", "error_msg"),
    [
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
  Value error, The expected dtype is <class 'int'>: the actual dtype is <class 'numpy.float64'>
""",  # noqa:E501
        ),
    ],
)
def test_validate_error(
    raise_exception, exception_tester, data, error_msg, model1, caplog
) -> None:
    """Verify that validate raises the expected errors."""
    grammar = PydanticGrammar("g", model=model1)

    with exception_tester(InvalidDataError, match=re.escape(error_msg.strip())):
        grammar.validate(data, raise_exception=raise_exception)

    assert caplog.records[0].levelname == "ERROR"
    assert error_msg.strip() in caplog.text.strip()


def test_convert_to_simple_grammar_warnings(model2, caplog) -> None:
    """Verify grammar conversion warnings."""
    grammar = PydanticGrammar("g", model=model2)
    grammar.to_simple_grammar()
    assert caplog.records[0].levelname == "WARNING"
    assert caplog.messages[0] in (
        "Unsupported type '<class 'types.UnionType'>' in PydanticGrammar 'g' for "
        "field 'name2' in conversion to SimpleGrammar.",
        # For python 3.9.
        "Unsupported type 'typing.Union' in PydanticGrammar 'g' for "
        "field 'name2' in conversion to SimpleGrammar.",
    )


@pytest.mark.parametrize(
    "descriptions",
    [
        {},
        {"name1": "name1 description"},
        {"name1": "name1 description", "name2": "name2 description"},
    ],
)
def test_set_descriptions(descriptions, model2) -> None:
    """Verify setting descriptions."""
    grammar = PydanticGrammar("g", model=model2)
    grammar.set_descriptions(descriptions)

    descriptions_ = {"name2": "Original description for name 2"}
    descriptions_.update(descriptions)
    assert grammar.descriptions == descriptions_

    for name in grammar:
        if (description := descriptions_.get(name)) is not None:
            assert grammar.schema["properties"][name]["description"] == description
        else:
            assert "description" not in grammar.schema["properties"][name]


def test_set_descriptions_no_rebuild(model2) -> None:
    """Verify setting descriptions that does nothing."""
    grammar = PydanticGrammar("g", model=model2)
    with pytest.raises(
        KeyError, match=re.escape("The name 'dummy' is not in the grammar.")
    ):
        grammar.set_descriptions({"dummy": "description"})


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
                        "description": "Original description for name 2",
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
def test_schema(model, schema) -> None:
    """Verify schema getter."""
    grammar = PydanticGrammar("g", model=model)
    assert grammar.schema == schema


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
def test_defaults_from_model(model, defaults) -> None:
    """Verify defaults from model."""
    grammar = PydanticGrammar("g", model=model)
    assert grammar.defaults == defaults


class ModelForPickling(BaseModel):
    """A model that must be in the global namespace to be pickled."""

    x: int
    y: str = ""


def test_serialize() -> None:
    """Check that a grammar can be properly serialized."""

    grammar = PydanticGrammar("g", model=ModelForPickling)
    pickled_grammar = pickle.loads(pickle.dumps(grammar))

    assert pickled_grammar.name == grammar.name
    assert pickled_grammar.required_names == grammar.required_names
    assert pickled_grammar.to_namespaced == grammar.to_namespaced
    assert pickled_grammar.from_namespaced == grammar.from_namespaced
