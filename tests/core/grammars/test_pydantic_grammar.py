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
from typing import Any

import pytest
from numpy import array
from numpy import dtype
from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic.fields import FieldInfo
from strenum import StrEnum

from gemseo.core.discipline.discipline_data import DisciplineData
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.pydantic import PydanticGrammar
from gemseo.core.grammars.pydantic import _create_model
from gemseo.utils.pydantic_ndarray import _NDArrayPydantic
from gemseo.utils.testing.helpers import do_not_raise

from .pydantic_models import get_model1
from .pydantic_models import get_model2
from .pydantic_models import get_model3

if TYPE_CHECKING:
    from _pytest.fixtures import SubRequest

    from gemseo.core.grammars.pydantic import ModelType


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
        (
            "Unsupported type '<class 'types.UnionType'>' in PydanticGrammar 'g' for "
            "field 'name2' in conversion to SimpleGrammar."
        ),
        # For python 3.9.
        (
            "Unsupported type 'typing.Union' in PydanticGrammar 'g' for "
            "field 'name2' in conversion to SimpleGrammar."
        ),
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
    grammar.descriptions.update(descriptions)

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
        grammar.descriptions.update({"dummy": "description"})


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


class MyEnum(StrEnum):
    a = auto()
    b = auto()


class DummyModel(BaseModel, validate_assignment=True):
    """Dummy Model."""

    dummy_var: int = Field(0, description="dummy variable")
    # The max_length argument is just for coverage purposes.
    dummy_enum: MyEnum = Field(MyEnum.a, max_length=1)
    dummy_enum_strict: MyEnum = Field(MyEnum.a, strict=True)

    @field_validator("dummy_var")
    @classmethod
    def dummy_var_must_be_positive(cls, v):
        if v <= 0:
            msg = "dummy_var must be positive"
            raise ValueError(msg)
        return v


def test_model_on_grammar_multi_instantiation() -> None:
    """Test that a new model is created each time a Pydantic grammar is instantiated."""

    grammar_1 = PydanticGrammar(name="g", model=DummyModel)
    grammar_2 = PydanticGrammar(name="g", model=DummyModel)
    assert id(grammar_1._PydanticGrammar__model) != id(
        grammar_2._PydanticGrammar__model
    )
    del grammar_1["dummy_var"]
    assert "dummy_var" in grammar_2


def test_copy_model():
    # With a model created internally.
    model = PydanticGrammar(name="g")._PydanticGrammar__model
    model_copy = _create_model(model)
    assert_model_equal(model, model_copy)

    # With a standard model.
    model_copy = _create_model(DummyModel)
    assert_model_equal(DummyModel, model_copy)

    obj = model_copy()
    assert obj.dummy_var == 0

    # Verify that the config dict and the validator are copied.
    with pytest.raises(ValueError, match="dummy_var must be positive"):
        obj.dummy_var = -1


def assert_model_equal(model, model_copy) -> None:
    """Assert that 2 models are identical grammar wise."""
    assert id(model_copy) != id(model)
    # assert model_copy.__name__ == model.__name__
    assert model_copy.__module__ == "gemseo.core.grammars.pydantic"

    for field_name, field_info in model.__pydantic_fields__.items():
        field_info_copy = model_copy.__pydantic_fields__[field_name]
        assert id(field_info) != id(field_info_copy)
        assert field_info.default == field_info_copy.default
        assert field_info.description == field_info_copy.description
        assert field_info.annotation == field_info_copy.annotation
        assert field_info.alias == field_info_copy.alias
        assert field_info.is_required() == field_info_copy.is_required()


def test_enum_validation():
    """Verify that an enum is not validated strictly."""
    grammar = PydanticGrammar(name="g", model=DummyModel)
    grammar.validate({"dummy_enum": "b", "dummy_enum_strict": "b"})

    # Ensure that the original model was not modified.
    assert len(DummyModel.__pydantic_fields__["dummy_enum"].metadata) == 1
    assert DummyModel.__pydantic_fields__["dummy_enum_strict"].metadata[0].strict


def test_create_model_field_info_not_shared(model1) -> None:
    """Verify that _create_model returns a model with independent FieldInfo objects."""
    copied = _create_model(model1)
    for field_name in model1.__pydantic_fields__:
        src_field = model1.__pydantic_fields__[field_name]
        cpy_field = copied.__pydantic_fields__[field_name]
        assert cpy_field is not src_field
        for src_item, cpy_item in zip(
            src_field.metadata, cpy_field.metadata, strict=False
        ):
            assert cpy_item is not src_item


def test_update_from_model_basic(model1) -> None:
    """Verify keys, required_names, and descriptions are populated correctly."""
    grammar = PydanticGrammar("g")
    grammar.update_from_model(model1)
    assert grammar.keys() == {"name1", "name2"}
    assert grammar.required_names == {"name1"}
    assert grammar.descriptions == {"name2": "Description of name2."}
    assert grammar.defaults == {"name2": [0]}


def test_update_from_model_no_merge(model1, model2) -> None:
    """Verify that calling without merge overwrites the existing field annotation."""
    grammar = PydanticGrammar("g")
    grammar.update_from_model(model1)
    grammar.update_from_model(model2, merge=False)
    # name2 should now have the int | str annotation from model2.
    assert grammar["name2"].annotation == int | str


def test_update_from_model_merge(model1, model2) -> None:
    """Verify that merge unions the field annotations."""
    grammar = PydanticGrammar("g")
    grammar.update_from_model(model1)
    grammar.update_from_model(model2, merge=True)
    assert grammar["name2"].annotation == _NDArrayPydantic[Any, dtype[int]] | int | str


def test_update_from_model_empty_model() -> None:
    """Verify that an empty model is a no-op."""

    class EmptyModel(BaseModel):
        pass

    grammar = PydanticGrammar("g")
    grammar.update_from_model(EmptyModel)
    assert not grammar


def test_update_from_model_incremental(model1) -> None:
    """Verify that two sequential calls accumulate fields correctly."""

    class ModelExtra(BaseModel):
        name3: float

    grammar = PydanticGrammar("g")
    grammar.update_from_model(model1)
    grammar.update_from_model(ModelExtra)
    assert grammar.keys() == {"name1", "name2", "name3"}
    assert grammar.required_names == {"name1", "name3"}
