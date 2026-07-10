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
from enum import Enum
from enum import auto
from platform import python_version
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import array
from numpy import dtype
from numpy import ndarray
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
from gemseo.utils.testing.helpers import assert_exception

from .pydantic_models import get_model1
from .pydantic_models import get_model2
from .pydantic_models import get_model3
from .pydantic_models import get_model4

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
model4 = pytest.fixture(get_model4)


@pytest.fixture
def model(
    request: SubRequest,
    model1: ModelType,
    model2: ModelType,
    model3: ModelType,
    model4: ModelType,
) -> ModelType | None:
    """Return a pydantic model.

    This fixture can be optionally parametrized.
    """
    if request.param is None:
        return None

    return {
        ModelID.ONE: model1,
        ModelID.TWO: model2,
        ModelID.THREE: model3,
        ModelID.FOUR: model4,
    }[request.param]


def test_init_with_model(model1) -> None:
    """Verify initializing with a Pydantic model."""
    grammar = PydanticGrammar("g", model=model1)
    assert grammar
    assert grammar.keys() == {"name1", "name2"}
    assert grammar.required_names == {"name1"}
    assert grammar.descriptions == {"name2": "Description of name2."}


def test_create_model_from_base_model() -> None:
    """Verify copying the bare ``BaseModel`` as a non-internal model."""
    copied = _create_model(BaseModel)
    assert issubclass(copied, BaseModel)
    assert not copied.__pydantic_fields__


def _build_model_with_separator() -> ModelType:
    """Build a Pydantic model whose field name contains the namespace separator."""
    from pydantic import create_model

    return create_model("BadModel", **{"x:y": (int, ...)})


def test_init_rejects_model_with_namespace_separator(snapshot) -> None:
    """Verify that the constructor rejects model field names with the separator."""
    with assert_exception(ValueError, snapshot):
        PydanticGrammar("g", model=_build_model_with_separator())


def test_update_from_model_rejects_namespace_separator(snapshot) -> None:
    """Verify that `update_from_model` rejects field names with the separator."""
    grammar = PydanticGrammar("g")
    with assert_exception(ValueError, snapshot):
        grammar.update_from_model(_build_model_with_separator())


def test_getitem(model1) -> None:
    """Verify getting an item."""
    grammar = PydanticGrammar("g", model=model1)
    assert_equal_types(grammar["name1"], int)


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


@pytest.mark.parametrize("raise_exception", [True, False])
@pytest.mark.parametrize(
    "data",
    [
        {"name1": 0.1, "name2": array([0])},
        {"name1": 0, "name2": True},
        {"name1": 0, "name2": array([0.0])},
    ],
)
def test_validate_error(raise_exception, data, model1, caplog, snapshot) -> None:
    """Verify that validate raises the expected errors."""
    grammar = PydanticGrammar("g", model=model1)

    if raise_exception:
        with assert_exception(InvalidDataError, snapshot):
            grammar.validate(data)
    else:
        grammar.validate(data, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"


def test_convert_to_simple_grammar_warnings(model2, caplog) -> None:
    """Verify grammar conversion warnings."""
    grammar = PydanticGrammar("g", model=model2)
    grammar.to_simple_grammar()
    assert caplog.records[0].levelname == "WARNING"
    union_type = "typing.Union" if python_version() >= "3.14" else "types.UnionType"
    assert caplog.messages[0] == (
        f"Unsupported type '<class '{union_type}'>' in PydanticGrammar 'g' for "
        "field 'name2' in conversion to SimpleGrammar."
    )


def test_convert_to_simple_grammar_ndarray_field(model1) -> None:
    """Verify that an NDArrayPydantic field is converted to ndarray."""
    grammar = PydanticGrammar("g", model=model1)
    simple_grammar = grammar.to_simple_grammar()
    assert simple_grammar["name2"] is ndarray


def test_convert_to_simple_grammar_warning_non_generic(caplog) -> None:
    """Verify the conversion warning shows the type for a non-generic annotation."""

    class CustomType:
        pass

    class Model(BaseModel):
        name1: CustomType

        model_config = {"arbitrary_types_allowed": True}

    grammar = PydanticGrammar("g", model=Model)
    grammar.to_simple_grammar()
    assert caplog.records[0].levelname == "WARNING"
    assert "CustomType" in caplog.messages[0]


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


def test_set_descriptions_no_rebuild(model2, snapshot) -> None:
    """Verify setting descriptions that does nothing."""
    grammar = PydanticGrammar("g", model=model2)
    with assert_exception(KeyError, snapshot):
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

    pickled_grammar.validate({"x": 1})
    with pytest.raises(InvalidDataError):
        pickled_grammar.validate({"x": "not-an-int"})


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
    del grammar_1["dummy_var"]
    assert "dummy_var" in grammar_2


def test_copy_model(snapshot):
    model_copy = _create_model(DummyModel)
    assert_model_equal(DummyModel, model_copy)

    obj = model_copy()
    assert obj.dummy_var == 0

    # Verify that the config dict and the validator are copied.
    with assert_exception(ValueError, snapshot):
        obj.dummy_var = -1


def assert_model_equal(model, model_copy) -> None:
    """Assert that 2 models are identical grammar wise."""
    assert id(model_copy) != id(model)
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


def test_update_error(snapshot) -> None:
    """Verify updating from another grammar type raises a clear error."""
    grammar = PydanticGrammar("g")
    with assert_exception(TypeError, snapshot):
        grammar.update(True)


def test_update_from_types_with_catch_all_type() -> None:
    """Verify that the None type accepts any value, as in the other grammars."""
    grammar = PydanticGrammar("g")
    grammar.update_from_types({"x": None})
    grammar.validate({"x": 1})
    grammar.validate({"x": "a"})
    grammar.validate({"x": None})


def test_update_keeps_model_defaults(model1) -> None:
    """Verify that updating from another grammar keeps the model-level defaults.

    Data missing an element that is optional in the source grammar must remain
    valid after the update.
    """
    grammar = PydanticGrammar("g")
    grammar.update(PydanticGrammar("g1", model=model1))
    assert grammar.required_names == {"name1"}
    assert grammar.defaults == {"name2": [0]}
    grammar.validate({"name1": 1})


def test_schema_is_cached() -> None:
    """Verify that `PydanticGrammar.schema` reuses the cached dict between reads."""
    grammar = PydanticGrammar("g")
    grammar.update_from_types({"x": int})
    first = grammar.schema
    assert grammar.schema is first
    grammar.update_from_types({"y": str})
    refreshed = grammar.schema
    assert refreshed is not first
    assert "y" in refreshed["properties"]
