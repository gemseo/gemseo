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

import logging
import pickle
from operator import attrgetter

import pytest
from numpy import array
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.testing import assert_allclose
from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationError
from pydantic import field_validator
from pydantic import model_validator

from gemseo.discipline.base_model_discipline import BaseModelDiscipline
from gemseo.doe import CustomDOE_Settings
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.scenario import MDOScenario
from gemseo.space import DesignSpace
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.testing.helper import assert_exception

logger = logging.getLogger(__name__)


class Y(BaseModel):
    """A submodel for the y values of the Sellar Problem."""

    y_1: NDArrayPydantic

    y_2: NDArrayPydantic


class SellarModel(BaseModel):
    """A model for the Sellar Problem data."""

    x: NDArrayPydantic = ones(1)

    y: Y = Y(y_1=ones(1), y_2=ones(1))

    z: NDArrayPydantic = array([4.0, 3.0])

    dummy_field: str = "A dummy field."


class Sellar1Pydantic(BaseModelDiscipline):
    """A version of the Sellar1 discipline that uses a Pydantic model."""

    __use_input_dummy_field: bool
    """Whether to use the dummy field as an input."""

    __use_output_dummy_field: bool
    """Whether use the dummy field as an output."""

    def __init__(
        self,
        model: SellarModel,
        use_input_dummy_field: bool = False,
        use_output_dummy_field: bool = False,
    ):
        self.__use_input_dummy_field = use_input_dummy_field
        self.__use_output_dummy_field = use_output_dummy_field
        super().__init__(model)

    def _run_from_model(self, sellar_model: SellarModel) -> None:
        x = sellar_model.x
        z = sellar_model.z
        y_2 = sellar_model.y.y_2
        if self.__use_input_dummy_field:
            dummy_input = sellar_model.dummy_field
            logger.info("Dummy input: %s", dummy_input)

        sellar_model.y.y_1 = array([(z[0] ** 2 + z[1] + x[0] - 0.2 * y_2[0]) ** 0.5])
        if self.__use_output_dummy_field:
            sellar_model.dummy_field = "A different string."


class ElementwiseModel(BaseModel):
    """A model whose fields are written elementwise."""

    x: NDArrayPydantic = ones(2)

    y: NDArrayPydantic = zeros(2)


class ElementwiseDiscipline(BaseModelDiscipline):
    """A discipline writing an output array elementwise."""

    def _run_from_model(self, model: ElementwiseModel) -> None:
        model.y[0] = model.x[0] + 1.0


class AccumulateDiscipline(BaseModelDiscipline):
    """A discipline updating a field in-place from its own value."""

    def _run_from_model(self, model: ElementwiseModel) -> None:
        model.y[0] += model.x[0]


class FrozenElementwiseModel(ElementwiseModel, frozen=True):
    """A frozen model whose fields are written elementwise."""


class FrozenSubModel(BaseModel, frozen=True):
    """A frozen sub-model whose fields are written elementwise."""

    y: NDArrayPydantic = zeros(2)


class FrozenSubModelParentModel(BaseModel):
    """A model with a frozen sub-model."""

    x: NDArrayPydantic = ones(2)

    sub: FrozenSubModel = FrozenSubModel()


class NestedElementwiseDiscipline(BaseModelDiscipline):
    """A discipline writing an output array of a sub-model elementwise."""

    def _run_from_model(self, model: FrozenSubModelParentModel) -> None:
        model.sub.y[0] = model.x[0] + 1.0


class FrozenFieldModel(BaseModel):
    """A model whose output field is frozen."""

    x: NDArrayPydantic = ones(2)

    y: NDArrayPydantic = Field(default=zeros(2), frozen=True)


@pytest.mark.parametrize(
    ("discipline_class", "model_class", "output_path"),
    [
        pytest.param(
            NestedElementwiseDiscipline,
            FrozenSubModelParentModel,
            "sub.y",
            id="NestedElementwiseDiscipline-FrozenSubModelParentModel",
        ),
        pytest.param(
            ElementwiseDiscipline,
            FrozenFieldModel,
            "y",
            id="ElementwiseDiscipline-FrozenFieldModel",
        ),
    ],
)
def test_frozen_model(discipline_class, model_class, output_path):
    """Check that a frozen output changed in place is not written back."""
    model = model_class()
    discipline = discipline_class(model)
    assert_allclose(model.x, ones(2))
    assert_allclose(attrgetter(output_path)(model), zeros(2))
    assert set(discipline.io.output_grammar) == {output_path}
    assert_allclose(discipline.io.output_grammar.defaults[output_path], zeros(2))


@pytest.mark.parametrize(
    ("discipline_class", "model_class", "output_path"),
    [
        pytest.param(
            NestedElementwiseDiscipline,
            FrozenSubModelParentModel,
            "sub.y",
            id="NestedElementwiseDiscipline-FrozenSubModelParentModel",
        ),
        pytest.param(
            ElementwiseDiscipline,
            FrozenFieldModel,
            "y",
            id="ElementwiseDiscipline-FrozenFieldModel",
        ),
    ],
)
def test_frozen_model_execution(discipline_class, model_class, output_path):
    """Check the execution of a discipline changing a frozen output in place."""
    discipline = discipline_class(model_class())
    output_data = discipline.execute({"x": array([3.0, 1.0])})
    assert_allclose(output_data[output_path], array([4.0, 0.0]))


class FrozenSubModelInputParentModel(BaseModel):
    """A model with a frozen sub-model holding an input."""

    sub: FrozenElementwiseModel = FrozenElementwiseModel()

    y: NDArrayPydantic = zeros(2)


class SubModelInputDiscipline(BaseModelDiscipline):
    """A discipline reading an input of a sub-model."""

    def _run_from_model(self, model: FrozenSubModelInputParentModel) -> None:
        model.y[0] = model.sub.x[0] + 1.0


@pytest.mark.parametrize(
    ("discipline_class", "model_class"),
    [
        pytest.param(
            AccumulateDiscipline,
            FrozenFieldModel,
            id="AccumulateDiscipline-FrozenFieldModel",
        ),
        pytest.param(
            ElementwiseDiscipline,
            FrozenElementwiseModel,
            id="ElementwiseDiscipline-FrozenElementwiseModel",
        ),
        pytest.param(
            SubModelInputDiscipline,
            FrozenSubModelInputParentModel,
            id="SubModelInputDiscipline-FrozenSubModelInputParentModel",
        ),
    ],
)
def test_frozen_input(snapshot, discipline_class, model_class):
    """Check that a frozen input or a frozen model holding an input raises an error."""
    model = model_class()
    with assert_exception(ValueError, snapshot):
        discipline_class(model)
    # The model is left untouched.
    assert_allclose(model.y, zeros(2))


class TwoFrozenInputsModel(BaseModel):
    """A model whose two inputs are frozen."""

    x: float = Field(default=1.0, frozen=True)

    z: float = Field(default=2.0, frozen=True)

    y: float = 0.0


class TwoFrozenInputsDiscipline(BaseModelDiscipline):
    """A discipline reading two frozen inputs."""

    def _run_from_model(self, model: TwoFrozenInputsModel) -> None:
        model.y = model.x + model.z


def test_frozen_inputs_reported_together():
    """Check that all the frozen inputs are reported by a single error."""
    with pytest.raises(ValueError) as exc_info:
        TwoFrozenInputsDiscipline(TwoFrozenInputsModel())
    message = str(exc_info.value)
    assert (
        "the field x of the model TwoFrozenInputsModel holding the input x" in message
    )
    assert (
        "the field z of the model TwoFrozenInputsModel holding the input z" in message
    )


class FrozenListFieldModel(BaseModel):
    """A model whose output list field is frozen."""

    x: float = 1.0

    y: list[float] = Field(default_factory=lambda: [0.0], frozen=True)


class FrozenListFieldDiscipline(BaseModelDiscipline):
    """A discipline changing a frozen output list in place."""

    def _run_from_model(self, model: FrozenListFieldModel) -> None:
        model.y[0] = model.x * 2


def test_frozen_list_field_changed_in_place():
    """Check that a frozen list changed in place is an output not written back."""
    model = FrozenListFieldModel()
    discipline = FrozenListFieldDiscipline(model)
    assert model.y == [0.0]
    assert set(discipline.io.output_grammar) == {"y"}
    assert discipline.execute({"x": 3.0})["y"] == [6.0]


class FrozenFieldAssignmentDiscipline(BaseModelDiscipline):
    """A discipline assigning a frozen output list as a whole."""

    def _run_from_model(self, model: FrozenListFieldModel) -> None:
        model.y = [model.x]


class FrozenScalarOutputModel(BaseModel, frozen=True):
    """A frozen model whose output is a scalar."""

    x: float = 1.0

    y: float = 0.0


class FrozenModelAssignmentDiscipline(BaseModelDiscipline):
    """A discipline assigning an output of a frozen model."""

    def _run_from_model(self, model: FrozenScalarOutputModel) -> None:
        model.y = model.x


@pytest.mark.parametrize(
    ("discipline_class", "model_class", "error_type"),
    [
        (FrozenFieldAssignmentDiscipline, FrozenListFieldModel, "frozen_field"),
        (FrozenModelAssignmentDiscipline, FrozenScalarOutputModel, "frozen_instance"),
    ],
)
def test_frozen_output_assigned(discipline_class, model_class, error_type):
    """Check that assigning a frozen output fails during the dry-run."""
    with pytest.raises(ValidationError) as exc_info:
        discipline_class(model_class())
    assert exc_info.value.errors()[0]["type"] == error_type


class FrozenTopModel(BaseModel, frozen=True):
    """A frozen model holding a non-frozen sub-model."""

    sub: ElementwiseModel = ElementwiseModel()


class SubModelFieldDiscipline(BaseModelDiscipline):
    """A discipline writing an output array of a sub-model elementwise."""

    def _run_from_model(self, model: FrozenTopModel) -> None:
        model.sub.y[0] = model.sub.x[0] + 1.0


def test_frozen_model_with_writable_sub_model():
    """Check that a frozen model is accepted when only its sub-models are written."""
    model = FrozenTopModel()
    discipline = SubModelFieldDiscipline(model)
    assert_allclose(model.sub.y, array([2.0, 0.0]))
    output_data = discipline.execute({"sub.x": array([3.0, 1.0])})
    assert_allclose(output_data["sub.y"], array([4.0, 0.0]))


class FrozenScalarModel(BaseModel, frozen=True):
    """A frozen sub-model assigned as a whole."""

    a: float = 0.0


class FrozenScalarParentModel(BaseModel):
    """A model whose frozen sub-model is assigned as a whole."""

    x: float = 1.0

    sub: FrozenScalarModel = FrozenScalarModel()


class FrozenSubModelAssignmentDiscipline(BaseModelDiscipline):
    """A discipline assigning a whole frozen sub-model then reading it."""

    def _run_from_model(self, model: FrozenScalarParentModel) -> None:
        model.sub = FrozenScalarModel(a=2.0 * model.x)
        model.x = model.sub.a


def test_frozen_sub_model_assigned_as_a_whole():
    """Check that a frozen sub-model assigned as a whole is written back as a whole."""
    model = FrozenScalarParentModel()
    FrozenSubModelAssignmentDiscipline(model)
    assert type(model.sub) is FrozenScalarModel
    assert model.sub.a == 2.0


def test_frozen_sub_model_assigned_as_a_whole_execution():
    """Check the execution of a discipline assigning a frozen sub-model as a whole."""
    discipline = FrozenSubModelAssignmentDiscipline(FrozenScalarParentModel())
    output_data = discipline.execute({"x": 3.0})
    assert output_data["sub.a"] == 6.0


class DoublingModel(BaseModel, validate_assignment=True):
    """A model validating the assignments and doubling `y` at validation."""

    x: float = 1.0

    y: float = 0.0

    @field_validator("y")
    @classmethod
    def double(cls, value: float) -> float:
        """Double a value of `y`.

        Args:
            value: The value of `y`.

        Returns:
            The doubled value.
        """
        return 2.0 * value


class DoublingDiscipline(BaseModelDiscipline):
    """A discipline assigning `x` to `y`."""

    def _run_from_model(self, model: DoublingModel) -> None:
        model.y = model.x


def test_write_back_does_not_validate_again(caplog):
    """Check that the write-back does not run the field validators again.

    The models defining field validators shall be reported at construction.
    """
    model = DoublingModel()
    with caplog.at_level(logging.WARNING):
        discipline = DoublingDiscipline(model)
    assert model.y == 2.0
    assert discipline.io.output_grammar.defaults["y"] == 2.0
    assert discipline.execute()["y"] == 2.0
    assert len(caplog.records) == 1
    assert "DoublingModel" in caplog.messages[0]
    assert "not rerun" in caplog.messages[0]


def test_no_warning_without_field_validators(caplog):
    """Check that no warning is logged for a model without field validators."""
    with caplog.at_level(logging.WARNING):
        ElementwiseDiscipline(ElementwiseModel())
    assert not [
        record for record in caplog.records if record.levelno >= logging.WARNING
    ]


class OrderedModel(BaseModel, validate_assignment=True):
    """A model validating the assignments and requiring `a` not greater than `b`."""

    a: float = 0.0

    b: float = 1.0

    @model_validator(mode="after")
    def check_order(self) -> OrderedModel:
        """Check that `a` is not greater than `b`.

        Returns:
            The model.

        Raises:
            ValueError: When `a` is greater than `b`.
        """
        if self.a > self.b:
            msg = "a must not be greater than b."
            raise ValueError(msg)
        return self


class OrderedDiscipline(BaseModelDiscipline):
    """A discipline increasing `b` then `a`."""

    def _run_from_model(self, model: OrderedModel) -> None:
        model.b = 10.0
        model.a = 5.0


def test_write_back_skips_intermediate_states():
    """Check that the write-back does not validate the intermediate model states."""
    model = OrderedModel()
    OrderedDiscipline(model)
    assert model.a == 5.0
    assert model.b == 10.0
    assert model.model_fields_set == {"a", "b"}


class ScalarModel(BaseModel):
    """A sub-model holding a scalar."""

    a: float = 1.0


class ModelListModel(BaseModel):
    """A model holding a sub-model and a list of sub-models."""

    z: ScalarModel = ScalarModel()

    zs: list[ScalarModel] = []


class ModelListDiscipline(BaseModelDiscipline):
    """A discipline putting a sub-model of the model into a list."""

    def _run_from_model(self, model: ModelListModel) -> None:
        model.zs = [model.z]


def test_write_back_untracks_models_in_containers():
    """Check that the sub-models in a container written back are not tracked."""
    model = ModelListModel()
    discipline = ModelListDiscipline(model)
    for sub_model in (model.zs[0], discipline.io.output_grammar.defaults["zs"][0]):
        assert type(sub_model) is ScalarModel
        assert "__tracking_state__" not in sub_model.__dict__


def test_write_back_of_models_in_containers_is_picklable():
    """Check that a model holding a container of sub-models written back pickles."""
    model = ModelListModel()
    discipline = ModelListDiscipline(model)
    assert type(pickle.loads(pickle.dumps(model)).zs[0]) is ScalarModel
    discipline = pickle.loads(pickle.dumps(discipline))
    assert discipline.io.output_grammar.defaults["zs"][0].a == 1.0


class ExtraModel(BaseModel, extra="allow"):
    """A sub-model allowing extra fields."""

    v: float = 0.0


class ExtraParentModel(BaseModel):
    """A model holding a sub-model allowing extra fields."""

    x: float = 1.0

    y: ExtraModel = ExtraModel()


class ExtraDiscipline(BaseModelDiscipline):
    """A discipline assigning a sub-model with an extra field."""

    def _run_from_model(self, model: ExtraParentModel) -> None:
        model.y = ExtraModel(v=model.x, extra_info=3)


def test_write_back_keeps_extra_fields():
    """Check that the extra fields of a sub-model written back are kept."""
    model = ExtraParentModel()
    ExtraDiscipline(model)
    assert model.y.model_extra == {"extra_info": 3}
    assert model.y.model_fields_set == {"v", "extra_info"}
    assert model.y.v == 1.0


class PostInitModel(BaseModel):
    """A sub-model incrementing its field `v` after its initialization."""

    v: float = 0.0

    def model_post_init(self, context: object) -> None:
        """Increment the field `v`.

        Args:
            context: The validation context.
        """
        self.v += 1.0


class PostInitParentModel(BaseModel):
    """A model holding a sub-model with a `model_post_init`."""

    x: float = 1.0

    y: PostInitModel = PostInitModel()


class PostInitDiscipline(BaseModelDiscipline):
    """A discipline assigning a sub-model with a `model_post_init`."""

    def _run_from_model(self, model: PostInitParentModel) -> None:
        model.y = PostInitModel(v=model.x)


def test_write_back_does_not_post_init_again():
    """Check that the write-back does not call `model_post_init` again."""
    model = PostInitParentModel()
    discipline = PostInitDiscipline(model)
    assert model.y.v == 2.0
    assert discipline.io.output_grammar.defaults["y.v"] == 2.0


def test_dry_run_updates_model():
    """Check that the grammar-inference dry-run writes its outputs into the model."""
    model = ElementwiseModel()
    discipline = ElementwiseDiscipline(model)
    assert_allclose(model.x, ones(2))
    assert_allclose(model.y, array([2.0, 0.0]))
    assert_allclose(discipline.io.input_grammar.defaults["x"], ones(2))
    assert_allclose(discipline.io.output_grammar.defaults["y"], array([2.0, 0.0]))
    output_data = discipline.execute()
    assert_allclose(output_data["y"], array([2.0, 0.0]))


class TotalModel(BaseModel):
    """A model with a method computing from its fields."""

    a: float = 1.0

    b: float = 2.0

    out: float = 0.0

    def compute_total(self) -> float:
        return self.a + self.b


class MethodDiscipline(BaseModelDiscipline):
    """A discipline computing its output through a model method."""

    def _run_from_model(self, model: TotalModel) -> None:
        model.out = model.compute_total()


class InnerModel(BaseModel):
    """A sub-model assigned wholesale by a discipline."""

    a: float = 1.0

    b: float = 2.0


class OuterModel(BaseModel):
    """A model whose sub-model is assigned wholesale."""

    x: float = 1.0

    inner: InnerModel = InnerModel()


class SubmodelAssignmentDiscipline(BaseModelDiscipline):
    """A discipline assigning a whole sub-model."""

    def _run_from_model(self, model: OuterModel) -> None:
        model.inner = InnerModel(a=2.0 * model.x, b=model.x + 1.0)


def test_whole_submodel_assignment_discipline():
    """Check the flattened grammars and execution for sub-model assignment."""
    discipline = SubmodelAssignmentDiscipline(OuterModel())
    assert set(discipline.io.input_grammar) == {"x"}
    assert set(discipline.io.output_grammar) == {"inner.a", "inner.b"}
    output_data = discipline.execute({"x": 3.0})
    assert output_data["inner.a"] == 6.0
    assert output_data["inner.b"] == 4.0


class SubModel(BaseModel):
    """A sub-model forwarded from one field to another."""

    a: float = 1.0

    b: float = 1.0


class PassThroughModel(BaseModel):
    """A model whose input sub-model is forwarded to an output field."""

    inp: SubModel = SubModel()

    out: SubModel = SubModel()

    z: float = 0.0


class PassThroughDisc(BaseModelDiscipline):
    """A discipline forwarding a whole sub-model to another field."""

    def _run_from_model(self, model: PassThroughModel) -> None:
        model.out = model.inp
        model.z = model.out.a + model.inp.b


def test_submodel_forwarded_to_another_field():
    """Check that forwarding a sub-model keeps the source field's reads as inputs."""
    model = PassThroughModel()
    discipline = PassThroughDisc(model)
    # The sub-model assigned as a whole is written back as a plain copy.
    assert type(model.out) is SubModel
    assert model.out is not model.inp
    assert set(discipline.io.input_grammar) == {"inp.a", "inp.b"}
    assert set(discipline.io.output_grammar) == {"out.a", "out.b", "z"}
    output_data = discipline.execute({"inp.a": 10.0, "inp.b": 100.0})
    assert output_data["z"] == 110.0


def test_method_based_discipline():
    """Check that fields accessed inside model methods are inferred as I/O."""
    discipline = MethodDiscipline(TotalModel())
    assert set(discipline.io.input_grammar) == {"a", "b"}
    assert set(discipline.io.output_grammar) == {"out"}
    assert discipline.io.input_grammar.name == "MethodDiscipline_input"
    assert discipline.io.output_grammar.name == "MethodDiscipline_output"
    output_data = discipline.execute({"a": 10.0})
    assert output_data["out"] == 12.0


def test_pickle_round_trip():
    """Check that a discipline with inferred grammars can be pickled."""
    discipline = Sellar1Pydantic(SellarModel())
    other_discipline = Sellar1Pydantic(SellarModel())
    unpickled = pickle.loads(pickle.dumps(discipline))
    pickle.dumps(other_discipline)
    assert_allclose(unpickled.execute()["y.y_1"], discipline.execute()["y.y_1"])


def test_execute_is_idempotent():
    """Check that executions do not accumulate into the model or the defaults."""
    model = ElementwiseModel()
    discipline = AccumulateDiscipline(model)
    for _ in range(3):
        output_data = discipline.execute()
        discipline.cache.clear()
        assert_allclose(output_data["y"], array([1.0, 0.0]))
        assert_allclose(discipline.io.input_grammar.defaults["y"], zeros(2))
        assert_allclose(model.y, array([1.0, 0.0]))


def test_input_defaults_precede_write_back():
    """Check that the input defaults are the model values before the write-back.

    A second discipline built on the same model sees the written-back values.
    """
    model = ElementwiseModel()
    discipline = AccumulateDiscipline(model)
    other_discipline = AccumulateDiscipline(model)
    assert_allclose(discipline.io.input_grammar.defaults["y"], zeros(2))
    assert_allclose(other_discipline.io.input_grammar.defaults["y"], array([1.0, 0.0]))
    assert_allclose(model.y, array([2.0, 0.0]))
    assert_allclose(discipline.execute()["y"], array([1.0, 0.0]))
    assert_allclose(other_discipline.execute()["y"], array([2.0, 0.0]))


def test_execution_does_not_mutate_model_in_place():
    """Check that an execution writing an output in-place leaves the model untouched."""
    model = ElementwiseModel()
    discipline = ElementwiseDiscipline(model)
    output_data = discipline.execute({"x": array([5.0, 0.0])})
    assert_allclose(output_data["y"], array([6.0, 0.0]))
    assert_allclose(model.y, array([2.0, 0.0]))
    assert_allclose(model.x, ones(2))


@pytest.mark.parametrize("use_dummy_input", [True, False])
@pytest.mark.parametrize("use_dummy_output", [True, False])
def test_sellar_1_pydantic(use_dummy_input, use_dummy_output):
    """Test the BaseModelDiscipline's automatic grammar generation."""
    discipline = Sellar1Pydantic(
        SellarModel(),
        use_input_dummy_field=use_dummy_input,
        use_output_dummy_field=use_dummy_output,
    )
    assert "x" in discipline.input_grammar.names
    assert "y.y_2" in discipline.input_grammar.names
    assert "z" in discipline.input_grammar.names
    assert "y.y_1" in discipline.output_grammar.names

    if use_dummy_input:
        assert "dummy_field" in discipline.input_grammar.names
    if use_dummy_output:
        assert "dummy_field" in discipline.output_grammar.names


@pytest.mark.parametrize("use_dummy_input", [True, False])
@pytest.mark.parametrize("use_dummy_output", [True, False])
@pytest.mark.parametrize(
    "reference_data",
    [
        {
            "input": {"x": ones(1), "y.y_2": ones(1), "z": array([4.0, 3.0])},
            "output": {"y.y_1": array([4.449719092257398])},
        },
        {
            "input": {
                "x": zeros(1),
                "y.y_2": array([3.755277766925886]),
                "z": array([1.977638883463326, 0.0]),
            },
            "output": {"y.y_1": array([1.777638883462956])},
        },
    ],
)
def test_sellar_1_pydantic_execution(use_dummy_input, use_dummy_output, reference_data):
    """Test the execution of the BaseModelDiscipline."""
    discipline = Sellar1Pydantic(
        SellarModel(),
        use_input_dummy_field=use_dummy_input,
        use_output_dummy_field=use_dummy_output,
    )
    # The parametrize data is shared across the parameter matrix:
    # do not modify it in place.
    input_data = dict(reference_data["input"])
    if use_dummy_input:
        input_data["dummy_field"] = "Test input"
    discipline.execute(input_data)
    output_data = discipline.get_output_data()
    if use_dummy_output:
        assert output_data["dummy_field"] == "A different string."
    assert_allclose(output_data["y.y_1"], reference_data["output"]["y.y_1"])


class IntermediateModel(BaseModel):
    """A model with a field computed and then re-used."""

    x: NDArrayPydantic = ones(1)

    a: NDArrayPydantic = ones(1)

    b: NDArrayPydantic = zeros(1)

    y: NDArrayPydantic = zeros(1)

    z: NDArrayPydantic = zeros(1)


class IntermediateDiscipline(BaseModelDiscipline):
    """A discipline re-using a computed field to compute another one."""

    def _run_from_model(self, model: IntermediateModel) -> None:
        model.y = model.a * model.x**2 + model.b
        model.z = model.y + 3.0


def test_intermediate_value_is_not_an_input():
    """Check that a computed field re-used to compute another one is not an input."""
    discipline = IntermediateDiscipline(IntermediateModel())
    assert set(discipline.io.input_grammar) == {"x", "a", "b"}
    assert set(discipline.io.output_grammar) == {"y", "z"}
    output_data = discipline.execute({
        "x": array([3.0]),
        "a": array([2.0]),
        "b": array([1.0]),
    })
    assert_allclose(output_data["y"], array([19.0]))
    assert_allclose(output_data["z"], array([22.0]))


def test_dry_run_writeback_is_plain_and_decoupled():
    """Check that the dry-run write-back holds plain, defaults-decoupled values."""
    model = SellarModel()
    discipline = Sellar1Pydantic(model)
    assert type(model.y.y_1) is ndarray
    assert_allclose(model.y.y_1, array([4.449719092257398]))
    model.y.y_1[0] = 123.0
    assert_allclose(
        discipline.io.output_grammar.defaults["y.y_1"], array([4.449719092257398])
    )


class ExampleModel(BaseModel):
    """A model shared by `DisciplineA` and `DisciplineB`."""

    a: float = 1.0
    b: float = Field(
        default=-99999999.99999,
        description="This field is initialized with a dummy value to simulate a case in"
        " which the user cannot easily provide the value of an intermediate"
        " variable of the process.",
    )
    c: float = 3.0
    d: float = 4.0
    e: float = Field(
        default=-99999999.99999,
        description="This field is initialized with a dummy value to simulate a case in"
        " which the user cannot easily provide a sensible default.",
    )


class DisciplineA(BaseModelDiscipline):
    """A discipline computing `b` and `c` from `a`."""

    def _run_from_model(self, model: ExampleModel) -> None:
        model.b = model.a + 0.5
        model.c = model.a * 2.0


class DisciplineB(BaseModelDiscipline):
    """A discipline computing `d` from `b` and `c`, and `e` for a sensible `b` only.

    `e` is written only when `b` is strictly between 1 and 2,
    which excludes the dummy value of `ExampleModel.b`:
    `e` is an output of this discipline only when its dry-run sees the value of `b`
    computed by a `DisciplineA` built beforehand on the same model.
    """

    def _run_from_model(self, model: ExampleModel) -> None:
        model.d = model.b + model.c
        if 1.0 < model.b < 2.0:
            model.e = (model.b + model.c) / 2.0


def test_model_update():
    """Test that a model is updated by the dry run at instantiation."""
    model = ExampleModel()
    DisciplineA(model=model)
    assert model.b == 1.5
    assert model.c == 2.0


def test_execution_does_not_update_model():
    """Test that executing a discipline does not update the shared model."""
    model = ExampleModel()
    discipline = DisciplineA(model=model)
    output_data = discipline.execute({"a": 5.0})
    assert output_data["b"] == 5.5
    assert output_data["c"] == 10.0
    assert model.b == 1.5
    assert model.c == 2.0


def test_disciplines_sharing_model():
    """Test that a model shared across disciplines is updated by the dry run."""
    model = ExampleModel()
    DisciplineA(model=model)
    discipline_b = DisciplineB(model=model)
    assert model.b == 1.5
    assert model.c == 2.0
    assert "e" in discipline_b.output_grammar.names
    assert model.d == 3.5
    assert model.e == 1.75


def test_discipline_scenario_sharing_model():
    """Test the scenario execution of disciplines sharing a model.

    This is a regression test for a model shared by several disciplines
    working inside a formulation: the scenario execution shall use its own
    copies of the model and shall not modify the shared model, which shall
    still hold the values computed by the dry-runs at construction.
    """
    model = ExampleModel()
    disciplines = [DisciplineA(model=model), DisciplineB(model=model)]
    design_space = DesignSpace()
    design_space.add_variable("a", lower_bound=1.0, upper_bound=10.0)
    scenario = MDOScenario(
        disciplines, design_space, formulation_settings=DisciplinaryOpt_Settings()
    )
    scenario.add_objective("c")
    scenario.execute(CustomDOE_Settings(samples=array([[4.0]])))
    # DisciplineA computes c = 2 * a and the sampled a is 4.0.
    assert scenario.optimization_result.f_opt == 8.0
    # The model keeps the values from the dry-run.
    assert model.b == 1.5
    assert model.c == 2.0
    assert model.d == 3.5
    assert model.e == 1.75
