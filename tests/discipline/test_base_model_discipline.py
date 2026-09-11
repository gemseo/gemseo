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

import pytest
from numpy import array
from numpy import ones
from numpy import zeros
from numpy.testing import assert_allclose
from pydantic import BaseModel

from gemseo.discipline.base_model_discipline import BaseModelDiscipline
from gemseo.util.pydantic_ndarray import NDArrayPydantic

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


def test_dry_run_leaves_model_pristine():
    """Check that the grammar inference does not modify the model and defaults."""
    model = ElementwiseModel()
    discipline = ElementwiseDiscipline(model)
    assert_allclose(model.x, ones(2))
    assert_allclose(model.y, zeros(2))
    assert_allclose(discipline.io.input_grammar.defaults["x"], ones(2))
    assert_allclose(discipline.io.output_grammar.defaults["y"], zeros(2))
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
    discipline = PassThroughDisc(PassThroughModel())
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
    """Check that in-place writes do not leak into the grammar defaults."""
    discipline = AccumulateDiscipline(ElementwiseModel())
    for _ in range(3):
        discipline.execute()
        discipline.cache.clear()
        assert_allclose(discipline.io.get_output_data()["y"], array([1.0, 0.0]))
        assert_allclose(discipline.io.input_grammar.defaults["y"], zeros(2))


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
