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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import eye
from numpy import inf
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.random import default_rng
from numpy.testing import assert_almost_equal
from scipy.linalg import solve

from gemseo import create_discipline
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import Discipline
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.mda.base_mda_solver_settings import BaseMDASolverSettings
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.problems.mdo.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.mdo.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import get_initial_data
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.seeder import SEED
from gemseo.utils.testing.helpers import concretize_classes

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

DIRNAME = os.path.dirname(__file__)


@pytest.fixture
def sellar_mda(sellar_disciplines):
    return MDAGaussSeidel(sellar_disciplines)


@pytest.fixture(scope="module")
def sellar_inputs():
    """Build dictionary with initial solution."""
    return get_initial_data()


@pytest.fixture
def base_mda_solver(sellar_disciplines) -> BaseMDASolver:
    """A concretized BaseMDASolver instance with the Sellar's disciplines."""
    with concretize_classes(BaseMDASolver):
        BaseMDASolver.Settings = BaseMDASolverSettings
        return BaseMDASolver(sellar_disciplines)


def test_reset(sellar_mda, sellar_inputs) -> None:
    """Test that the MDA successfully resets its disciplines after their executions."""
    disciplines = sellar_mda.disciplines
    for discipline in disciplines:
        discipline.execute(sellar_inputs)
        assert discipline.execution_status.value == ExecutionStatus.Status.DONE


def test_input_couplings() -> None:
    with concretize_classes(BaseMDASolver):
        BaseMDASolver.Settings = BaseMDASolverSettings
        mda = BaseMDASolver([Sellar1()])
        mda._set_resolved_variables([])

    assert len(mda.get_current_resolved_variables_vector()) == 0

    with concretize_classes(BaseMDASolver):
        BaseMDASolver.Settings = BaseMDASolverSettings
        mda = BaseMDASolver(
            create_discipline([
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
                "SobieskiStructure",
            ])
        )
        mda._compute_input_coupling_names()
        sorted_c = ["y_12", "y_21", "y_23", "y_31", "y_32"]
        assert mda._input_couplings == sorted_c


def test_resolved_couplings() -> None:
    """Tests the resolved coupling names."""
    disciplines = create_disciplines_from_desc(
        [
            ("A", ["x"], ["a"]),
            ("B", ["a", "y"], ["b"]),
            ("C", ["b"], ["y"]),
        ],
    )

    mda = MDAJacobi(disciplines)
    assert set(mda._resolved_variable_names) == set(mda._input_couplings)

    mda = MDAGaussSeidel(disciplines)
    assert set(mda._resolved_variable_names) == set(
        mda.coupling_structure.strong_couplings
    )

    with pytest.raises(AttributeError):
        mda._resolved_variable_names = "a"


def test_jacobian(sellar_mda, sellar_inputs) -> None:
    """Check the Jacobian computation."""
    sellar_mda.settings.use_lu_fact = True
    sellar_mda.matrix_type = JacobianAssembly.JacobianType.LINEAR_OPERATOR
    with pytest.raises(
        ValueError, match="Unsupported LU factorization for LinearOperators"
    ):
        sellar_mda.linearize(
            sellar_inputs,
            compute_all_jacobians=True,
        )

    sellar_mda.settings.use_lu_fact = False
    sellar_mda.linearize(sellar_inputs)
    assert sellar_mda.jac == {}

    sellar_mda._differentiated_input_names = []
    sellar_mda._differentiated_output_names = []

    sellar_mda.linearize(sellar_inputs)


def test_expected_workflow(sellar_mda) -> None:
    """"""
    expected = (
        "{MDAGaussSeidel(DONE), [Sellar1(DONE), Sellar2(DONE), SellarSystem(DONE)]}"
    )
    assert str(sellar_mda.get_process_flow().get_execution_flow()) == expected


def test_warm_start() -> None:
    """Check that the warm start does not fail even at first execution."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda_sellar = MDAGaussSeidel(disciplines)
    mda_sellar.settings.warm_start = True
    mda_sellar.execute()


def test_weak_strong_coupling_mda_jac() -> None:
    """Tests a particular coupling structure jacobian."""
    disciplines = analytic_disciplines_from_desc((
        {"y1": "x"},
        {"c1": "y1+x+0.2*c2"},
        {"c2": "y1+x+1.-0.3*c1"},
        {"obj": "x+c1+c2"},
    ))
    mda = MDAGaussSeidel(disciplines)

    assert mda.check_jacobian(input_names=["x"], output_names=["obj"])


def analytic_disciplines_from_desc(descriptions):
    return [
        AnalyticDiscipline(desc, name=str(i)) for i, desc in enumerate(descriptions)
    ]


@pytest.mark.parametrize(
    ("desc", "log_message"),
    [
        (
            (
                {"y": "x"},
                {"y": "z"},
            ),
            "The following outputs are defined multiple times: ['y'].",
        ),
        (
            ({"y": "x+y", "c1": "1-0.2*c2"}, {"c2": "0.1*c1"}),
            "The following disciplines contain self-couplings and strong couplings: "
            "['0'].",
        ),
    ],
)
def test_consistency_fail(desc, log_message, caplog) -> None:
    """Test that the consistency check is done properly.

    Args:
        desc: The mathematical expressions to create analytic disciplines.
        log_message: The expected warning message.
        caplog: Fixture to access and control log capturing.
    """
    with concretize_classes(BaseMDA):
        BaseMDA.Settings = BaseMDASettings
        BaseMDA(analytic_disciplines_from_desc(desc))
    assert log_message in caplog.text


@pytest.mark.parametrize("mda_class", [MDAJacobi, MDAGaussSeidel])
@pytest.mark.parametrize(
    "grammar_type", [Discipline.GrammarType.JSON, Discipline.GrammarType.SIMPLE]
)
def test_array_couplings(mda_class, grammar_type) -> None:
    disciplines = create_disciplines_from_desc(
        [("A", ["x", "y1"], ["y2"]), ("B", ["x", "y2"], ("y1",))],
        grammar_type=grammar_type,
    )

    a_disc = disciplines[0]
    del a_disc.io.input_grammar["y1"]
    a_disc.io.input_grammar.update_from_data({"y1": 2.0})

    with pytest.raises(InvalidDataError):
        a_disc.execute({"x": 2.0})


def test_stopping_criteria(base_mda_solver, caplog) -> None:
    """Test the stopping criteria."""
    caplog.clear()

    base_mda_solver.settings.tolerance = 1.0

    base_mda_solver.normed_residual = 0.5
    assert base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)
    base_mda_solver.normed_residual = 1.5
    assert not base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)

    base_mda_solver.settings.max_mda_iter = 1

    base_mda_solver._current_iter = 1
    assert base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)
    assert (
        caplog.records[0].message
        == "BaseMDASolver has reached its maximum number of iterations, but the "
        "normalized residual norm 1.5 is still above the tolerance 1.0."
    )
    base_mda_solver._current_iter = 0
    assert not base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)

    base_mda_solver.settings.max_consecutive_unsuccessful_iterations = 1

    base_mda_solver._BaseMDASolver__n_consecutive_unsuccessful_iterations = 1
    assert base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)
    assert (
        caplog.records[1].message
        == "BaseMDASolver has reached its maximum number of unsuccessful iterations, "
        "but the normalized residual norm 1.5 is still above the tolerance 1.0."
    )
    base_mda_solver._BaseMDASolver__n_consecutive_unsuccessful_iterations = 0
    assert not base_mda_solver._check_stopping_criteria(update_iteration_metrics=False)


def test_coupling_structure(sellar_disciplines) -> None:
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = CouplingStructure(sellar_disciplines)
    mda_sellar = MDAGaussSeidel(
        sellar_disciplines, coupling_structure=coupling_structure
    )
    assert mda_sellar.coupling_structure == coupling_structure


def test_log_convergence(base_mda_solver, caplog) -> None:
    """Check that the boolean log_convergence is correctly set."""
    caplog.clear()
    caplog.set_level(logging.INFO)

    base_mda_solver._set_resolved_variables(["y_1", "y_2"])
    base_mda_solver.io.data.update({"y_1": array([1.0]), "y_2": array([1.0])})
    base_mda_solver._compute_residuals({"y_1": array([2.0]), "y_2": array([1.0])})

    base_mda_solver.settings.log_convergence = False
    base_mda_solver._BaseMDASolver__update_iteration_metrics()
    assert len(caplog.records) == 0

    base_mda_solver.settings.log_convergence = True
    base_mda_solver._BaseMDASolver__update_iteration_metrics()
    assert (
        caplog.records[0].message
        == "BaseMDASolver running... Normalized residual norm = 1.00e+00 (iter. 2)"
    )


def test_not_numeric_couplings(caplog) -> None:
    """Test that a message is logged when an MDA includes non-numeric couplings."""
    caplog.set_level("DEBUG")
    sellar1 = Sellar1()
    # Tweak the output grammar and set y_1 as an array of string
    prop = sellar1.io.output_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items", prop)
    sub_prop["type"] = "string"

    # Tweak the input grammar and set y_1 as an array of string
    sellar2 = Sellar2()
    prop = sellar2.io.input_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items", prop)
    sub_prop["type"] = "string"

    with concretize_classes(BaseMDA):
        BaseMDA.Settings = BaseMDASettings
        BaseMDA([sellar1, sellar2])
        msg = "The coupling variable(s) {'y_1'} is/are not an array of numeric values."
        assert msg in caplog.text


def test_sequence_transformers_setters(sellar_mda) -> None:
    assert sellar_mda.acceleration_method == AccelerationMethod.NONE
    sellar_mda.acceleration_method = AccelerationMethod.SECANT
    assert sellar_mda.acceleration_method == AccelerationMethod.SECANT

    assert sellar_mda.over_relaxation_factor == 1.0
    sellar_mda.over_relaxation_factor = 0.5
    assert sellar_mda.over_relaxation_factor == 0.5


@pytest.fixture(scope="module")
def disciplines() -> list[LinearDiscipline]:
    return create_disciplines_from_desc([
        ("A", ["x", "b"], ["a"]),
        ("B", ["a", "b", "y"], ["b"]),
        ("C", ["b"], ["y"]),
    ])


@pytest.fixture(scope="module")
def reference_mda_jacobian(disciplines) -> dict[str, dict[str, ndarray]]:
    """Compute the Jacobian of the MDA to serve as a reference."""
    mda = MDANewtonRaphson(
        disciplines,
        max_mda_iter=100,
        tolerance=1e-12,
    )

    mda.add_differentiated_inputs("x")
    mda.add_differentiated_outputs("y")

    return mda.linearize()


@pytest.mark.parametrize("mode", [DerivationMode.DIRECT, DerivationMode.ADJOINT])
@pytest.mark.parametrize("matrix_type", JacobianAssembly.JacobianType)
def test_matrix_free_linearization(
    mode, matrix_type, disciplines, reference_mda_jacobian, caplog
) -> None:
    disciplines[1].matrix_free_jacobian = True

    mda = MDANewtonRaphson(
        disciplines,
        max_mda_iter=100,
        tolerance=1e-12,
    )

    mda.matrix_type = matrix_type
    mda.linearization_mode = mode

    mda.add_differentiated_inputs("x")
    mda.add_differentiated_outputs("y")
    mda.linearize()

    if matrix_type == JacobianAssembly.JacobianType.MATRIX:
        assert (
            "The Jacobian is given as a linear operator. Performing the assembly "
            "required to apply it to the identity which is not performant."
            in caplog.text
        )

    assert allclose(reference_mda_jacobian["y"]["x"], mda.jac["y"]["x"], atol=1e-12)


class LinearImplicitDiscipline(Discipline):
    def __init__(self, name, input_names, output_names, size=1) -> None:
        super().__init__(name=name)
        self.size = size

        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

        self.io.residual_to_state_variable = {"r": "w"}

        self.io.state_equations_are_solved = False
        self.mat = default_rng(SEED).standard_normal((size, size))

        self.io.input_grammar.defaults = {k: 0.5 * ones(size) for k in input_names}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        if self.io.state_equations_are_solved:
            self.io.data["w"] = solve(self.mat, self.io.data["a"])

        self.io.data["r"] = self.mat.dot(self.io.data["w"]) - self.io.data["a"]

    def _compute_jacobian(self, inputs, outputs) -> None:
        self._init_jacobian(inputs, outputs, fill_missing_keys=True)

        self.jac["r"]["w"] = self.mat
        self.jac["r"]["a"] = -eye(self.size)
        self.jac["w"]["a"] = solve(self.mat, eye(self.size))


@pytest.fixture(scope="module")
def coupled_disciplines():
    return [
        LinearDiscipline("A", ["x", "w", "c"], ["a"], inputs_size=10, outputs_size=10),
        LinearImplicitDiscipline("B", ["a", "w"], ["w", "r"], size=10),
        LinearDiscipline("C", ["a"], ["c"], inputs_size=10, outputs_size=10),
    ]


def test_mda_with_residuals(coupled_disciplines) -> None:
    coupled_disciplines[1].io.state_equations_are_solved = True
    mda = MDANewtonRaphson(
        coupled_disciplines,
        tolerance=1e-14,
        max_mda_iter=100,
        acceleration_method=AccelerationMethod.SECANT,
        over_relaxation_factor=1.0,
    )
    output = mda.execute()

    coupled_disciplines[1].io.state_equations_are_solved = False
    mda = MDANewtonRaphson(
        coupled_disciplines,
        tolerance=1e-14,
        max_mda_iter=100,
        acceleration_method=AccelerationMethod.SECANT,
        over_relaxation_factor=1.0,
    )
    output_ref = mda.execute()

    assert compare_dict_of_arrays(output, output_ref, tolerance=1e-12)


class DiscWithNonNumericInputs1(Discipline):
    def __init__(self):
        super().__init__()
        self.io.input_grammar.update_from_data({"x": zeros(1)})
        self.io.input_grammar.update_from_data({"a": zeros(1)})
        self.io.input_grammar.update_from_data({"b_file": array(["my_b_file"])})

        self.io.output_grammar.update_from_data({"y": zeros(1)})
        self.io.output_grammar.update_from_data({"b": zeros(1)})
        self.io.output_grammar.update_from_data({"a_file": "my_a_file"})

        self.io.input_grammar.defaults["x"] = array([0.5])
        self.io.input_grammar.defaults["a"] = array([0.5])
        self.io.input_grammar.defaults["b_file"] = array(["my_b_file"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = self.io.data["x"]
        a = self.io.data["a"]
        y = x
        b = 2 * a
        self.io.data.update({"y": y, "a_file": "my_a_file", "b": b})

    def _compute_jacobian(
        self,
        input_names,
        output_names,
    ) -> None:
        self.jac = {}
        self.jac["y"] = {}
        self.jac["b"] = {}
        self.jac["y"]["x"] = array([[1.0]])
        self.jac["y"]["a"] = array([[0.0]])
        self.jac["b"]["x"] = array([[0.0]])
        self.jac["b"]["a"] = array([[2.0]])


class DiscWithNonNumericInputs2(Discipline):
    def __init__(self):
        super().__init__()
        self.io.input_grammar.update_from_data({"y": zeros(1)})
        self.io.input_grammar.update_from_data({"a_file": "my_a_file"})
        self.io.output_grammar.update_from_data({"x": zeros(1)})
        self.io.output_grammar.update_from_data({"b_file": array(["my_b_file"])})
        self.io.input_grammar.defaults["y"] = array([0.5])
        self.io.input_grammar.defaults["a_file"] = "my_a_file"

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y = self.io.data["y"]
        x = 1.0 - 0.3 * y
        self.io.data.update({"x": x, "b_file": array(["my_b_file"])})

    def _compute_jacobian(
        self,
        input_names,
        output_names,
    ) -> None:
        self.jac = {}
        self.jac["x"] = {}
        self.jac["x"]["y"] = array([[-0.3]])


class DiscWithNonNumericInputs3(Discipline):
    def __init__(self):
        super().__init__()
        self.io.input_grammar.update_from_data({"x": zeros(1)})
        self.io.input_grammar.update_from_data({"y": zeros(1)})
        self.io.input_grammar.update_from_data({"b": zeros(1)})
        self.io.input_grammar.update_from_data({"a_file": "my_a_file"})

        self.io.output_grammar.update_from_data({"obj": zeros(1)})
        self.io.output_grammar.update_from_data({"out_file_2": "my_a_file"})
        self.io.input_grammar.defaults["x"] = array([0.5])
        self.io.input_grammar.defaults["y"] = array([0.5])
        self.io.input_grammar.defaults["b"] = array([0.5])
        self.io.input_grammar.defaults["a_file"] = "my_a_file"

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = self.io.data["x"]
        y = self.io.data["y"]
        obj = x - y
        self.io.data.update({"obj": obj, "out_file_2": "my_out_file"})

    def _compute_jacobian(
        self,
        input_names,
        output_names,
    ) -> None:
        x = self.io.data["x"][0]
        y = self.io.data["y"][0]
        self.jac = {}
        self.jac["obj"] = {}
        self.jac["obj"]["x"] = array([[1.0]])
        self.jac["obj"]["y"] = array([[-1.0]])
        self.jac["obj"]["b"] = array([[x - y]])


@pytest.mark.parametrize(
    ("mda_class", "include_weak_couplings"),
    [(MDAJacobi, True), (MDAGaussSeidel, True), (MDANewtonRaphson, False)],
)
def test_mda_with_non_numeric_couplings(mda_class, include_weak_couplings):
    """Test that MDAs can handle non-numeric couplings.

    Args:
        mda_class: The specific MDA to be tested.
        include_weak_couplings: Whether to include weak couplings in the MDA. The Newton
            method does not support weak couplings.
    """
    disciplines = [
        DiscWithNonNumericInputs1(),
        DiscWithNonNumericInputs2(),
    ]

    if include_weak_couplings:
        disciplines.append(DiscWithNonNumericInputs3())

    mda = mda_class(disciplines)
    mda.add_differentiated_inputs(["a"])
    mda.add_differentiated_outputs(["obj"] if include_weak_couplings else ["b"])

    inputs = {
        "a_file": "test",
        "b_file": array(["test"]),
    }
    mda_output = mda.execute(inputs)

    if include_weak_couplings:
        assert_almost_equal(mda_output["obj"], 0.0, decimal=5)
    else:
        assert_almost_equal(mda_output["b"], 1.0, decimal=5)

    assert mda.check_jacobian(
        input_data=inputs,
        input_names=["a"],
        output_names=["obj"] if include_weak_couplings else ["b"],
        linearization_mode="adjoint",
        threshold=1e-3,
    )


def test_scaling_method() -> None:
    """Test changing the `scaling` argument of an MDA."""
    sellar1 = Sellar1()
    sellar2 = Sellar2()

    with concretize_classes(BaseMDA):
        BaseMDA.Settings = BaseMDASettings
        mda = BaseMDA([sellar1, sellar2])
        mda.scaling = BaseMDA.ResidualScaling.NO_SCALING
        assert mda.scaling == BaseMDA.ResidualScaling.NO_SCALING


def test_namespaces(sellar_mda) -> None:
    """Test the well functionning with namespaces."""
    sellar_mda.add_namespace_to_output(sellar_mda.NORMALIZED_RESIDUAL_NORM, "foo")
    sellar_mda.execute()


def test_settings_type_error():
    settings = "toto"
    msg = (
        f"The Pydantic model must be a {BaseMDA.Settings.__name__}; "
        f"got {settings.__class__.__name__}"
    )

    BaseMDA.Settings = BaseMDASettings
    with concretize_classes(BaseMDA), pytest.raises(ValueError, match=msg):
        BaseMDA([], settings_model=settings)


def test_set_bounds():
    """Test that bounds are properly set."""
    lower_bound = -array([1.0])
    upper_bound = array([1.0])

    mda = MDAJacobi([
        LinearDiscipline("A", ["x", "b"], ["a"]),
        LinearDiscipline("B", ["a"], ["b", "y"]),
    ])

    mda.set_bounds({"a": (lower_bound, None)})

    assert mda.lower_bound_vector is None
    assert mda.upper_bound_vector is None

    mda._execute_disciplines_and_update_local_data()

    assert (mda.lower_bound_vector == array([-1.0, -inf])).all()
    assert (mda.upper_bound_vector == array([+inf, +inf])).all()

    assert (mda._sequence_transformer.lower_bound == array([-1.0, -inf])).all()
    assert (mda._sequence_transformer.upper_bound == array([+inf, +inf])).all()

    mda.set_bounds({"b": (2.0 * lower_bound, upper_bound)})

    assert (mda.lower_bound_vector == array([-1.0, -2.0])).all()
    assert (mda.upper_bound_vector == array([+inf, 1.0])).all()

    assert (mda._sequence_transformer.lower_bound == array([-1.0, -2.0])).all()
    assert (mda._sequence_transformer.upper_bound == array([+inf, 1.0])).all()
