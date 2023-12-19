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

import logging
import os

import pytest
from numpy import allclose
from numpy import array
from numpy import eye
from numpy import ndarray
from numpy import ones
from numpy.random import default_rng
from scipy.linalg import solve

from gemseo import SEED
from gemseo import create_discipline
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.base_mda_solver import BaseMDASolver
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda import MDA
from gemseo.mda.newton import MDANewtonRaphson
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.scalable.linear.linear_discipline import LinearDiscipline
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar import get_inputs
from gemseo.utils.comparisons import compare_dict_of_arrays
from gemseo.utils.testing.helpers import concretize_classes

DIRNAME = os.path.dirname(__file__)


@pytest.fixture()
def sellar_mda(sellar_disciplines):
    return MDAGaussSeidel(sellar_disciplines)


@pytest.fixture(scope="module")
def sellar_inputs():
    """Build dictionary with initial solution."""
    return get_inputs()


def test_reset(sellar_mda, sellar_inputs):
    """Test that the MDA successfully resets its disciplines after their executions."""
    disciplines = sellar_mda.disciplines
    for discipline in disciplines:
        discipline.execute(sellar_inputs)
        assert discipline.status == MDODiscipline.ExecutionStatus.DONE

    sellar_mda.reset_statuses_for_run()
    for discipline in disciplines:
        assert discipline.status == MDODiscipline.ExecutionStatus.PENDING


def test_input_couplings():
    with concretize_classes(BaseMDASolver):
        mda = BaseMDASolver([Sellar1()])
        mda._set_resolved_variables([])

    assert len(mda.get_current_resolved_variables_vector()) == 0

    with concretize_classes(BaseMDASolver):
        mda = BaseMDASolver(
            create_discipline([
                "SobieskiPropulsion",
                "SobieskiAerodynamics",
                "SobieskiMission",
                "SobieskiStructure",
            ])
        )
        mda._compute_input_couplings()
        sorted_c = ["y_12", "y_21", "y_23", "y_31", "y_32"]
        assert mda._input_couplings == sorted_c


def test_resolved_couplings():
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
    assert set(mda._resolved_variable_names) == set(mda.strong_couplings)

    with pytest.raises(AttributeError):
        mda._resolved_variable_names = "a"


def test_jacobian(sellar_mda, sellar_inputs):
    """Check the Jacobian computation."""
    sellar_mda.use_lu_fact = True
    sellar_mda.matrix_type = JacobianAssembly.JacobianType.LINEAR_OPERATOR
    with pytest.raises(
        ValueError, match="Unsupported LU factorization for LinearOperators"
    ):
        sellar_mda.linearize(
            sellar_inputs,
            compute_all_jacobians=True,
        )

    sellar_mda.use_lu_fact = False
    sellar_mda.linearize(sellar_inputs)
    assert sellar_mda.jac == {}

    sellar_mda._differentiated_inputs = []
    sellar_mda._differentiated_outputs = []

    sellar_mda.linearize(sellar_inputs)


def test_expected_workflow(sellar_mda):
    """"""
    expected = (
        "{MDAGaussSeidel(None), [Sellar1(None), Sellar2(None), "
        "SellarSystem(None), ], }"
    )
    assert str(sellar_mda.get_expected_workflow()) == expected


def test_warm_start():
    """Check that the warm start does not fail even at first execution."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda_sellar = MDAGaussSeidel(disciplines)
    mda_sellar.warm_start = True
    mda_sellar.execute()


def test_weak_strong_coupling_mda_jac():
    """Tests a particular coupling structure jacobian."""
    disciplines = analytic_disciplines_from_desc((
        {"y1": "x"},
        {"c1": "y1+x+0.2*c2"},
        {"c2": "y1+x+1.-0.3*c1"},
        {"obj": "x+c1+c2"},
    ))
    mda = MDAGaussSeidel(disciplines)

    assert mda.check_jacobian(inputs=["x"], outputs=["obj"])


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
def test_consistency_fail(desc, log_message, caplog):
    """Test that the consistency check is done properly.

    Args:
        desc: The mathematical expressions to create analytic disciplines.
        log_message: The expected warning message.
        caplog: Fixture to access and control log capturing.
    """
    with concretize_classes(MDA):
        MDA(analytic_disciplines_from_desc(desc))
    assert log_message in caplog.text


@pytest.mark.parametrize("mda_class", [MDAJacobi, MDAGaussSeidel])
@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.GrammarType.JSON, MDODiscipline.GrammarType.SIMPLE]
)
def test_array_couplings(mda_class, grammar_type):
    disciplines = create_disciplines_from_desc(
        [("A", ["x", "y1"], ["y2"]), ("B", ["x", "y2"], ("y1",))],
        grammar_type=grammar_type,
    )

    a_disc = disciplines[0]
    del a_disc.input_grammar["y1"]
    a_disc.input_grammar.update_from_data({"y1": 2.0})
    assert not a_disc.input_grammar.is_array("y1")

    with pytest.raises(InvalidDataError):
        a_disc.execute({"x": 2.0})


def test_convergence_warning(caplog):
    with concretize_classes(BaseMDASolver):
        mda = BaseMDASolver([Sellar1(), Sellar2(), SellarSystem()])
    mda.tolerance = 1.0
    mda.normed_residual = 2.0
    mda.max_mda_iter = 1
    caplog.clear()

    residual_is_small, max_iter_is_reached = mda._warn_convergence_criteria()
    assert not residual_is_small
    assert not max_iter_is_reached

    mda.scaling = BaseMDASolver.ResidualScaling.NO_SCALING

    mda._set_resolved_variables(mda.strong_couplings)
    mda.local_data.update({"y_1": array([1.0]), "y_2": array([1.0])})
    mda._update_residuals({"y_1": array([2.0]), "y_2": array([2.0])})

    mda._compute_residual()
    mda._warn_convergence_criteria()
    assert len(caplog.records) == 1
    assert (
        "BaseMDASolver has reached its maximum number of iterations"
        in caplog.records[0].message
    )

    mda.normed_residual = 1e-14
    residual_is_small, _ = mda._warn_convergence_criteria()
    assert residual_is_small


def test_coupling_structure(sellar_disciplines):
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = MDOCouplingStructure(sellar_disciplines)
    mda_sellar = MDAGaussSeidel(
        sellar_disciplines, coupling_structure=coupling_structure
    )
    assert mda_sellar.coupling_structure == coupling_structure


def test_log_convergence(caplog):
    """Check that the boolean log_convergence is correctly set."""
    with concretize_classes(BaseMDASolver):
        mda = BaseMDASolver([Sellar1(), Sellar2(), SellarSystem()])
    assert not mda.log_convergence

    mda.log_convergence = True
    assert mda.log_convergence

    caplog.set_level(logging.INFO)

    mda._set_resolved_variables(mda.strong_couplings)
    mda.local_data.update({"y_1": array([1.0]), "y_2": array([1.0])})
    mda._update_residuals({"y_1": array([2.0]), "y_2": array([1.0])})

    mda._compute_residual(store_it=False)
    assert (
        "BaseMDASolver running... Normed residual = 1.00e+00 (iter. 0)"
        not in caplog.text
    )

    mda._compute_residual(log_normed_residual=True)
    assert (
        "BaseMDASolver running... Normed residual = 1.00e+00 (iter. 0)" in caplog.text
    )


def test_not_numeric_couplings():
    """Test that an exception is raised if strings are used as couplings in MDA."""
    sellar1 = Sellar1()
    # Tweak the output grammar and set y_1 as an array of string
    prop = sellar1.output_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items", prop)
    sub_prop["type"] = "string"

    # Tweak the input grammar and set y_1 as an array of string
    sellar2 = Sellar2()
    prop = sellar2.input_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items", prop)
    sub_prop["type"] = "string"

    with pytest.raises(
        TypeError, match=r"The coupling variables \['y\_1'\] must be numeric\."
    ), concretize_classes(MDA):
        MDA([sellar1, sellar2])


@pytest.mark.parametrize("mda_class", [MDAJacobi, MDAGaussSeidel, MDANewtonRaphson])
def test_get_sub_disciplines(
    mda_class,
):
    """Test the get_sub_disciplines method.

    Args:
        mda_class: The specific MDA to be tested.
    """
    disciplines = [Sellar1(), Sellar2()]
    mda = mda_class(disciplines)
    assert mda.get_sub_disciplines() == mda.disciplines == disciplines


def test_sequence_transformers_setters(sellar_mda):
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
):
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


class LinearImplicitDiscipline(MDODiscipline):
    def __init__(self, name, input_names, output_names, size=1):
        super().__init__(name=name)
        self.size = size

        self.input_grammar.update_from_names(input_names)
        self.output_grammar.update_from_names(output_names)

        self.residual_variables = {"r": "w"}

        self.run_solves_residuals = False
        self.mat = default_rng(SEED).standard_normal((size, size))

        self.default_inputs = {k: 0.5 * ones(size) for k in input_names}

    def _run(self) -> None:
        if self.run_solves_residuals:
            self.local_data["w"] = solve(self.mat, self.local_data["a"])

        self.local_data["r"] = self.mat.dot(self.local_data["w"]) - self.local_data["a"]

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


def test_mda_with_residuals(coupled_disciplines):
    coupled_disciplines[1].run_solves_residuals = True
    mda = MDANewtonRaphson(
        coupled_disciplines,
        tolerance=1e-14,
        max_mda_iter=100,
        acceleration_method=AccelerationMethod.SECANT,
        over_relaxation_factor=1.0,
    )
    output = mda.execute()

    coupled_disciplines[1].run_solves_residuals = False
    mda = MDANewtonRaphson(
        coupled_disciplines,
        tolerance=1e-14,
        max_mda_iter=100,
        acceleration_method=AccelerationMethod.SECANT,
        over_relaxation_factor=1.0,
    )
    output_ref = mda.execute()

    assert compare_dict_of_arrays(output, output_ref, tolerance=1e-12)
