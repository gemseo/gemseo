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
from typing import TYPE_CHECKING

import numpy as np
import pytest

from gemseo import create_discipline
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.derivation_modes import DerivationMode
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.errors import InvalidDataError
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda import MDA
from gemseo.mda.newton import MDANewtonRaphson
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import Sellar2
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.utils.testing.helpers import concretize_classes

if TYPE_CHECKING:
    from gemseo.problems.scalable.linear.linear_discipline import LinearDiscipline

DIRNAME = os.path.dirname(__file__)


@pytest.fixture()
def sellar_mda(sellar_disciplines):
    return MDAGaussSeidel(sellar_disciplines)


@pytest.fixture(scope="module")
def sellar_inputs():
    """Build dictionary with initial solution."""
    x_local = np.array([0.0], dtype=np.float64)
    x_shared = np.array([1.0, 0.0], dtype=np.float64)
    y_0 = np.ones(1, dtype=np.complex128)
    y_1 = np.ones(1, dtype=np.complex128)
    return {"x_local": x_local, "x_shared": x_shared, "y_0": y_0, "y_1": y_1}


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
    with concretize_classes(MDA):
        mda = MDA([Sellar1()])
    assert len(mda._current_working_couplings()) == 0

    with concretize_classes(MDA):
        mda = MDA(
            create_discipline(
                [
                    "SobieskiPropulsion",
                    "SobieskiAerodynamics",
                    "SobieskiMission",
                    "SobieskiStructure",
                ]
            )
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
    assert set(mda._resolved_coupling_names) == set(mda._input_couplings)

    mda = MDAGaussSeidel(disciplines)
    assert set(mda._resolved_coupling_names) == set(mda.strong_couplings)

    with pytest.raises(
        RuntimeError,
        match="The resolved coupling names have already been set, any changes could "
        "make the MDA solution inconsistent.",
    ):
        mda._resolved_coupling_names = "a"


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
    disciplines = analytic_disciplines_from_desc(
        (
            {"y1": "x"},
            {"c1": "y1+x+0.2*c2"},
            {"c2": "y1+x+1.-0.3*c1"},
            {"obj": "x+c1+c2"},
        )
    )
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

    with pytest.raises(TypeError, match="must be of type array"):
        mda_class(disciplines, grammar_type=grammar_type)


def test_convergence_warning(caplog):
    with concretize_classes(MDA):
        mda = MDA([Sellar1()])
    mda.tolerance = 1.0
    mda.normed_residual = 2.0
    mda.max_mda_iter = 1
    caplog.clear()
    residual_is_small, max_iter_is_reached = mda._warn_convergence_criteria()
    assert not residual_is_small
    assert not max_iter_is_reached

    mda.scaling = MDA.ResidualScaling.NO_SCALING
    mda._compute_residual(np.array([1, 2]), np.array([10, 10]))
    mda._warn_convergence_criteria()
    assert len(caplog.records) == 1
    assert (
        "MDA has reached its maximum number of iterations" in caplog.records[0].message
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
    with concretize_classes(MDA):
        mda = MDA([Sellar1(), Sellar2(), SellarSystem()])
    assert not mda.log_convergence

    mda.log_convergence = True
    assert mda.log_convergence

    caplog.set_level(logging.INFO)

    mda._compute_residual(np.array([1, 2]), np.array([2, 1]), store_it=False)
    assert "MDA running... Normed residual = 1.00e+00 (iter. 0)" not in caplog.text

    mda._compute_residual(np.array([1, 2]), np.array([2, 1]), log_normed_residual=True)
    assert "MDA running... Normed residual = 1.00e+00 (iter. 0)" in caplog.text


def test_not_numeric_couplings():
    """Test that an exception is raised if strings are used as couplings in MDA."""
    sellar1 = Sellar1()
    # Tweak the ouput grammar and set y_1 as an array of string
    prop = sellar1.output_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items")
    sub_prop["type"] = "string"

    # Tweak the input grammar and set y_1 as an array of string
    sellar2 = Sellar2()
    prop = sellar2.input_grammar.schema.get("properties").get("y_1")
    sub_prop = prop.get("items")
    sub_prop["type"] = "string"

    with pytest.raises(
        TypeError, match=r"The coupling variables \['y\_1'\] must be of type array\."
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
    return create_disciplines_from_desc(
        [
            ("A", ["x", "b"], ["a"]),
            ("B", ["a", "b", "y"], ["b"]),
            ("C", ["b"], ["y"]),
        ]
    )


@pytest.fixture(scope="module")
def reference_mda_jacobian(disciplines) -> dict[str, dict[str, np.ndarray]]:
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

    assert np.allclose(reference_mda_jacobian["y"]["x"], mda.jac["y"]["x"], atol=1e-12)
