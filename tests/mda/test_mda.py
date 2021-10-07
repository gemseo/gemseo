# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import logging
import os

import numpy as np
import pytest

from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.mda import MDA
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem

DIRNAME = os.path.dirname(__file__)


@pytest.fixture
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
        assert discipline.status == MDODiscipline.STATUS_DONE

    sellar_mda.reset_statuses_for_run()
    for discipline in disciplines:
        assert discipline.status == MDODiscipline.STATUS_PENDING


def test_input_couplings():
    mda = MDA([Sellar1()])
    assert len(mda._current_input_couplings()) == 0


def test_jacobian(sellar_mda, sellar_inputs):
    """Check the Jacobian computation."""
    sellar_mda.use_lu_fact = True
    sellar_mda.matrix_type = JacobianAssembly.LINEAR_OPERATOR
    with pytest.raises(
        ValueError, match="Unsupported LU factorization for LinearOperators"
    ):
        sellar_mda.linearize(
            sellar_inputs,
            force_all=True,
        )

    sellar_mda.use_lu_fact = False
    sellar_mda.linearize(sellar_inputs)
    assert sellar_mda.jac == {}

    sellar_mda._differentiated_inputs = None
    sellar_mda._differentiated_outputs = None

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
    return [AnalyticDiscipline(expressions_dict=desc) for desc in descriptions]


@pytest.mark.parametrize(
    "desc",
    [
        (
            {"y": "x"},
            {"y": "z"},
        ),
        (
            {"y": "x+y", "c1": "1-0.2*c2"},
            {"c2": "0.1*c1"},
        ),
    ],
)
def test_consistency_fail(desc):
    disciplines = analytic_disciplines_from_desc(desc)
    with pytest.raises(
        ValueError,
        match="Too many coupling constraints|Outputs are defined multiple times",
    ):
        MDA(disciplines)


@pytest.mark.parametrize("mda_class", [MDAJacobi, MDAGaussSeidel])
@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.JSON_GRAMMAR_TYPE, MDODiscipline.SIMPLE_GRAMMAR_TYPE]
)
def test_array_couplings(mda_class, grammar_type):
    disciplines = create_disciplines_from_desc(
        [("A", ["x", "y1"], ["y2"]), ("B", ["x", "y2"], ("y1",))],
        grammar_type=grammar_type,
    )

    a_disc = disciplines[0]
    a_disc.input_grammar.remove_item("y1")
    a_disc.default_inputs["y1"] = 2.0
    a_disc.input_grammar.initialize_from_base_dict({"y1": 2.0})
    assert not a_disc.input_grammar.is_type_array("y1")

    with pytest.raises(InvalidDataException):
        a_disc.execute({"x": 2.0})

    with pytest.raises(ValueError, match="must be of type array"):
        mda_class(disciplines, grammar_type=grammar_type)


def test_convergence_warning(caplog):
    mda = MDA([Sellar1()])
    mda.tolerance = 1.0
    mda.normed_residual = 2.0
    mda.max_mda_iter = 1
    caplog.clear()
    residual_is_small, _ = mda._warn_convergence_criteria(10)
    assert not residual_is_small
    assert len(caplog.records) == 1
    assert (
        "MDA has reached its maximum number of iterations" in caplog.records[0].message
    )

    mda.normed_residual = 1e-14
    residual_is_small, _ = mda._warn_convergence_criteria(1)
    assert residual_is_small

    mda.max_mda_iter = 2
    _, max_iter_is_reached = mda._warn_convergence_criteria(2)
    assert max_iter_is_reached
    _, max_iter_is_reached = mda._warn_convergence_criteria(1)
    assert not max_iter_is_reached


def test_coupling_structure(sellar_disciplines):
    """Check that an MDA is correctly instantiated from a coupling structure."""
    coupling_structure = MDOCouplingStructure(sellar_disciplines)
    mda_sellar = MDAGaussSeidel(
        sellar_disciplines, coupling_structure=coupling_structure
    )
    assert mda_sellar.coupling_structure == coupling_structure


def test_log_convergence(caplog):
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]

    mda = MDA(disciplines)
    assert not mda.log_convergence

    mda.log_convergence = True
    assert mda.log_convergence

    caplog.set_level(logging.INFO)

    mda._compute_residual(np.array([1, 2]), np.array([2, 1]), 1)
    assert "MDA running... Normed residual = 1.00e+00 (iter. 1)" not in caplog.text

    mda._compute_residual(
        np.array([1, 2]), np.array([2, 1]), 1, log_normed_residual=True
    )
    assert "MDA running... Normed residual = 1.00e+00 (iter. 1)" in caplog.text
