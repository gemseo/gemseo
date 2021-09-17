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
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.mda import MDA
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem

DIRNAME = os.path.dirname(__file__)


@pytest.fixture
def sellar_disciplines():
    return [Sellar1(), Sellar2(), SellarSystem()]


@pytest.fixture
def sellar_mda(sellar_disciplines):
    # initialize MDA from Sellar disciplines.
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
