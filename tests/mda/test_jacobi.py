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

from typing import TYPE_CHECKING

import pytest
from numpy import allclose as allclose_
from numpy import array
from numpy import isclose
from numpy.testing import assert_equal

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.process.mda_jacobi import SobieskiMDAJacobi
from gemseo.utils.discipline import DummyDiscipline

from .test_gauss_seidel import SelfCoupledDisc
from .utils import generate_parallel_doe

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


def allclose(a, b):
    return allclose_(a, b, atol=1e-8, rtol=0.0)


@pytest.fixture(scope="module")
def mda_setting() -> StrKeyMapping:
    """Returns the setting for all subsequent MDAs."""
    return {"tolerance": 1e-12, "max_mda_iter": 50}


@pytest.fixture(scope="module")
def reference(mda_setting) -> MDAJacobi:
    """An instance of Jacobi MDA on the Sobieski problem."""
    mda_jacobi = SobieskiMDAJacobi(
        **mda_setting, acceleration_method=AccelerationMethod.NONE
    )
    mda_jacobi.execute()
    return mda_jacobi


@pytest.mark.parametrize("relaxation", [0.8, 1.0, 1.2])
def test_over_relaxation(mda_setting, relaxation, reference) -> None:
    """Tests the relaxation factor."""
    mda = SobieskiMDAJacobi(
        **mda_setting,
        acceleration_method=AccelerationMethod.NONE,
        over_relaxation_factor=relaxation,
    )
    mda.execute()

    assert allclose(
        mda.get_current_resolved_residual_vector(),
        reference.get_current_resolved_residual_vector(),
    )
    assert allclose(
        mda.get_current_resolved_variables_vector(),
        reference.get_current_resolved_variables_vector(),
    )


@pytest.mark.parametrize("acceleration", AccelerationMethod)
def test_acceleration_methods(mda_setting, acceleration, reference) -> None:
    """Tests the acceleration methods."""
    mda = SobieskiMDAJacobi(**mda_setting, acceleration_method=acceleration)
    mda.execute()

    assert mda._current_iter <= reference._current_iter

    assert allclose(
        mda.get_current_resolved_residual_vector(),
        reference.get_current_resolved_residual_vector(),
    )
    assert allclose(
        mda.get_current_resolved_variables_vector(),
        reference.get_current_resolved_variables_vector(),
    )


def test_jacobi_sobieski() -> None:
    """Test the execution of Jacobi on Sobieski."""
    mda = SobieskiMDAJacobi()
    mda.execute()
    mda.default_input_data["x_shared"] += 0.02
    mda.warm_start = True
    mda.execute()
    assert mda.residual_history[-1] < 1e-4

    assert mda.io.data[mda.NORMALIZED_RESIDUAL_NORM][0] < 1e-4


def test_mda_jacobi_parallel() -> None:
    """Comparison of Jacobi on Sobieski problem: 1 and 4 processes."""
    sorted_couplings = ["y_12", "y_14", "y_21", "y_23", "y_24", "y_31", "y_32", "y_34"]

    mda_seq = SobieskiMDAJacobi()
    assert mda_seq._input_couplings == sorted_couplings
    assert mda_seq.parallel_execution is None
    mda_seq_local_data = mda_seq.execute()

    mda_parallel = SobieskiMDAJacobi(n_processes=4)
    assert mda_seq._input_couplings == sorted_couplings
    assert mda_parallel.parallel_execution is not None
    mda_parallel_local_data = mda_parallel.execute()

    assert_equal(mda_seq_local_data, mda_parallel_local_data)


def test_jacobi_sellar(sellar_disciplines) -> None:
    """Test the execution of Jacobi on Sobieski."""
    mda = MDAJacobi(sellar_disciplines)
    mda.execute()

    assert mda.residual_history[-1] < 1e-4
    assert mda.io.data[mda.NORMALIZED_RESIDUAL_NORM][0] < 1e-4


def test_expected_workflow() -> None:
    """Test MDAJacobi process_flow should be list of one tuple of disciplines (meaning
    parallel execution)"""
    disc1 = DummyDiscipline()
    disc2 = DummyDiscipline()
    disc3 = DummyDiscipline()
    disciplines = [disc1, disc2, disc3]

    mda = MDAJacobi(disciplines, n_processes=1)
    expected = (
        "{MDAJacobi(PENDING), [DummyDiscipline(PENDING), DummyDiscipline(PENDING), "
        "DummyDiscipline(PENDING)]}"
    )
    assert str(mda.get_process_flow().get_execution_flow()) == expected

    mda = MDAJacobi(disciplines, n_processes=2)
    expected = (
        "{MDAJacobi(PENDING), (DummyDiscipline(PENDING), DummyDiscipline(PENDING), "
        "DummyDiscipline(PENDING))}"
    )
    assert str(mda.get_process_flow().get_execution_flow()) == expected


def test_expected_workflow_with_adapter() -> None:
    discs = create_discipline([
        "SobieskiPropulsion",
        "SobieskiStructure",
        "SobieskiAerodynamics",
        "SobieskiMission",
    ])
    design_space = SobieskiDesignSpace()
    scn_propu = create_scenario(
        discs,
        "DisciplinaryOpt",
        "y_4",
        design_space.filter("x_3", copy=True),
        name="PropulsionScenario",
    )
    adapter_propu = MDOScenarioAdapter(scn_propu, ["x_1", "x_2"], ["x_3"])
    scn_aero = create_scenario(
        discs,
        "DisciplinaryOpt",
        "y_4",
        design_space.filter("x_2", copy=True),
        name="AeroScenario",
    )
    adapter_aero = MDOScenarioAdapter(scn_aero, ["x_1", "x_3"], ["x_2"])
    scn_struct = create_scenario(
        discs,
        "DisciplinaryOpt",
        "y_4",
        design_space.filter("x_1", copy=True),
        name="StructureScenario",
    )
    adapter_struct = MDOScenarioAdapter(scn_struct, ["x_2", "x_3"], ["x_1"])
    adapters = [adapter_propu, adapter_aero, adapter_struct]

    mda = MDAJacobi(adapters)

    expected = (
        "{MDAJacobi(PENDING), ("
        "{PropulsionScenario(PENDING), [SobieskiPropulsion(PENDING), "
        "SobieskiStructure(PENDING), SobieskiAerodynamics(PENDING), "
        "SobieskiMission(PENDING)]}, "
        "{AeroScenario(PENDING), [SobieskiPropulsion(PENDING), "
        "SobieskiStructure(PENDING), SobieskiAerodynamics(PENDING), "
        "SobieskiMission(PENDING)]}, "
        "{StructureScenario(PENDING), [SobieskiPropulsion(PENDING), "
        "SobieskiStructure(PENDING), SobieskiAerodynamics(PENDING), "
        "SobieskiMission(PENDING)]})}"
    )

    assert str(mda.get_process_flow().get_execution_flow()) == expected


def test_self_coupled() -> None:
    sc_disc = SelfCoupledDisc()
    mda = MDAJacobi([sc_disc], tolerance=1e-14, max_mda_iter=40)
    out = mda.execute()
    assert abs(out["y"] - 2.0 / 3.0) < 1e-6


def test_log_convergence(sellar_disciplines) -> None:
    """Check that the boolean log_convergence is correctly set."""
    mda = MDAJacobi(sellar_disciplines)
    assert not mda._log_convergence
    mda = MDAJacobi(sellar_disciplines, log_convergence=True)
    assert mda._log_convergence


def test_parallel_doe() -> None:
    """Test the execution of Jacobi in parallel."""
    obj = generate_parallel_doe("MDAJacobi", 7)
    assert isclose(array([-obj]), array([608.175]), atol=1e-3)


def test_no_coupling():
    """Check what happens when the disciplines are not coupled."""
    disciplines = [AnalyticDiscipline({"y": "a"}), AnalyticDiscipline({"z": "2*a"})]
    mda = MDAJacobi(disciplines)
    mda.default_input_data["a"] = array([1.0])
    assert not mda.get_process_flow()._get_disciplines_couplings(
        DependencyGraph(disciplines)
    )
    local_data = mda.execute()
    assert_equal(
        local_data,
        {
            "a": array([1.0]),
            "y": array([1.0]),
            "z": array([2.0]),
            "MDA residuals norm": array([0.0]),
        },
    )
