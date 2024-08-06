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
from numpy.testing import assert_almost_equal

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.process.mda_gauss_seidel import SobieskiMDAGaussSeidel
from gemseo.utils.testing.helpers import image_comparison

from ..core.test_chain import two_virtual_disciplines  # noqa: F401

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


def allclose(a, b):
    return allclose_(a, b, atol=1e-10, rtol=0.0)


@pytest.fixture(scope="module")
def mda_setting() -> StrKeyMapping:
    """Returns the setting for all subsequent MDAs."""
    return {"tolerance": 1e-12, "max_mda_iter": 30}


@pytest.fixture(scope="module")
def reference(mda_setting) -> MDAGaussSeidel:
    """An instance of Gauss-Seidel MDA on the Sobieski problem."""
    mda_gauss_seidel = SobieskiMDAGaussSeidel(**mda_setting)
    mda_gauss_seidel.execute()
    return mda_gauss_seidel


@pytest.mark.parametrize("relaxation", [0.8, 1.0, 1.2])
def test_over_relaxation(mda_setting, relaxation, reference) -> None:
    """Tests the relaxation factor."""
    mda = SobieskiMDAGaussSeidel(**mda_setting, over_relaxation_factor=relaxation)
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
    mda = SobieskiMDAGaussSeidel(**mda_setting, acceleration_method=acceleration)
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


@image_comparison(["sobieski"])
def test_sobieski(tmp_wd) -> None:
    """Test the execution of Gauss-Seidel on Sobieski."""
    mda = SobieskiMDAGaussSeidel(tolerance=1e-12, max_mda_iter=30)
    mda.default_inputs["x_shared"] += 0.1
    mda.execute()
    mda.default_inputs["x_shared"] += 0.1
    mda.warm_start = True
    mda.execute()

    assert mda.residual_history[-1] < 1e-4

    mda.plot_residual_history(save=False)


def test_expected_workflow() -> None:
    """Test MDA GaussSeidel workflow should be disciplines sequence."""
    disc1 = MDODiscipline()
    disc2 = MDODiscipline()
    disc3 = MDODiscipline()
    disciplines = [disc1, disc2, disc3]

    mda = MDAGaussSeidel(disciplines)
    expected = (
        "{MDAGaussSeidel(None), [MDODiscipline(None), "
        "MDODiscipline(None), MDODiscipline(None), ], }"
    )
    assert str(mda.get_expected_workflow()) == expected


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

    mda = MDAGaussSeidel(adapters)

    expected = (
        "{MDAGaussSeidel(None), ["
        "{PropulsionScenario(None), [SobieskiPropulsion(None), "
        "SobieskiStructure(None), SobieskiAerodynamics(None), "
        "SobieskiMission(None), ], }, "
        "{AeroScenario(None), [SobieskiPropulsion(None), "
        "SobieskiStructure(None), SobieskiAerodynamics(None), "
        "SobieskiMission(None), ], }, "
        "{StructureScenario(None), [SobieskiPropulsion(None), "
        "SobieskiStructure(None), SobieskiAerodynamics(None), "
        "SobieskiMission(None), ], }, "
        "], }"
    )
    assert str(mda.get_expected_workflow()) == expected


def test_self_coupled() -> None:
    for plus_y in [False, True]:
        sc_disc = SelfCoupledDisc(plus_y)
        mda = MDAGaussSeidel([sc_disc], tolerance=1e-14, max_mda_iter=40)
        _ = mda.execute()
        # assert abs(out["y"] - 2. / 3.) < 1e-6

        mda.add_differentiated_inputs(["x"])
        mda.add_differentiated_outputs(["o"])
        jac1 = mda.linearize()

        mda.set_jacobian_approximation()
        mda.cache.clear()
        jac2 = mda.linearize()
        assert abs(jac1["o"]["x"][0, 0] - jac2["o"]["x"][0, 0]) < 1e-3


class SelfCoupledDisc(MDODiscipline):
    def __init__(self, plus_y=False) -> None:
        MDODiscipline.__init__(self)
        self.input_grammar.update_from_names(["y", "x"])
        self.output_grammar.update_from_names(["y", "o"])
        self.default_inputs["y"] = array([0.25])
        self.default_inputs["x"] = array([0.0])
        self.coeff = 1.0
        if not plus_y:
            self.coeff = -1.0

    def _run(self) -> None:
        self.local_data["y"] = (
            1.0 + self.coeff * 0.5 * self.local_data["y"] + self.local_data["x"]
        )
        self.local_data["o"] = self.local_data["y"] + self.local_data["x"]

    def _compute_jacobian(self, inputs=None, outputs=None) -> None:
        self.jac = {
            "y": {"y": self.coeff * array([[0.5]]), "x": array([[1.0]])},
            "o": {"y": array([[1.0]]), "x": array([[1.0]])},
        }


def test_log_convergence() -> None:
    """Check that the boolean log_convergence is correctly set."""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda = MDAGaussSeidel(disciplines)
    assert not mda.log_convergence
    mda = MDAGaussSeidel(disciplines, log_convergence=True)
    assert mda.log_convergence


def test_parallel_doe(generate_parallel_doe_data) -> None:
    """Test the execution of GaussSeidel in parallel.

    Args:
        generate_parallel_doe_data: Fixture that returns the optimum solution to
            a parallel DOE scenario for a particular `main_mda_name`.
    """
    obj = generate_parallel_doe_data("MDAGaussSeidel")
    assert isclose(array([-obj]), array([608.185]), atol=1e-3)


@pytest.mark.parametrize(
    ("baseline_images", "n_iterations", "logscale"),
    [
        (["all_iter_default_log"], None, None),
        (["all_iter_modified_log"], None, [1e-10, 10.0]),
        (["five_iter_default_log"], 5, None),
        (["five_iter_modified_log"], 5, [1e-10, 10.0]),
        (["n_iter_larger_than_history"], 50, None),
    ],
)
@image_comparison(None, tol=0.098)
def test_plot_residual_history(baseline_images, n_iterations, logscale, caplog) -> None:
    """Test the residual history plot.

    Args:
        baseline_images: The reference images for the test.
        n_iterations: The number of iterations to plot.
        logscale: The limits of the ``y`` axis.
        caplog: Fixture to access and control log capturing.
    """
    mda = SobieskiMDAGaussSeidel(max_mda_iter=15)
    mda.execute()
    mda.default_inputs["x_shared"] += 0.1
    mda.execute()
    mda.plot_residual_history(save=False, n_iterations=n_iterations, logscale=logscale)

    if n_iterations == 50:
        assert (
            "Requested 50 iterations but the residual history contains only "
            f"{len(mda.residual_history)}, plotting all the residual history."
            in caplog.text
        )


def test_virtual_exe_mda(two_virtual_disciplines):  # noqa: F811
    """Test an MDA with disciplines in virtual execution mode."""
    chain = MDAGaussSeidel(two_virtual_disciplines)
    chain.execute()
    assert chain.local_data["x"] == 1.0
    assert chain.local_data["y"] == 2.0


def test_max_mda_iter_0():
    """Check that Gauss-Seidel calls the disciplines only once when max_mda_iter=0."""
    mda = SobieskiMDAGaussSeidel(max_mda_iter=0)
    assert mda.NORMALIZED_RESIDUAL_NORM not in mda.output_grammar

    mda.execute()

    for discipline in mda.disciplines:
        assert discipline.n_calls == 1

    output_data = mda.get_output_data()

    expected_output_data = {}
    local_data = dict(mda.default_inputs)
    for discipline in mda.disciplines:
        discipline.cache.clear()
        discipline.execute(local_data)
        output_data_ = discipline.get_output_data()
        expected_output_data.update(output_data_)
        local_data.update(output_data_)

    for output_name, output_value in output_data.items():
        assert_almost_equal(output_value, expected_output_data[output_name])
