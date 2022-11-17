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
#     Matthias De Lozzo
from __future__ import annotations

import pytest
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.problems.sobieski.design_space import create_design_space
from gemseo.problems.sobieski.disciplines import create_disciplines_with_physical_naming
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import array
from numpy import ndarray
from numpy.testing import assert_allclose
from numpy.testing import assert_equal

FACTORS = [0.5, 1.0, 1.5]
DECIMAL = 5


@pytest.mark.parametrize("dtype", ["float64", "complex128"])
@pytest.mark.parametrize(
    "discipline_class",
    [SobieskiMission, SobieskiAerodynamics, SobieskiStructure, SobieskiPropulsion],
)
def test_dtype(dtype, discipline_class):
    """Check that the NumPy dtype is correctly passed to the underlying discipline."""
    discipline = discipline_class.create_with_physical_naming(dtype=dtype)
    assert discipline._discipline.dtype == dtype


@pytest.mark.parametrize("enable_delay", [False, True])
def test_enable_delay(enable_delay):
    """Check that enable_delay is correctly passed to the underlying SobieskiMission."""
    discipline = SobieskiMission.create_with_physical_naming(enable_delay=enable_delay)
    assert discipline._discipline.enable_delay is enable_delay


def compute_input_data(
    input_data: dict[str, ndarray], factor: float
) -> dict[str, ndarray]:
    """Compute new input data from existing ones.

    Args:
        input_data: The reference input data.
        factor: The factor applied to the reference input values.

    Returns:
        The new input data data.
    """
    return {k: v * factor for k, v in input_data.items()}


def compute_output_data(discipline: MDODiscipline, factor: float) -> dict[str, ndarray]:
    """Compute the output data of a discipline from default inputs.

    Args:
        discipline: The discipline to evaluate.
        factor: The factor applied to the default input values.

    Returns:
        The output data.
    """
    discipline.execute(compute_input_data(discipline.default_inputs, factor))
    return discipline.get_output_data()


@pytest.mark.parametrize("factor", FACTORS)
def test_mission_execute(factor):
    """Check the output data of SobieskiMissionPhysicalNaming."""
    output_data = compute_output_data(SobieskiMission(), factor)
    output_data_pn = compute_output_data(
        SobieskiMission.create_with_physical_naming(), factor
    )
    assert_equal(output_data_pn["range"], output_data["y_4"])


@pytest.mark.parametrize("factor", FACTORS)
def test_structure_execute(factor):
    """Check the output data of SobieskiStructurePhysicalNaming."""
    output_data = compute_output_data(SobieskiStructure(), factor)
    output_data_pn = compute_output_data(
        SobieskiStructure.create_with_physical_naming(), factor
    )
    assert_equal(output_data_pn["y_1"], output_data["y_1"])
    assert_equal(output_data_pn["y_11"], output_data["y_11"])
    assert_equal(output_data_pn["t_w_4"], output_data["y_14"][0:1])
    assert_equal(output_data_pn["t_w_2"], output_data["y_12"][0:1])
    assert_equal(output_data_pn["f_w"], output_data["y_1"][1:2])
    assert_equal(output_data_pn["stress"], output_data["g_1"][0:5])
    assert_equal(output_data_pn["twist_c"], output_data["g_1"][5:7])


@pytest.mark.parametrize("factor", FACTORS)
def test_aerodynamics_execute(factor):
    """Check the output data of SobieskiAerodynamicsPhysicalNaming."""
    output_data = compute_output_data(SobieskiAerodynamics(), factor)
    output_data_pn = compute_output_data(
        SobieskiAerodynamics.create_with_physical_naming(), factor
    )
    assert_equal(output_data_pn["y_2"], output_data["y_2"])
    assert_equal(output_data_pn["cl"], output_data["y_21"])
    assert_equal(output_data_pn["cd"], output_data["y_23"])
    assert_equal(output_data_pn["cl_cd"], output_data["y_24"])
    assert_equal(output_data_pn["dp_dx"], output_data["g_2"])


@pytest.mark.parametrize("factor", FACTORS)
def test_propulsion_execute(factor):
    """Check the output data of SobieskiPropulsionPhysicalNaming."""
    output_data = compute_output_data(SobieskiPropulsion(), factor)
    output_data_pn = compute_output_data(
        SobieskiPropulsion.create_with_physical_naming(), factor
    )
    assert_equal(output_data_pn["y_3"], output_data["y_3"])
    assert_equal(output_data_pn["esf"], output_data["y_32"])
    assert_equal(output_data_pn["esf_c"], output_data["g_3"][0:2])
    assert_equal(output_data_pn["throttle_c"], output_data["g_3"][2:3])
    assert_equal(output_data_pn["temperature"], output_data["g_3"][3:4])
    assert_equal(output_data_pn["sfc"], output_data["y_3"][0:1])
    assert_equal(output_data_pn["e_w"], output_data["y_3"][1:2])


@pytest.mark.parametrize(
    "discipline_class",
    [SobieskiMission, SobieskiAerodynamics, SobieskiStructure, SobieskiPropulsion],
)
@pytest.mark.parametrize("factor", FACTORS)
def test_mission_linearize(discipline_class, factor):
    """Check the Jacobian data of the different disciplines."""
    discipline = discipline_class.create_with_physical_naming()
    input_data = compute_input_data(discipline.default_inputs, factor)
    discipline.check_jacobian(input_data=input_data)


@pytest.fixture()
def mda():
    """An MDA for the Sobieski's use case with physical naming."""
    return MDAGaussSeidel(create_disciplines_with_physical_naming())


@pytest.mark.parametrize("factor", FACTORS)
def test_coupling(factor, mda):
    """Check the MDA results of the four disciplines."""
    design_space = create_design_space(physical_naming=True)
    input_data = compute_input_data(
        design_space.get_current_value(as_dict=True), factor
    )
    mda.execute(input_data)
    y_1 = mda.local_data["y_1"]
    y_2 = mda.local_data["y_2"]
    y_3 = mda.local_data["y_3"]
    y_4 = mda.local_data["range"]
    design_space = create_design_space()
    input_data = compute_input_data(
        design_space.get_current_value(as_dict=True), factor
    )
    original_mda = MDAGaussSeidel(
        [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
    )
    original_mda.execute(input_data)
    assert_allclose(original_mda.local_data["y_1"], y_1, rtol=1e-2)
    assert_allclose(original_mda.local_data["y_2"], y_2, rtol=1e-2)
    assert_allclose(original_mda.local_data["y_3"], y_3, rtol=1e-2)
    assert_allclose(original_mda.local_data["y_4"], y_4, rtol=1e-2)


@pytest.fixture
def scenario_pn() -> DOEScenario:
    """A DOEScenario for the Sobieski's SSBJ use case with physical naming."""
    scn = DOEScenario(
        create_disciplines_with_physical_naming(),
        "MDF",
        "range",
        create_design_space(physical_naming=True),
    )
    for constraint_name in [
        "stress",
        "twist_c",
        "dp_dx",
        "esf_c",
        "throttle_c",
        "temperature",
    ]:
        scn.add_constraint(constraint_name, constraint_type="ineq")
    scn.add_observable("y_1")
    scn.add_observable("y_2")
    scn.add_observable("y_3")
    return scn


@pytest.fixture
def scenario() -> DOEScenario:
    """A DOEScenario for the Sobieski's SSBJ use case without physical naming."""
    scn = DOEScenario(
        [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ],
        "MDF",
        "y_4",
        create_design_space(),
    )
    for constraint_name in ["g_1", "g_2", "g_3"]:
        scn.add_constraint(constraint_name, constraint_type="ineq")
    scn.add_observable("y_1")
    scn.add_observable("y_2")
    scn.add_observable("y_3")
    return scn


def test_scenario(scenario_pn, scenario):
    """Check the MDA results of the four disciplines."""
    scenario_pn.execute({"algo": "OT_HALTON", "n_samples": 10})
    dataset_pn = scenario_pn.export_to_dataset(opt_naming=False)
    scenario.execute({"algo": "OT_HALTON", "n_samples": 10})
    dataset = scenario.export_to_dataset(opt_naming=False)
    data_pn = dataset_pn.get_data_by_names(
        [
            "t_c",
            "altitude",
            "mach",
            "ar",
            "sweep",
            "area",
            "taper_ratio",
            "wingbox_area",
            "cf",
            "throttle",
        ],
        False,
    )
    data = dataset.get_data_by_names(["x_shared", "x_1", "x_2", "x_3"], False)
    assert_allclose(data, data_pn)

    data = dataset.get_data_by_names(["g_1", "g_2", "g_3"], False)
    data_pn = dataset_pn.get_data_by_names(
        [
            "stress",
            "twist_c",
            "dp_dx",
            "esf_c",
            "throttle_c",
            "temperature",
        ],
        False,
    )
    assert_allclose(data, data_pn)

    data = dataset_pn.get_data_by_names(["y_1", "y_2", "y_3"], False)
    data_pn = dataset.get_data_by_names(["y_1", "y_2", "y_3"], False)
    assert_allclose(data, data_pn)


def test_solution(mda):
    """Check the value of the range at optimum."""
    problem = mda.disciplines[0]._discipline.sobieski_problem
    optimum_design = problem.optimum_design
    input_data = {
        "t_c": optimum_design[4],
        "altitude": optimum_design[5],
        "mach": optimum_design[6],
        "ar": optimum_design[7],
        "sweep": optimum_design[8],
        "area": optimum_design[9],
        "taper_ratio": optimum_design[0],
        "wingbox_area": optimum_design[1],
        "cf": optimum_design[2],
        "throttle": optimum_design[3],
    }
    mda.execute({k: array([v]) for k, v in input_data.items()})
    optimum_range = problem.optimum_range
    assert_allclose(mda.local_data["range"], optimum_range, rtol=1e0)
