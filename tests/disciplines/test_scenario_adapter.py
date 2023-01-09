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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path

import pytest
from gemseo.algos.database import Database
from gemseo.core.chain import MDOChain
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.disciplines.scenario_adapter import MDOObjScenarioAdapter
from gemseo.disciplines.scenario_adapter import MDOScenarioAdapter
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from numpy import all as np_all
from numpy import allclose
from numpy import array
from numpy import matmul
from numpy import ones
from numpy import zeros
from numpy import zeros_like


def create_design_space():
    """"""
    return SobieskiProblem().design_space


@pytest.fixture
def scenario():
    """An MDO scenario solving the Sobieski problem with MDF and L-BFGS-B."""
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]
    design_space = create_design_space()
    design_space.filter(["x_1", "x_2", "x_3"])
    mdo_scenario = MDOScenario(
        disciplines,
        formulation="MDF",
        objective_name="y_4",
        design_space=design_space,
        maximize_objective=True,
        name="MyScenario",
    )
    mdo_scenario.default_inputs = {"max_iter": 35, "algo": "L-BFGS-B"}
    return mdo_scenario


def test_default_name(scenario):
    """Check the default name of an MDOScenarioAdapter."""
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])
    assert adapter.name == "MyScenario_adapter"


def test_name(scenario):
    """Check that the name of the MDOScenarioAdapter is correctly set."""
    name = "MyAdapter"
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"], name=name)
    assert adapter.name == name


def test_adapter(scenario):
    """Test the MDOAdapter."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs)
    gen = MDOFunctionGenerator(adapter)
    func = gen.get_function(inputs, outputs)
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    f_x1 = func(x_shared)
    f_x2 = func(x_shared)
    assert f_x1 == f_x2
    x_shared = array([0.09, 60000, 1.4, 2.5, 70, 1500])

    f_x3 = func(x_shared)
    assert f_x3 > 4947.0


def test_adapter_set_x0_before_opt(scenario):
    """Test the MDOScenarioAdapter with set_x0_before_opt."""
    inputs = ["x_1", "x_2", "x_3", "x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_x0_before_opt=True)
    gen = MDOFunctionGenerator(adapter)
    x_shared = array([0.25, 1.0, 1.0, 0.5, 0.09, 60000, 1.4, 2.5, 70, 1500])
    func = gen.get_function(inputs, outputs)
    f_x3 = func(x_shared)
    assert f_x3 > 4947.0


def test_adapter_set_and_reset_x0(scenario):
    """Test that set and reset x_0 cannot be done at MDOScenarioAdapter instantiation."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    msg = "Inconsistent options for MDOScenarioAdapter."
    with pytest.raises(ValueError, match=msg):
        MDOScenarioAdapter(
            scenario, inputs, outputs, set_x0_before_opt=True, reset_x0_before_opt=True
        )


def test_adapter_miss_dvs(scenario):
    inputs = ["x_shared"]
    outputs = ["y_4", "missing_dv"]
    scenario.design_space.add_variable("missing_dv")
    MDOScenarioAdapter(scenario, inputs, outputs)


def test_adapter_reset_x0_before_opt(scenario):
    """Check MDOScenarioAdapter.reset_x0_before_opt()."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    design_space = scenario.design_space
    initial_design = design_space.dict_to_array(
        design_space.get_current_value(as_dict=True)
    )
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, reset_x0_before_opt=True)
    adapter.execute()
    x_shared = adapter.default_inputs["x_shared"] * 1.01
    adapter.default_inputs["x_shared"] = x_shared
    # initial_x is reset to the initial design value before optimization;
    # thus the optimization starts from initial_design.
    adapter.execute()
    initial_x = adapter.scenario.formulation.opt_problem.database.get_x_by_iter(0)
    assert np_all(initial_x == initial_design)

    adapter = MDOScenarioAdapter(scenario, inputs, outputs)
    adapter.execute()
    new_initial_design = design_space.dict_to_array(
        design_space.get_current_value(as_dict=True)
    )
    adapter.default_inputs["x_shared"] = x_shared
    # initial_x is NOT reset to the initial design value before optimization;
    # thus the optimization starts from the last design value (=new_initial_design).
    adapter.execute()
    initial_x = adapter.scenario.formulation.opt_problem.database.get_x_by_iter(0)
    assert np_all(initial_x == new_initial_design)
    assert not np_all(initial_x == initial_design)


def test_adapter_set_bounds(scenario):
    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_bounds_before_opt=True)

    # Execute the adapter with default bounds
    adapter.execute()
    ds = scenario.design_space
    assert np_all(ds.get_lower_bounds() == [0.1, 0.75, 0.75, 0.1])
    assert np_all(ds.get_upper_bounds() == [0.4, 1.25, 1.25, 1.0])

    # Execute the adapter with passed bounds
    input_data = dict()
    lower_bounds = ds.array_to_dict(zeros(4))
    lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
    upper_bounds = ds.array_to_dict(ones(4))
    upper_suffix = MDOScenarioAdapter.UPPER_BND_SUFFIX
    for bounds, suffix in [
        (lower_bounds, lower_suffix),
        (upper_bounds, upper_suffix),
    ]:
        bounds = {name + suffix: val for name, val in bounds.items()}
        input_data.update(bounds)
    adapter.execute(input_data)
    assert np_all(ds.get_lower_bounds() == zeros(4))
    assert np_all(ds.get_upper_bounds() == ones(4))


def test_chain(scenario):
    """"""
    mda = scenario.formulation.mda
    inputs = list(mda.get_input_data_names()) + scenario.design_space.variables_names
    outputs = ["x_1", "x_2", "x_3"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs)

    # Allow re exec when DONE for the chain execution
    mda.re_exec_policy = mda.RE_EXECUTE_DONE_POLICY
    chain = MDOChain([mda, adapter, mda])

    # Sobieski Z opt
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    chain.execute({"x_shared": x_shared})

    y_4 = chain.local_data["y_4"]
    assert y_4 > 2908.0


def test_compute_jacobian(scenario):
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])
    adapter.execute()
    adapter._compute_jacobian()
    expected_output_names = {"y_4", "mult_dot_constr_jac"}

    assert set(adapter.jac.keys()) == expected_output_names

    for output_name in expected_output_names:
        assert set(adapter.jac[output_name].keys()) == {"x_shared"}


def test_compute_jacobian_with_bound_inputs(scenario):
    adapter = MDOScenarioAdapter(
        scenario, ["x_shared"], ["y_4"], set_bounds_before_opt=True
    )
    expected_input_names = ["x_shared", "x_1_lower_bnd"]
    adapter.execute()
    adapter._compute_jacobian(inputs=expected_input_names)
    expected_output_names = {"y_4", "mult_dot_constr_jac"}

    assert set(adapter.jac.keys()) == expected_output_names

    for output_name in expected_output_names:
        assert set(adapter.jac[output_name].keys()) == set(expected_input_names)


def test_compute_jacobian_exceptions(scenario):
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])

    # Pass invalid inputs
    with pytest.raises(
        ValueError, match="The following are not inputs of the adapter: bar, foo."
    ):
        adapter._compute_jacobian(inputs=["x_shared", "foo", "bar"])

    # Pass invalid outputs
    with pytest.raises(
        ValueError, match="The following are not outputs of the adapter: bar, foo."
    ):
        adapter._compute_jacobian(outputs=["y_4", "foo", "bar"])

    # Pass invalid differentiated outputs
    scenario.add_constraint(["g_1"])
    scenario.add_constraint(["g_2"])
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4", "g_1", "g_2"])
    with pytest.raises(
        ValueError, match="Post-optimal Jacobians of g_1, g_2 cannot be computed."
    ):
        adapter._compute_jacobian(outputs=["y_4", "g_2", "g_1"])

    # Pass a multi-valued objective
    scenario.formulation.opt_problem.objective.outvars = ["y_4"] * 2
    with pytest.raises(ValueError, match="The objective must be single-valued."):
        adapter._compute_jacobian()


def build_struct_scenario():
    ds = SobieskiProblem().design_space
    sc_str = MDOScenario(
        disciplines=[SobieskiStructure()],
        formulation="DisciplinaryOpt",
        objective_name="y_11",
        design_space=deepcopy(ds).filter("x_1"),
        name="StructureScenario",
        maximize_objective=True,
    )
    sc_str.add_constraint("g_1", constraint_type="ineq")
    sc_str.default_inputs = {"max_iter": 20, "algo": "NLOPT_SLSQP"}

    return sc_str


def build_prop_scenario():
    ds = SobieskiProblem().design_space
    sc_prop = MDOScenario(
        disciplines=[SobieskiPropulsion()],
        formulation="DisciplinaryOpt",
        objective_name="y_34",
        design_space=deepcopy(ds).filter("x_3"),
        name="PropulsionScenario",
    )
    sc_prop.add_constraint("g_3", constraint_type="ineq")
    sc_prop.default_inputs = {"max_iter": 20, "algo": "NLOPT_SLSQP"}

    return sc_prop


def check_adapter_jacobian(adapter, inputs, objective_threshold, lagrangian_threshold):
    opt_problem = adapter.scenario.formulation.opt_problem
    outvars = opt_problem.objective.outvars
    constraints = opt_problem.get_constraints_names()

    # Test the Jacobian accuracy as objective Jacobian
    assert adapter.check_jacobian(
        inputs=inputs, outputs=outvars, threshold=objective_threshold
    )

    # Test the Jacobian accuracy as Lagrangian Jacobian (should be better)
    disc_jac_approx = DisciplineJacApprox(adapter)
    outputs = outvars + constraints
    func_approx_jac = disc_jac_approx.compute_approx_jac(outputs, inputs)
    post_opt_analysis = adapter.post_optimal_analysis
    lagr_jac = post_opt_analysis.compute_lagrangian_jac(func_approx_jac, inputs)
    assert disc_jac_approx.check_jacobian(
        lagr_jac, outvars, inputs, adapter, threshold=lagrangian_threshold
    )


def test_adapter_jacobian():
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    struct_adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11", "g_1"], reset_x0_before_opt=True
    )
    check_adapter_jacobian(
        struct_adapter,
        ["x_shared"],
        objective_threshold=5e-2,
        lagrangian_threshold=5e-2,
    )

    # Minimization scenario
    prop_scenario = build_prop_scenario()
    prop_adapter = MDOScenarioAdapter(
        prop_scenario, ["x_shared"], ["y_34", "g_3"], reset_x0_before_opt=True
    )
    check_adapter_jacobian(
        prop_adapter,
        ["x_shared"],
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )


def test_add_outputs():
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    struct_adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11"], reset_x0_before_opt=True
    )
    struct_adapter.add_outputs(["g_1"])
    check_adapter_jacobian(
        struct_adapter,
        ["x_shared"],
        objective_threshold=5e-2,
        lagrangian_threshold=5e-2,
    )


def check_obj_scenario_adapter(
    scenario, outputs, minimize, objective_threshold, lagrangian_threshold
):
    dim = scenario.design_space.dimension
    problem = scenario.formulation.opt_problem
    objective = problem.objective
    outvars = objective.outvars
    problem.objective = MDOFunction(
        lambda _: 123.456,
        objective.name,
        MDOFunction.TYPE_OBJ,
        lambda _: zeros(dim),
        "123.456",
        objective.args,
        objective.dim,
        outvars,
    )
    adapter = MDOObjScenarioAdapter(scenario, ["x_shared"], outputs)

    adapter.execute()
    local_value = adapter.local_data[outvars[0]]
    assert (
        minimize
        and allclose(local_value, array(123.456))
        or allclose(local_value, array(-123.456))
    )

    check_adapter_jacobian(
        adapter, ["x_shared"], objective_threshold, lagrangian_threshold
    )


def test_obj_scenario_adapter():
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    check_obj_scenario_adapter(
        struct_scenario,
        ["y_11", "g_1"],
        minimize=False,
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )

    # Minimization scenario
    prop_scenario = build_prop_scenario()
    check_obj_scenario_adapter(
        prop_scenario,
        ["y_34", "g_3"],
        minimize=True,
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )


def test_lagrange_multipliers_outputs():
    """Test the output of Lagrange multipliers."""
    struct_scenario = build_struct_scenario()
    x1_low_mult_name = MDOScenarioAdapter.get_bnd_mult_name("x_1", False)
    x1_upp_mult_name = MDOScenarioAdapter.get_bnd_mult_name("x_1", True)
    g1_mult_name = MDOScenarioAdapter.get_cstr_mult_name("g_1")
    mult_names = [x1_low_mult_name, x1_upp_mult_name, g1_mult_name]
    # Check the absence of multipliers when not required
    adapter = MDOScenarioAdapter(struct_scenario, ["x_shared"], ["y_11", "g_1"])
    assert not adapter.is_all_outputs_existing(mult_names)
    # Check the multipliers when required
    adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11", "g_1"], output_multipliers=True
    )
    assert adapter.is_all_outputs_existing(mult_names)
    adapter.execute()
    problem = struct_scenario.formulation.opt_problem
    x_opt = problem.solution.x_opt
    obj_grad = problem.nonproc_objective.jac(x_opt)
    g1_jac = problem.nonproc_constraints[0].jac(x_opt)
    x1_low_mult, x1_upp_mult, g1_mult = adapter.get_outputs_by_name(mult_names)
    lagr_grad = obj_grad + matmul(g1_mult.T, g1_jac) - x1_low_mult + x1_upp_mult
    assert allclose(lagr_grad, zeros_like(lagr_grad))


@pytest.mark.parametrize("export_name", ["", "local_database"])
def test_keep_opt_history(tmp_wd, scenario, export_name):
    """Test the option that keeps the local history of sub optimizations, with and
    without the export option."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        keep_opt_history=True,
        opt_history_file_prefix=export_name,
    )
    adapter.execute()
    adapter.execute({"x_shared": adapter.default_inputs["x_shared"] + 1.0})

    assert len(adapter.databases) == 2

    for database in adapter.databases:
        assert isinstance(database, Database)
        assert len(database) > 2

    if export_name:
        path = Path(export_name)
        assert (path.parent / f"{path.name}_1.h5").exists()
        assert (path.parent / f"{path.name}_2.h5").exists()


@pytest.mark.parametrize("set_x0_before_opt", [True, False])
def test_scenario_adapter_serialization(tmp_wd, scenario, set_x0_before_opt):
    """Test that an MDOScenarioAdapter can be serialized, loaded and executed.

    The focus of this test is to guarantee that the loaded MDOChain instance can be
    executed, if an AttributeError is raised, it means that the attribute is missing in
    MDOScenarioAdapter._ATTR_TO_SERIALIZE.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        scenario: Fixture that returns a DOEScenario for the Sobieski's SSBJ use case
            without physical naming.
    """
    adapter = MDOScenarioAdapter(
        scenario, ["x_shared"], ["y_4"], set_x0_before_opt=set_x0_before_opt
    )

    with open("adapter.pkl", "wb") as file:
        pickle.dump(adapter, file)

    with open("adapter.pkl", "rb") as file:
        adapter = pickle.load(file)

    adapter.execute()
    assert adapter.scenario.optimization_result.is_feasible
