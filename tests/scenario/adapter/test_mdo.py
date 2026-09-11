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
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import all as np_all
from numpy import allclose
from numpy import array
from numpy import atleast_2d
from numpy import matmul
from numpy import ones
from numpy import ones_like
from numpy import zeros
from numpy import zeros_like

from gemseo import create_scenario
from gemseo.core.discipline import Discipline
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.discipline_adapter_generator import DisciplineAdapterGenerator
from gemseo.core.problem.database import Database
from gemseo.discipline.analytic import AnalyticDiscipline
from gemseo.discipline.chain.chain import DisciplineChain
from gemseo.discipline.chain.parallel_chain import ParallelDisciplineChain
from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.doe.scipy.settings.lhs import LHS_Settings
from gemseo.formulation.mdf_settings import MDF_Settings
from gemseo.optimization.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.optimization.nlopt.settings.nlopt_slsqp_settings import NLOPT_SLSQP_Settings
from gemseo.optimization.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.optimization.scipy_global.settings.shgo import SHGO_Settings
from gemseo.optimization.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.mdo.sobieski.discipline import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski.discipline import SobieskiMission
from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion
from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.problem.mdo.sobieski.standalone.design_space import SobieskiDesignSpace
from gemseo.scenario.adapter.mdo import MDOScenarioAdapter
from gemseo.scenario.mdo import MDOScenario
from gemseo.space.design import DesignSpace
from gemseo.util.derivative.derivatives_approx import DisciplineJacApprox
from gemseo.util.name_generator import NameGenerator
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import StrKeyMapping
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker


def create_design_space():
    """"""
    return SobieskiDesignSpace()


@pytest.fixture
def scenario():
    """An MDO scenario solving the Sobieski problem with MDF and L_BFGS_B."""
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
        design_space,
        name="MyScenario",
        formulation_settings=MDF_Settings(),
    )
    mdo_scenario.add_objective("y_4", minimize=False)
    mdo_scenario.set_algorithm(L_BFGS_B_Settings(max_iter=35))
    return mdo_scenario


def test_default_name(scenario) -> None:
    """Check the default name of an MDOScenarioAdapter."""
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])
    assert adapter.name == "MyScenario_adapter"


def test_name(scenario) -> None:
    """Check that the name of the MDOScenarioAdapter is correctly set."""
    name = "MyAdapter"
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"], name=name)
    assert adapter.name == name


def test_adapter(scenario) -> None:
    """Test the MDOAdapter."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs)
    gen = DisciplineAdapterGenerator(adapter)
    func = gen.get_function(inputs, outputs)
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    f_x1 = func.evaluate(x_shared)
    f_x2 = func.evaluate(x_shared)
    assert f_x1 == f_x2
    x_shared = array([0.09, 60000, 1.4, 2.5, 70, 1500])

    f_x3 = func.evaluate(x_shared)
    assert f_x3 > 4947.0


def test_adapter_set_x0_before_exec(scenario) -> None:
    """Test the MDOScenarioAdapter with set_x0_before_exec."""
    inputs = ["x_1", "x_2", "x_3", "x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_x0_before_exec=True)
    gen = DisciplineAdapterGenerator(adapter)
    x_shared = array([0.25, 1.0, 1.0, 0.5, 0.09, 60000, 1.4, 2.5, 70, 1500])
    func = gen.get_function(inputs, outputs)
    f_x3 = func.evaluate(x_shared)
    assert f_x3 > 4947.0


def test_adapter_set_and_reset_x0(scenario, snapshot) -> None:
    """Test that set and reset x_0 cannot be done at MDOScenarioAdapter
    instantiation."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    with assert_exception(ValueError, snapshot):
        MDOScenarioAdapter(
            scenario,
            inputs,
            outputs,
            set_x0_before_exec=True,
            reset_x0_before_exec=True,
        )


def test_adapter_miss_dvs(scenario) -> None:
    inputs = ["x_shared"]
    outputs = ["y_4", "missing_dv"]
    scenario.design_space.add_variable("missing_dv")
    MDOScenarioAdapter(scenario, inputs, outputs)


def test_adapter_reset_x0_before_exec(scenario) -> None:
    """Check MDOScenarioAdapter.reset_x0_before_exec()."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    design_space = scenario.design_space
    initial_design = design_space.convert_dict_to_array(
        design_space.get_current_value(as_dict=True)
    )
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, reset_x0_before_exec=True)
    adapter.execute()
    x_shared = adapter.io.input_grammar.defaults["x_shared"] * 1.01
    adapter.io.input_grammar.defaults["x_shared"] = x_shared
    # initial_x is reset to the initial design value before optimization;
    # thus the optimization starts from initial_design.
    adapter.execute()
    initial_x = adapter.scenario.formulation.problem.database.get_x_vect(1)
    assert np_all(initial_x == initial_design)

    adapter = MDOScenarioAdapter(scenario, inputs, outputs)
    adapter.execute()
    new_initial_design = design_space.convert_dict_to_array(
        design_space.get_current_value(as_dict=True)
    )
    adapter.io.input_grammar.defaults["x_shared"] = x_shared
    # initial_x is NOT reset to the initial design value before optimization;
    # thus the optimization starts from the last design value (=new_initial_design).
    adapter.execute()
    initial_x = adapter.scenario.formulation.problem.database.get_x_vect(1)
    assert np_all(initial_x == new_initial_design)
    assert not np_all(initial_x == initial_design)


def test_adapter_set_bounds(scenario) -> None:
    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_bounds_before_exec=True)

    # Execute the adapter with default bounds
    adapter.execute()
    ds = scenario.design_space
    assert np_all(ds.get_lower_bounds() == [0.1, 0.75, 0.75, 0.1])
    assert np_all(ds.get_upper_bounds() == [0.4, 1.25, 1.25, 1.0])

    # Execute the adapter with passed bounds
    input_data = {}
    lower_bounds = ds.convert_array_to_dict(zeros(4))
    lower_suffix = MDOScenarioAdapter.LOWER_BND_SUFFIX
    upper_bounds = ds.convert_array_to_dict(ones(4))
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


def test_chain(scenario) -> None:
    """"""
    mda = scenario.formulation.mda
    inputs = list(mda.io.input_grammar) + scenario.design_space.variable_names
    outputs = ["x_1", "x_2", "x_3"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs)

    # Allow re exec when DONE for the chain execution
    chain = DisciplineChain([mda, adapter, mda])

    # Sobieski Z opt
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    chain.execute({"x_shared": x_shared})

    y_4 = chain.io.output_data["y_4"]
    assert y_4 > 2908.0


def test_compute_jacobian(scenario) -> None:
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])
    adapter.execute()
    adapter._compute_jacobian()
    expected_output_names = {"y_4", "mult_dot_constr_jac"}

    assert set(adapter.jac.keys()) == expected_output_names

    for output_name in expected_output_names:
        assert set(adapter.jac[output_name].keys()) == {"x_shared"}


def test_compute_jacobian_with_bound_inputs(scenario) -> None:
    adapter = MDOScenarioAdapter(
        scenario, ["x_shared"], ["y_4"], set_bounds_before_exec=True
    )
    expected_input_names = ["x_shared", "x_1_lower_bnd"]
    adapter.execute()
    adapter._compute_jacobian(input_names=expected_input_names)
    expected_output_names = {"y_4", "mult_dot_constr_jac"}

    assert set(adapter.jac.keys()) == expected_output_names

    for output_name in expected_output_names:
        assert set(adapter.jac[output_name].keys()) == set(expected_input_names)


def test_compute_jacobian_exceptions(scenario, snapshot) -> None:
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])

    # Pass invalid inputs
    with assert_exception(ValueError, snapshot):
        adapter._compute_jacobian(input_names=["x_shared", "foo", "bar"])

    # Pass invalid outputs
    with assert_exception(ValueError, snapshot):
        adapter._compute_jacobian(output_names=["y_4", "foo", "bar"])

    # Pass invalid differentiated outputs
    scenario.add_constraint("g_1")
    scenario.add_constraint("g_2")
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4", "g_1", "g_2"])
    with assert_exception(ValueError, snapshot):
        adapter._compute_jacobian(output_names=["y_4", "g_2", "g_1"])

    # Pass a multi-valued objective
    scenario.formulation.problem.objective.output_names = ["y_4"] * 2
    scenario.formulation.problem.objective.dim = 2
    with assert_exception(ValueError, snapshot):
        adapter._compute_jacobian()


def build_struct_scenario():
    ds = SobieskiDesignSpace()
    sc_str = MDOScenario(
        [SobieskiStructure()], ds.filter("x_1", copy=True), name="StructureScenario"
    )
    sc_str.add_objective("y_11", minimize=False)
    sc_str.add_constraint("g_1", constraint_type=sc_str.ConstraintType.INEQ)
    sc_str.set_algorithm(NLOPT_SLSQP_Settings(max_iter=20))
    return sc_str


def build_prop_scenario():
    ds = SobieskiDesignSpace()
    sc_prop = MDOScenario(
        [SobieskiPropulsion()], ds.filter("x_3", copy=True), name="PropulsionScenario"
    )
    sc_prop.add_objective("y_34")
    sc_prop.add_constraint("g_3", constraint_type=sc_prop.ConstraintType.INEQ)
    sc_prop.set_algorithm(NLOPT_SLSQP_Settings(max_iter=20))
    return sc_prop


def check_adapter_jacobian(
    adapter, inputs, objective_threshold, lagrangian_threshold
) -> None:
    opt_problem = adapter.scenario.formulation.problem
    output_names = opt_problem.objective.output_names
    constraints = opt_problem.constraints.get_names()

    # Test the Jacobian accuracy as objective Jacobian
    checker = DisciplineJacobianChecker(adapter)
    assert checker.check(
        inputs=inputs,
        outputs=output_names,
        atol=objective_threshold,
        rtol=objective_threshold,
    )

    # Test the Jacobian accuracy as Lagrangian Jacobian (should be better).
    # The checker leaves the adapter linearized at the check point,
    # so its input data is already the one to differentiate at.
    disc_jac_approx = DisciplineJacApprox(adapter)
    outputs = output_names + constraints
    func_approx_jac = disc_jac_approx.compute_approx_jac(outputs, inputs)
    post_opt_analysis = adapter.post_optimal_analysis
    lagr_jac = post_opt_analysis.compute_lagrangian_jac(func_approx_jac, inputs)
    assert disc_jac_approx.check_jacobian(
        output_names,
        inputs,
        analytic_jacobian=lagr_jac,
        atol=lagrangian_threshold,
        rtol=lagrangian_threshold,
    )


def test_adapter_jacobian() -> None:
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    struct_adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11", "g_1"], reset_x0_before_exec=True
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
        prop_scenario, ["x_shared"], ["y_34", "g_3"], reset_x0_before_exec=True
    )
    check_adapter_jacobian(
        prop_adapter,
        ["x_shared"],
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )


def test_add_outputs() -> None:
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    struct_adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11"], reset_x0_before_exec=True
    )
    struct_adapter.add_outputs(["g_1"])
    check_adapter_jacobian(
        struct_adapter,
        ["x_shared"],
        objective_threshold=5e-2,
        lagrangian_threshold=5e-2,
    )


def replace_objective_by_constant(scenario) -> str:
    """Replace the objective of a scenario by the constant 123.456.

    The value of the objective is then decoupled from the output of the discipline
    that shares its name.

    Args:
        scenario: The scenario whose objective is to be replaced.

    Returns:
        The name of the output of the objective.
    """
    dim = scenario.design_space.dimension
    problem = scenario.formulation.problem
    objective = problem.objective
    problem.objective = ArrayFunction(
        lambda _: 123.456,
        name=objective.name,
        f_type=ArrayFunction.FunctionType.OBJ,
        jac=lambda _: zeros(dim),
        expr="123.456",
        input_names=objective.input_names,
        dim=objective.dim,
        output_names=objective.output_names,
    )
    return objective.output_names[0]


def check_optimal_objective_adapter(
    scenario, outputs, minimize, objective_threshold, lagrangian_threshold
) -> None:
    """Check the optimal objective output of a scenario adapter.

    Args:
        scenario: The scenario whose objective is to be replaced by a constant.
        outputs: The names of the outputs of the adapter.
        minimize: Whether the objective of the scenario is to be minimized.
        objective_threshold: The tolerance for the Jacobian of the objective.
        lagrangian_threshold: The tolerance for the Jacobian of the Lagrangian.
    """
    output_name = replace_objective_by_constant(scenario)
    adapter = MDOScenarioAdapter(
        scenario, ["x_shared"], outputs, output_optimal_objective=True
    )

    adapter.execute()
    local_value = adapter.io.output_data[output_name]
    assert (minimize and allclose(local_value, array(123.456))) or allclose(
        local_value, array(-123.456)
    )

    check_adapter_jacobian(
        adapter, ["x_shared"], objective_threshold, lagrangian_threshold
    )


def test_output_optimal_objective() -> None:
    # Maximization scenario
    struct_scenario = build_struct_scenario()
    check_optimal_objective_adapter(
        struct_scenario,
        ["y_11", "g_1"],
        minimize=False,
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )

    # Minimization scenario
    prop_scenario = build_prop_scenario()
    check_optimal_objective_adapter(
        prop_scenario,
        ["y_34", "g_3"],
        minimize=True,
        objective_threshold=1e-5,
        lagrangian_threshold=1e-5,
    )


def test_output_optimal_objective_is_opt_in() -> None:
    """Check that the objective output is that of the disciplines by default."""
    scenario = build_prop_scenario()
    output_name = replace_objective_by_constant(scenario)
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_34", "g_3"])

    adapter.execute()
    # The objective of the problem is the constant 123.456
    # but the discipline computing y_34 knows nothing about it.
    assert not allclose(adapter.io.output_data[output_name], array(123.456))


def test_instantiation_before_add_objective() -> None:
    """Check that the adapter can be built before the objective is set.

    The optimal objective value requires a single-valued objective,
    whose dimension cannot be determined while the objective is unset.
    """
    discipline = AnalyticDiscipline({"y": "x**2 + z"}, name="d")
    discipline.io.input_grammar.defaults["z"] = array([1.0])
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([discipline], design_space)
    adapter = MDOScenarioAdapter(scenario, ["z"], ["y"], output_optimal_objective=True)
    scenario.add_objective("y")
    scenario.set_algorithm(PYDOE_FULLFACT_Settings(n_samples=3))

    # The optimum is x=0, so the optimal objective value is z.
    assert allclose(adapter.execute({"z": array([2.0])})["y"], array([2.0]))


def test_multi_objective_exception(snapshot) -> None:
    """Check the error raised at instantiation for a multi-objective problem.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    discipline = AnalyticDiscipline({"y": "x**2 + z", "w": "10*x + z"}, name="d")
    discipline.io.input_grammar.defaults["z"] = array([1.0])
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    scenario = MDOScenario([discipline], design_space)
    scenario.add_objective(["y", "w"])
    with assert_exception(ValueError, snapshot):
        MDOScenarioAdapter(scenario, ["z"], ["y"], output_optimal_objective=True)


def test_retrieve_top_level_outputs_multi_objective_exception(
    scenario, snapshot
) -> None:
    """Check the error raised for the optimal objective of a multi-objective problem.

    The dimension of the objective may only be known once it has been evaluated;
    this residual case is the one this check is left for,
    as instantiation rejects a problem already known to be multi-objective.

    Args:
        scenario: A fixture returning an MDO scenario solving the Sobieski problem.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    adapter = MDOScenarioAdapter(
        scenario, ["x_shared"], ["y_4"], output_optimal_objective=True
    )
    adapter.execute()

    # Pass a multi-valued objective
    scenario.formulation.problem.objective.output_names = ["y_4"] * 2
    scenario.formulation.problem.objective.dim = 2
    with assert_exception(ValueError, snapshot):
        adapter._retrieve_top_level_outputs()


def test_lagrange_multipliers_outputs() -> None:
    """Test the output of Lagrange multipliers."""
    struct_scenario = build_struct_scenario()
    x1_low_mult_name = MDOScenarioAdapter.get_bnd_mult_name("x_1", False)
    x1_upp_mult_name = MDOScenarioAdapter.get_bnd_mult_name("x_1", True)
    g1_mult_name = MDOScenarioAdapter.get_cstr_mult_name("g_1")
    mult_names = [x1_low_mult_name, x1_upp_mult_name, g1_mult_name]
    # Check the absence of multipliers when not required
    adapter = MDOScenarioAdapter(struct_scenario, ["x_shared"], ["y_11", "g_1"])
    assert not adapter.io.output_grammar.has_names(mult_names)
    # Check the multipliers when required
    adapter = MDOScenarioAdapter(
        struct_scenario, ["x_shared"], ["y_11", "g_1"], output_multipliers=True
    )
    assert adapter.io.output_grammar.has_names(mult_names)
    adapter.execute()
    problem = struct_scenario.formulation.problem
    x_opt = problem.solution.x_opt
    obj_grad = problem.objective.original.jac(x_opt)
    g1_jac = next(problem.constraints.get_originals()).jac(x_opt)
    local_data = adapter.io.get_merged_data(as_dict=False)
    lagr_grad = (
        obj_grad
        + matmul(local_data[g1_mult_name].T, g1_jac)
        - local_data[x1_low_mult_name]
        + local_data[x1_upp_mult_name]
    )
    assert allclose(lagr_grad, zeros_like(lagr_grad))


@pytest.mark.parametrize("keep_databases", [True, False])
def test_keep_databases(tmp_wd, scenario, keep_databases) -> None:
    """Test the option that keeps the local history of sub optimizations."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        keep_databases=keep_databases,
    )
    adapter.execute()
    adapter.execute({"x_shared": adapter.io.input_grammar.defaults["x_shared"] + 1.0})

    assert len(adapter.databases) == (2 if keep_databases else 0)

    for database in adapter.databases:
        assert isinstance(database, Database)
        assert len(database) > 2


@pytest.mark.parametrize(
    ("save_databases", "database_file_prefix"),
    [(True, "local_database"), (True, ""), (False, "local_database"), (False, "")],
)
def test_save_databases(tmp_wd, scenario, save_databases, database_file_prefix) -> None:
    """Test the option that saves the local history of sub optimizations, with and
    without the file prefix."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        save_databases=save_databases,
        database_file_prefix=database_file_prefix,
    )
    adapter.execute()
    adapter.execute({"x_shared": adapter.io.input_grammar.defaults["x_shared"] + 1.0})

    path = Path(database_file_prefix)
    if database_file_prefix:
        prefix = path.name
    else:
        prefix = MDOScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX

    assert (path.parent / f"{prefix}_1.h5").exists() is save_databases
    assert (path.parent / f"{prefix}_2.h5").exists() is save_databases


@pytest.mark.parametrize("set_x0_before_exec", [True, False])
def test_scenario_adapter_serialization(tmp_wd, scenario, set_x0_before_exec) -> None:
    """Test that an MDOScenarioAdapter can be serialized, loaded and executed.

    The focus of this test is to guarantee
    that the loaded DisciplineChain instance can be executed,
    if an AttributeError is raised, it means that the attribute is missing in
    `MDOScenarioAdapter._ATTR_NOT_TO_SERIALIZE`.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        scenario: Fixture that returns n MDOScenario for the Sobieski's SSBJ use case
            without physical naming.
    """
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        set_x0_before_exec=set_x0_before_exec,
        keep_databases=True,
        database_file_prefix="test",
    )

    with open("adapter.pkl", "wb") as file:
        pickle.dump(adapter, file)

    with open("adapter.pkl", "rb") as file:
        adapter = pickle.load(file)

    adapter.execute()
    assert adapter.scenario.optimization_result.is_feasible


def test_parallel_adapter(tmp_wd, scenario):
    """Test the execution of an MDOScenarioAdapter in multiprocessing."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        keep_databases=True,
        save_databases=True,
        database_file_prefix="test",
        naming=NameGenerator.Naming.UUID,
    )
    design_space = SobieskiDesignSpace()
    design_space.filter(["x_shared"])
    mdo_scenario = MDOScenario([adapter], design_space)
    mdo_scenario.add_objective("y_4", minimize=False)
    mdo_scenario.execute(LHS_Settings(n_samples=10, n_processes=2))
    assert len(list(tmp_wd.rglob("test_*.h5"))) == 10


class SampleWiseDiscipline(Discipline):
    """A discipline computing y=z-x and w=10x+z for one or more design points."""

    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__(name="d")
        self.io.input_grammar.update_from_names(["x", "z"])
        self.io.output_grammar.update_from_names(["y", "w"])
        self.io.input_grammar.defaults["z"] = array([1.0])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        z = input_data["z"]
        return {"y": z - x, "w": 10 * x + z}


@pytest.mark.parametrize(
    "algorithm_settings",
    [
        PYDOE_FULLFACT_Settings(n_samples=3),
        PYDOE_FULLFACT_Settings(n_samples=3, n_processes=2),
        PYDOE_FULLFACT_Settings(n_samples=3, vectorize=True),
    ],
    ids=["serial", "parallel", "vectorized"],
)
def test_optimum_evaluated_after_doe(algorithm_settings) -> None:
    """Check the outputs of the adapter when the optimum is the last sample of a DOE.

    Args:
        algorithm_settings: The settings of the DOE algorithm of the scenario.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.0)
    scenario = MDOScenario([SampleWiseDiscipline()], design_space)
    scenario.add_objective("y")
    scenario.set_algorithm(algorithm_settings)
    adapter = MDOScenarioAdapter(scenario, ["z"], ["y", "w"])

    output_data = adapter.execute({"z": array([2.0])})
    # The optimum is the last sample, i.e. x=1,
    # but a parallel DOE evaluates the samples in sub-processes
    # and a vectorized one evaluates the whole sample set at once,
    # so the disciplines of this process hold no data for that single design point.
    assert allclose(output_data["y"], array([1.0]))
    assert allclose(output_data["w"], array([12.0]))


@pytest.mark.parametrize(
    ("algorithm_settings", "expected"),
    [
        (PYDOE_FULLFACT_Settings(n_samples=3), False),
        (PYDOE_FULLFACT_Settings(n_samples=3, n_processes=2), True),
        (PYDOE_FULLFACT_Settings(n_samples=3, vectorize=True), True),
        (MultiStart_Settings(n_processes=2), True),
        (DIFFERENTIAL_EVOLUTION_Settings(), False),
        (DIFFERENTIAL_EVOLUTION_Settings(workers=2), True),
        (DIFFERENTIAL_EVOLUTION_Settings(workers=-1), True),
        (SHGO_Settings(workers=2), True),
    ],
    ids=[
        "doe",
        "parallel_doe",
        "vectorized_doe",
        "parallel_multi_start",
        "global_optimizer",
        "parallel_global_optimizer",
        "global_optimizer_using_every_core",
        "parallel_shgo",
    ],
)
def test_is_last_evaluation_unusable(algorithm_settings, expected) -> None:
    """Check the drivers whose last evaluation leaves no data in the disciplines.

    Args:
        algorithm_settings: The settings of the algorithm of the scenario.
        expected: Whether the disciplines of this process are expected
            not to hold the data of the last design point evaluated by the scenario.
    """
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.0)
    scenario = MDOScenario([SampleWiseDiscipline()], design_space)
    scenario.add_objective("y")
    scenario.set_algorithm(algorithm_settings)
    adapter = MDOScenarioAdapter(scenario, ["z"], ["y", "w"])

    assert adapter._MDOScenarioAdapter__is_last_evaluation_unusable() is expected


class DisciplineMain(Discipline):
    """Discipline that takes as inputs alpha and computes beta=2*alpha.

    Jacobians are computed in _run method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["alpha"])
        self.io.output_grammar.update_from_names(["beta"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        alpha = input_data["alpha"]
        self.io.output_data["beta"] = 2.0 * alpha
        self._has_jacobian = True
        self.jac = {"beta": {"alpha": 2.0 * atleast_2d(ones_like(alpha))}}


class DisciplineMainWithJacobian(Discipline):
    """Discipline that takes as inputs alpha and computes beta=2*alpha.

    Jacobian are computed _compute_jacobian method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["alpha"])
        self.io.output_grammar.update_from_names(["beta"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        alpha = input_data["alpha"]
        self.io.output_data["beta"] = 2.0 * alpha

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        alpha = self.io.input_data["alpha"]
        self.io.output_data["beta"] = 2.0 * alpha
        self.jac = {"beta": {"alpha": 2.0 * atleast_2d(ones_like(alpha))}}


class DisciplineSub1(Discipline):
    """Discipline that takes as inputs x and computes f=3*x."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x"])
        self.io.output_grammar.update_from_names(["f"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        self.io.output_data["f"] = 3.0 * x
        self._has_jacobian = True
        self.jac = {"f": {"x": 3.0 * atleast_2d(ones_like(x))}}


class DisciplineSub2(Discipline):
    """Discipline that takes x and beta and compute g=x+beta."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x", "beta"])
        self.io.output_grammar.update_from_names(["g"])
        self.io.input_grammar.defaults = {"x": array([0.0]), "beta": array([0.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        beta = input_data["beta"]
        self.io.output_data["g"] = x + beta
        self._has_jacobian = True
        self.jac = {
            "g": {"x": atleast_2d(ones_like(x)), "beta": atleast_2d(ones_like(beta))}
        }


@pytest.fixture(
    params=[
        [DisciplineMain(), DisciplineSub1(), DisciplineSub2()],
        [DisciplineMainWithJacobian(), DisciplineSub1(), DisciplineSub2()],
        [DisciplineChain([DisciplineMain(), DisciplineSub1(), DisciplineSub2()])],
        [
            DisciplineChain([
                DisciplineMainWithJacobian(),
                DisciplineSub1(),
                DisciplineSub2(),
            ])
        ],
        [
            DisciplineMain(),
            ParallelDisciplineChain([DisciplineSub1(), DisciplineSub2()]),
        ],
    ]
)
def disciplines_fixture(request):
    """Disciplines to be used in the scenario adapter."""
    return request.param


@pytest.fixture
def scenario_fixture(disciplines_fixture):
    """Fixture generating a discipline depending only on main design variable.

    This discipline is linerarized in the _run.
    """
    design_space = create_design_space()
    design_space.add_variable(
        "x", lower_bound=-1.5, upper_bound=1.5, value=array([1.0]), size=1
    )
    scenario = create_scenario(
        disciplines_fixture,
        "f",
        design_space,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint(
        "g", constraint_type=ArrayFunction.ConstraintType.INEQ, value=5
    )
    scenario.set_algorithm(SLSQP_Settings(max_iter=10))
    return MDOScenarioAdapter(scenario, ["alpha"], ["f"], set_x0_before_exec=True)


def test_scenario_adapter(scenario_fixture) -> None:
    """Test the scenario execution."""
    design_space = create_design_space()
    design_space.add_variable(
        "alpha", lower_bound=-1.5, upper_bound=1.5, value=array([1.0]), size=1
    )
    scenario = create_scenario(
        [scenario_fixture],
        "f",
        design_space,
        formulation_name="DisciplinaryOpt",
    )
    scenario.set_algorithm(SLSQP_Settings(max_iter=10))
    scenario.execute()
    assert scenario.formulation.problem.solution is not None


def test_run_scenario_adapter(scenario_fixture) -> None:
    """Test te execution of the scenario adapter."""
    out = scenario_fixture.execute({"alpha": array([0.0])})
    assert "f" in out


def test_linearize_scenario_adapter(scenario_fixture) -> None:
    """Test the linearization of the scenario adapter."""
    out = scenario_fixture.linearize(
        {"alpha": array([0.0])}, compute_all_jacobians=True
    )
    assert "f" in out


def test_multiple_linearize() -> None:
    """Tests two linearizations and linearize in the _run method."""
    disc2 = DisciplineChain([DisciplineMain(), DisciplineSub1(), DisciplineSub2()])
    disc2.io.input_grammar.defaults = {"x": array([0.0]), "alpha": array([0.0])}
    disc2.add_differentiated_inputs("x")
    disc2.add_differentiated_outputs("g")
    disc2.linearize()
    disc2._differentiated_input_names = []
    disc2._differentiated_output_names = []
    disc2.linearize()
    assert "g" in disc2.jac
    disc2._differentiated_input_names = ["alpha"]
    disc2._differentiated_output_names = ["g"]
    disc2.linearize()
    assert "g" in disc2.jac
    assert "alpha" in disc2.jac["g"]
    assert "x" not in disc2.jac["g"]


def test_mdo_adapter_of_evaluation_scenario(evaluation_scenario, snapshot) -> None:
    """Check the error raised by an MDO adapter without an optimization problem.

    Args:
        evaluation_scenario: Fixture that returns an evaluation scenario.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with assert_exception(TypeError, snapshot):
        MDOScenarioAdapter(evaluation_scenario, input_names=["z"], output_names=["y"])
