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
import re
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
from gemseo.algos.database import Database
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.disciplines.scenario_adapters.mdo_objective_scenario_adapter import (
    MDOObjectiveScenarioAdapter,
)
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.derivatives.derivatives_approx import DisciplineJacApprox
from gemseo.utils.name_generator import NameGenerator

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


def create_design_space():
    """"""
    return SobieskiDesignSpace()


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
        "y_4",
        design_space,
        formulation_name="MDF",
        maximize_objective=True,
        name="MyScenario",
    )
    mdo_scenario.set_algorithm(algo_name="L-BFGS-B", max_iter=35)
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


def test_adapter_set_x0_before_opt(scenario) -> None:
    """Test the MDOScenarioAdapter with set_x0_before_opt."""
    inputs = ["x_1", "x_2", "x_3", "x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_x0_before_opt=True)
    gen = DisciplineAdapterGenerator(adapter)
    x_shared = array([0.25, 1.0, 1.0, 0.5, 0.09, 60000, 1.4, 2.5, 70, 1500])
    func = gen.get_function(inputs, outputs)
    f_x3 = func.evaluate(x_shared)
    assert f_x3 > 4947.0


def test_adapter_set_and_reset_x0(scenario) -> None:
    """Test that set and reset x_0 cannot be done at MDOScenarioAdapter
    instantiation."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    msg = "Inconsistent options for MDOScenarioAdapter."
    with pytest.raises(ValueError, match=msg):
        MDOScenarioAdapter(
            scenario, inputs, outputs, set_x0_before_opt=True, reset_x0_before_opt=True
        )


def test_adapter_miss_dvs(scenario) -> None:
    inputs = ["x_shared"]
    outputs = ["y_4", "missing_dv"]
    scenario.design_space.add_variable("missing_dv")
    MDOScenarioAdapter(scenario, inputs, outputs)


def test_adapter_reset_x0_before_opt(scenario) -> None:
    """Check MDOScenarioAdapter.reset_x0_before_opt()."""
    inputs = ["x_shared"]
    outputs = ["y_4"]
    design_space = scenario.design_space
    initial_design = design_space.convert_dict_to_array(
        design_space.get_current_value(as_dict=True)
    )
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, reset_x0_before_opt=True)
    adapter.execute()
    x_shared = adapter.io.input_grammar.defaults["x_shared"] * 1.01
    adapter.io.input_grammar.defaults["x_shared"] = x_shared
    # initial_x is reset to the initial design value before optimization;
    # thus the optimization starts from initial_design.
    adapter.execute()
    initial_x = adapter.scenario.formulation.optimization_problem.database.get_x_vect(1)
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
    initial_x = adapter.scenario.formulation.optimization_problem.database.get_x_vect(1)
    assert np_all(initial_x == new_initial_design)
    assert not np_all(initial_x == initial_design)


def test_adapter_set_bounds(scenario) -> None:
    inputs = ["x_shared"]
    outputs = ["y_4"]
    adapter = MDOScenarioAdapter(scenario, inputs, outputs, set_bounds_before_opt=True)

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
    chain = MDOChain([mda, adapter, mda])

    # Sobieski Z opt
    x_shared = array([0.06000319728113519, 60000, 1.4, 2.5, 70, 1500])
    chain.execute({"x_shared": x_shared})

    y_4 = chain.io.data["y_4"]
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
        scenario, ["x_shared"], ["y_4"], set_bounds_before_opt=True
    )
    expected_input_names = ["x_shared", "x_1_lower_bnd"]
    adapter.execute()
    adapter._compute_jacobian(input_names=expected_input_names)
    expected_output_names = {"y_4", "mult_dot_constr_jac"}

    assert set(adapter.jac.keys()) == expected_output_names

    for output_name in expected_output_names:
        assert set(adapter.jac[output_name].keys()) == set(expected_input_names)


def test_compute_jacobian_exceptions(scenario) -> None:
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4"])

    # Pass invalid inputs
    with pytest.raises(
        ValueError,
        match=re.escape("The following are not inputs of the adapter: bar, foo."),
    ):
        adapter._compute_jacobian(input_names=["x_shared", "foo", "bar"])

    # Pass invalid outputs
    with pytest.raises(
        ValueError,
        match=re.escape("The following are not outputs of the adapter: bar, foo."),
    ):
        adapter._compute_jacobian(output_names=["y_4", "foo", "bar"])

    # Pass invalid differentiated outputs
    scenario.add_constraint(["g_1"])
    scenario.add_constraint(["g_2"])
    adapter = MDOScenarioAdapter(scenario, ["x_shared"], ["y_4", "g_1", "g_2"])
    with pytest.raises(
        ValueError,
        match=re.escape("Post-optimal Jacobians of g_1, g_2 cannot be computed."),
    ):
        adapter._compute_jacobian(output_names=["y_4", "g_2", "g_1"])

    # Pass a multi-valued objective
    scenario.formulation.optimization_problem.objective.output_names = ["y_4"] * 2
    with pytest.raises(
        ValueError, match=re.escape("The objective must be single-valued.")
    ):
        adapter._compute_jacobian()


def build_struct_scenario():
    ds = SobieskiDesignSpace()
    sc_str = MDOScenario(
        [SobieskiStructure()],
        "y_11",
        ds.filter("x_1", copy=True),
        name="StructureScenario",
        formulation_name="DisciplinaryOpt",
        maximize_objective=True,
    )
    sc_str.add_constraint("g_1", constraint_type="ineq")
    sc_str.set_algorithm(algo_name="NLOPT_SLSQP", max_iter=20)
    return sc_str


def build_prop_scenario():
    ds = SobieskiDesignSpace()
    sc_prop = MDOScenario(
        [SobieskiPropulsion()],
        "y_34",
        ds.filter("x_3", copy=True),
        formulation_name="DisciplinaryOpt",
        name="PropulsionScenario",
    )
    sc_prop.add_constraint("g_3", constraint_type="ineq")
    sc_prop.set_algorithm(algo_name="NLOPT_SLSQP", max_iter=20)
    return sc_prop


def check_adapter_jacobian(
    adapter, inputs, objective_threshold, lagrangian_threshold
) -> None:
    opt_problem = adapter.scenario.formulation.optimization_problem
    output_names = opt_problem.objective.output_names
    constraints = opt_problem.constraints.get_names()

    # Test the Jacobian accuracy as objective Jacobian
    assert adapter.check_jacobian(
        input_names=inputs, output_names=output_names, threshold=objective_threshold
    )

    # Test the Jacobian accuracy as Lagrangian Jacobian (should be better)
    disc_jac_approx = DisciplineJacApprox(adapter)
    outputs = output_names + constraints
    func_approx_jac = disc_jac_approx.compute_approx_jac(outputs, inputs)
    post_opt_analysis = adapter.post_optimal_analysis
    lagr_jac = post_opt_analysis.compute_lagrangian_jac(func_approx_jac, inputs)
    assert disc_jac_approx.check_jacobian(
        output_names, inputs, analytic_jacobian=lagr_jac, threshold=lagrangian_threshold
    )


def test_adapter_jacobian() -> None:
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


def test_add_outputs() -> None:
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
) -> None:
    dim = scenario.design_space.dimension
    problem = scenario.formulation.optimization_problem
    objective = problem.objective
    output_names = objective.output_names
    problem.objective = MDOFunction(
        lambda _: 123.456,
        objective.name,
        MDOFunction.FunctionType.OBJ,
        lambda _: zeros(dim),
        "123.456",
        objective.input_names,
        objective.dim,
        output_names,
    )
    adapter = MDOObjectiveScenarioAdapter(scenario, ["x_shared"], outputs)

    adapter.execute()
    local_value = adapter.io.data[output_names[0]]
    assert (minimize and allclose(local_value, array(123.456))) or allclose(
        local_value, array(-123.456)
    )

    check_adapter_jacobian(
        adapter, ["x_shared"], objective_threshold, lagrangian_threshold
    )


def test_obj_scenario_adapter() -> None:
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
    problem = struct_scenario.formulation.optimization_problem
    x_opt = problem.solution.x_opt
    obj_grad = problem.objective.original.jac(x_opt)
    g1_jac = next(problem.constraints.get_originals()).jac(x_opt)
    local_data = adapter.io.data
    lagr_grad = (
        obj_grad
        + matmul(local_data[g1_mult_name].T, g1_jac)
        - local_data[x1_low_mult_name]
        + local_data[x1_upp_mult_name]
    )
    assert allclose(lagr_grad, zeros_like(lagr_grad))


@pytest.mark.parametrize("keep_opt_history", [True, False])
def test_keep_opt_history(tmp_wd, scenario, keep_opt_history) -> None:
    """Test the option that keeps the local history of sub optimizations."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        keep_opt_history=keep_opt_history,
    )
    adapter.execute()
    adapter.execute({"x_shared": adapter.io.input_grammar.defaults["x_shared"] + 1.0})

    assert len(adapter.databases) == (2 if keep_opt_history else 0)

    for database in adapter.databases:
        assert isinstance(database, Database)
        assert len(database) > 2


@pytest.mark.parametrize(
    ("save_opt_history", "opt_history_file_prefix"),
    [(True, "local_database"), (True, ""), (False, "local_database"), (False, "")],
)
def test_save_opt_history(
    tmp_wd, scenario, save_opt_history, opt_history_file_prefix
) -> None:
    """Test the option that saves the local history of sub optimizations, with and
    without the file prefix."""
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        save_opt_history=save_opt_history,
        opt_history_file_prefix=opt_history_file_prefix,
    )
    adapter.execute()
    adapter.execute({"x_shared": adapter.io.input_grammar.defaults["x_shared"] + 1.0})

    path = Path(opt_history_file_prefix)
    if opt_history_file_prefix:
        prefix = path.name
    else:
        prefix = MDOScenarioAdapter.DEFAULT_DATABASE_FILE_PREFIX

    assert (path.parent / f"{prefix}_1.h5").exists() is save_opt_history
    assert (path.parent / f"{prefix}_2.h5").exists() is save_opt_history


@pytest.mark.parametrize("set_x0_before_opt", [True, False])
def test_scenario_adapter_serialization(tmp_wd, scenario, set_x0_before_opt) -> None:
    """Test that an MDOScenarioAdapter can be serialized, loaded and executed.

    The focus of this test is to guarantee that the loaded MDOChain instance can be
    executed, if an AttributeError is raised, it means that the attribute is missing in
    ``MDOScenarioAdapter._ATTR_NOT_TO_SERIALIZE``.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        scenario: Fixture that returns a DOEScenario for the Sobieski's SSBJ use case
            without physical naming.
    """
    adapter = MDOScenarioAdapter(
        scenario,
        ["x_shared"],
        ["y_4"],
        set_x0_before_opt=set_x0_before_opt,
        keep_opt_history=True,
        opt_history_file_prefix="test",
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
        keep_opt_history=True,
        save_opt_history=True,
        opt_history_file_prefix="test",
        naming=NameGenerator.Naming.UUID,
    )
    design_space = SobieskiDesignSpace()
    design_space.filter(["x_shared"])
    scenario_doe = DOEScenario(
        [adapter],
        "y_4",
        design_space=design_space,
        maximize_objective=True,
        formulation_name="DisciplinaryOpt",
    )
    scenario_doe.execute(LHS_Settings(n_samples=10, n_processes=2))
    assert len(list(tmp_wd.rglob("test_*.h5"))) == 10


class DisciplineMain(Discipline):
    """Discipline that takes as inputs alpha and computes beta=2*alpha.

    Jacobians are computed in _run method.
    """

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["alpha"])
        self.io.output_grammar.update_from_names(["beta"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        alpha = self.io.data["alpha"]
        self.io.data["beta"] = 2.0 * alpha
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
        alpha = self.io.data["alpha"]
        self.io.data["beta"] = 2.0 * alpha

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        alpha = self.io.data["alpha"]
        self.io.data["beta"] = 2.0 * alpha
        self.jac = {"beta": {"alpha": 2.0 * atleast_2d(ones_like(alpha))}}


class DisciplineSub1(Discipline):
    """Discipline that takes as inputs x and computes f=3*x."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x"])
        self.io.output_grammar.update_from_names(["f"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = self.io.data["x"]
        self.io.data["f"] = 3.0 * x
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
        x = self.io.data["x"]
        beta = self.io.data["beta"]
        self.io.data["g"] = x + beta
        self._has_jacobian = True
        self.jac = {
            "g": {"x": atleast_2d(ones_like(x)), "beta": atleast_2d(ones_like(beta))}
        }


@pytest.fixture(
    params=[
        [DisciplineMain(), DisciplineSub1(), DisciplineSub2()],
        [DisciplineMainWithJacobian(), DisciplineSub1(), DisciplineSub2()],
        [MDOChain([DisciplineMain(), DisciplineSub1(), DisciplineSub2()])],
        [MDOChain([DisciplineMainWithJacobian(), DisciplineSub1(), DisciplineSub2()])],
        [DisciplineMain(), MDOParallelChain([DisciplineSub1(), DisciplineSub2()])],
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
    scenario.add_constraint("g", MDOFunction.ConstraintType.INEQ, value=5)
    scenario.set_algorithm(algo_name="SLSQP", max_iter=10)
    return MDOScenarioAdapter(scenario, ["alpha"], ["f"], set_x0_before_opt=True)


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
    scenario.set_algorithm(algo_name="SLSQP", max_iter=10)
    scenario.execute()
    assert scenario.formulation.optimization_problem.solution is not None


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
    disc2 = MDOChain([DisciplineMain(), DisciplineSub1(), DisciplineSub2()])
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
