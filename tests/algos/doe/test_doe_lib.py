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
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from sys import platform
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import inf
from numpy import ndarray
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_equal

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import execute_algo
from gemseo.algos.database import Database
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.doe.pydoe.pydoe import PyDOELibrary
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.discipline import DummyDiscipline

if TYPE_CHECKING:
    from gemseo.scenarios.doe_scenario import DOEScenario
    from gemseo.typing import StrKeyMapping

FACTORY = DOELibraryFactory()


@pytest.fixture(scope="module")
def lhs():
    """A PyDOE-based LHS."""
    return PyDOELibrary("PYDOE_LHS")


@pytest.fixture(scope="module")
def fullfact():
    """A PyDOE-based full-factorial DOE."""
    return PyDOELibrary("PYDOE_FULLFACT")


@pytest.fixture(scope="module")
def mc():
    """A SciPyDOE-based Monte Carlo DOE."""
    return SciPyDOE("MC")


@pytest.fixture(scope="module")
def custom_doe():
    """A custom DOE."""
    return CustomDOE()


def test_fail_sample(lhs) -> None:
    problem = Power2(exception_error=True)
    lhs.execute(problem, n_samples=4)


def test_evaluate_samples(fullfact) -> None:
    problem = Power2()
    fullfact.execute(problem, n_samples=2, wait_time_between_samples=1)


def test_evaluate_samples_multiproc(fullfact) -> None:
    problem = Power2()
    n_samples = 8
    fullfact.execute(
        problem,
        n_samples=n_samples,
        n_processes=2,
        wait_time_between_samples=0.01,
        eval_jac=True,
    )
    new_pb = Power2()
    x_history = problem.database.get_x_vect_history()
    assert len(x_history) == n_samples
    for sample in x_history:
        val_ref = new_pb.objective.evaluate(sample)
        val_sample = problem.database.get_function_value("pow2", sample)
        assert val_ref == val_sample

        grad_ref = new_pb.objective.jac(sample)
        grad_sample = problem.database.get_function_value("@pow2", sample)
        assert (grad_ref == grad_sample).all()


def compute_obj_and_obs(x: float = 0.0) -> tuple[float, float]:
    """Compute the objective and observable variables.

    Args:
         x: The input x value.

    Returns:
        obj: The objective value.
        obs: The observable value.
    """
    obj = x
    obs = x + 1.0
    return obj, obs


def test_evaluate_samples_multiproc_with_observables() -> None:
    """Evaluate a DoE in // with multiprocessing and with observables."""
    disc = create_discipline("AutoPyDiscipline", py_func=compute_obj_and_obs)
    disc.cache = None
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=3.0, value=2.0)

    scenario = create_scenario(
        [disc],
        "obj",
        design_space,
        scenario_type="DOE",
        formulation_name="DisciplinaryOpt",
    )

    samples = array([[float(i)] for i in range(4)])
    scenario.add_observable("obs")
    scenario.execute(algo_name="CustomDOE", n_processes=2, samples=samples)

    database = scenario.formulation.optimization_problem.database
    for i, (x, data) in enumerate(database.items()):
        assert x.wrapped_array[0] == pytest.approx(float(i))
        assert data["obj"] == float(i)
        assert data["obs"] == float(i + 1)

    # In multi-processing mode,
    # the disciplinary calls are only made on the worker processes
    # Under Linux, the counters are updated from the subprocesses counters,
    # Under Windows, the discipline counters on the main process are not updated.
    # Without leveraging the cache mechanism,
    # the discipline shall be called 8 times.
    if platform == "win32":
        assert disc.execution_statistics.n_executions == 0
    else:
        assert disc.execution_statistics.n_executions == 8


@pytest.fixture(scope="module")
def variables_space():
    """A mock design space."""
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=2.0, value=1.0)
    design_space.add_variable("y", lower_bound=-1.0, upper_bound=1.0, value=0.0)
    return design_space


@pytest.mark.parametrize(
    "settings",
    [{"n_samples": 4}, {"settings_model": PYDOE_FULLFACT_Settings(n_samples=4)}],
)
@pytest.mark.parametrize(
    ("transformation", "expected_points"),
    [
        ({"unit_sampling": True}, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        ({}, [[0.0, -1.0], [2.0, -1.0], [0.0, 1.0], [2.0, 1.0]]),
    ],
)
def test_compute_doe_from_space(
    fullfact, variables_space, settings, transformation, expected_points
) -> None:
    """Check the computation of a DOE in a variables space."""
    points = fullfact.compute_doe(variables_space, **settings, **transformation)
    assert (points == array(expected_points)).all()


SAMPLES = array([[0.0, 0.2, 0.3], [0.4, 0.5, 0.6]])


@pytest.mark.parametrize(
    "settings",
    [{"samples": SAMPLES}, {"settings_model": CustomDOE_Settings(samples=SAMPLES)}],
)
def test_compute_doe_from_dimension(custom_doe, settings):
    """Check BaseDOELibrary.compute_doe from the dimension of the variables space."""
    assert_equal(custom_doe.compute_doe(3, **settings), SAMPLES)


@pytest.fixture(scope="module")
def doe_database(request) -> Database:
    """The DOE-based database with either deterministic or random variables."""
    if request.param:
        space = ParameterSpace()
        space.add_random_variable("var", "OTNormalDistribution")
    else:
        space = DesignSpace()
        space.add_variable("var", lower_bound=-3.0, upper_bound=4.0, value=1.0)

    problem = OptimizationProblem(space)
    problem.objective = MDOFunction(lambda x: x, "func")
    execute_algo(
        problem,
        algo_name="CustomDOE",
        samples=array([[-2.0], [0.0], [1.0]]),
        algo_type="doe",
    )
    return problem.database


@pytest.mark.parametrize("doe_database", [True, False], indirect=["doe_database"])
@pytest.mark.parametrize("var", [-2.0, 0.0, 1.0])
def test_transformation(doe_database, var) -> None:
    """Check that the transformation of variables works correctly.

    For the deterministic variables, the transformation is affine, based on the bounds
    of the variables.

    For the uncertain variables, the transformation is probabilistic, based on inverse
    transformation sampling.
    """
    # We test with a certain tolerance
    # due to the sensitivity of probabilistic normalization
    # in the case var=-2 when updating openturns to v1.24
    assert_almost_equal(
        doe_database.get_function_value("func", array([var]), tolerance=1e-15),
        array([var]),
        decimal=15,
    )


def test_pre_run_debug(lhs, caplog) -> None:
    """Check a DEBUG message logged just after sampling the input unit hypercube."""
    caplog.set_level("DEBUG")
    problem = Power2()
    lhs.execute(problem, n_samples=2)
    message = (
        "The DOE algorithm PYDOE_LHS of PyDOELibrary has generated 2 samples "
        "in the input unit hypercube of dimension 3."
    )
    message_is_logged = False
    for _, log_level, log_message in caplog.record_tuples:
        if message in log_message:
            message_is_logged = True
            assert log_level == logging.DEBUG
            break

    assert message_is_logged


@pytest.mark.parametrize("algo_name", ["OT_MONTE_CARLO", "PYDOE_LHS"])
def test_seed(algo_name) -> None:
    """Check the use of the seed at the BaseDOELibrary level."""
    problem = Power2()
    library = DOELibraryFactory().create(algo_name)

    # The BaseDOELibrary has a seed and increments it
    # at the beginning of each execution.
    assert library.seed == 0
    library.execute(problem, n_samples=2)
    assert library.seed == 1
    assert len(problem.database) == 2

    # We execute a second time, still with the seed of the BaseDOELibrary.
    # For that,
    # we need to reset the current iteration because max_iter is reached
    # (for BaseDOELibrary, max_iter == n_samples).
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    library.execute(problem, n_samples=2)
    assert library.seed == 2
    assert len(problem.database) == 4

    # We execute a third time,
    # with a seed passed as an option of the BaseDOELibrary
    # and equal to the previous one.
    # By doing so,
    # the input samples will be the same and the functions won't be evaluated.
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    if algo_name == "PYDOE_LHS":
        library.execute(problem, n_samples=2, random_state=2)
    else:
        library.execute(problem, n_samples=2, seed=2)

    assert library.seed == 3
    # There is no new evaluation in the database:
    assert len(problem.database) == 4

    # Lastly, we check that the BaseDOELibrary uses its own seed again.
    problem.reset(
        database=False, design_space=False, function_calls=False, preprocessing=False
    )
    library.execute(problem, n_samples=2)
    assert library.seed == 4
    # There are new evaluations in the database:
    assert len(problem.database) == 6


@pytest.mark.parametrize(
    ("var_type1", "var_type2"),
    [
        ("integer", "integer"),
        ("integer", "float"),
        ("float", "float"),
    ],
)
def test_variable_types(var_type1, var_type2) -> None:
    """Verify that input data provided to a discipline match the design space types."""
    design_variable_type_to_python_type = DesignSpace.VARIABLE_TYPES_TO_DTYPES

    class Disc(DummyDiscipline):
        def __init__(self) -> None:
            super().__init__("foo")
            self.io.input_grammar.update_from_names(("x", "y"))
            self.io.output_grammar.update_from_names(("z",))

        def execute(self, input_data):
            assert (
                input_data["x"].dtype == design_variable_type_to_python_type[var_type1]
            )
            assert (
                input_data["y"].dtype == design_variable_type_to_python_type[var_type2]
            )
            return {"z": 0.0}

    design_space = DesignSpace()
    design_space.add_variable(
        "x", lower_bound=0, upper_bound=1, value=0, type_=var_type1
    )
    design_space.add_variable(
        "y", lower_bound=0, upper_bound=1, value=0, type_=var_type2
    )

    scenario = create_scenario(
        [Disc()],
        "z",
        design_space,
        scenario_type="DOE",
        formulation_name="DisciplinaryOpt",
    )

    scenario.execute(algo_name="PYDOE_LHS", n_samples=1)


@pytest.mark.parametrize(("l_b", "u_b"), [(-inf, inf), (1, inf), (-inf, 1)])
def test_uunormalized_components(mc, l_b, u_b) -> None:
    """Check that an error is raised when the design space is unbounded."""
    design_space = DesignSpace()
    design_space.add_variable("x", 2, lower_bound=1, upper_bound=3)
    design_space.add_variable("y", 3, lower_bound=l_b, upper_bound=u_b)
    design_space.add_variable("z", lower_bound=0, upper_bound=1)

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(sum, "f")

    error_message = "The components {2, 3, 4} of the design space are unbounded."
    with pytest.raises(ValueError, match=re.escape(error_message)):
        mc.compute_doe(design_space, n_samples=3)

    with pytest.raises(ValueError, match=re.escape(error_message)):
        mc.execute(problem, n_samples=3)


def test_uunormalized_components_with_parameter_space(mc) -> None:
    """Check that an error is not raised when the design space is a parameter space."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", "OTNormalDistribution")

    # The parameter space is unbounded.
    assert not parameter_space.normalize["x"]

    problem = OptimizationProblem(parameter_space)
    problem.objective = MDOFunction(sum, "f")

    mc.compute_doe(parameter_space, n_samples=3)
    mc.execute(problem, n_samples=3)


def f(x):
    return 2 * x


class Counter:
    def __init__(self):
        self.total = 0

    def callback(self, index, data):
        self.total += data[0]["f"]


@pytest.fixture
def problem():
    """A problem."""
    design_space = DesignSpace()
    design_space.add_variable("x")

    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(f, "f")
    return problem


@pytest.mark.parametrize("n_processes", [1, 2])
def test_callback(custom_doe, n_processes, problem):
    """Check the use of callbacks."""

    counter = Counter()
    custom_doe.execute(
        problem,
        samples=array([[1.0], [2.0]]),
        n_processes=n_processes,
        callbacks=(counter.callback,),
    )
    assert counter.total == 6


@pytest.mark.parametrize("n_processes", [1, 2])
@pytest.mark.parametrize("use_database", [False, True])
def test_use_database(custom_doe, n_processes, problem, use_database):
    """Check the option use_database."""
    custom_doe.execute(
        problem,
        samples=array([[1.0], [2.0]]),
        n_processes=n_processes,
        use_database=use_database,
    )
    assert bool(problem.database) is use_database


def test_serialize(custom_doe, tmp_wd):
    """Verify that BaseDOELibrary is serializable."""
    output_path = Path("out.pk")
    with open(output_path, "wb") as outf:
        pickle.dump(custom_doe, outf)

    with open(output_path, "rb") as outf:
        pickle.load(outf)


class _DummyDisc(Discipline):
    default_grammar_type = Discipline.GrammarType.SIMPLE

    def __init__(self) -> None:
        super().__init__("foo")
        self.io.input_grammar.update_from_names("x")
        self.io.output_grammar.update_from_names(("z", "t"))
        self.io.output_grammar.update_from_types({
            "s1": float,
            "s2": float,
            "z": ndarray,
            "t": ndarray,
        })

    def _run(self, input_data: StrKeyMapping):
        x = self.io.data["x"]
        self.io.data["z"] = array([sum(x)])
        self.io.data["t"] = 2 * x + 3
        self.io.data["s1"] = x[0]
        self.io.data["s2"] = x[1]


def test_parallel_doe_db(tmp_wd):
    """Verify the backup database in parallel with vector and scalar outputs."""

    def _create_scn() -> DOEScenario:
        design_space = DesignSpace()
        design_space.add_variable("x", lower_bound=0, upper_bound=1, size=2)

        scenario = create_scenario(
            [_DummyDisc()],
            "z",
            design_space,
            scenario_type="DOE",
            formulation_name="DisciplinaryOpt",
        )
        scenario.add_constraint("t")
        scenario.add_constraint("s1")
        scenario.add_constraint("s2")
        return scenario

    scenario_ser = _create_scn()
    bk_file_ser = Path("ser_out.h5")
    scenario_ser.set_optimization_history_backup(
        bk_file_ser, at_each_function_call=True, at_each_iteration=True
    )
    scenario_ser.execute(algo_name="PYDOE_FULLFACT", n_samples=4, n_processes=1)

    scenario_par = _create_scn()
    bk_file_par = Path("par_out.h5")
    scenario_par.set_optimization_history_backup(
        bk_file_par, at_each_function_call=True, at_each_iteration=True
    )
    scenario_par.execute(algo_name="PYDOE_FULLFACT", n_samples=4, n_processes=2)

    db_ser = Database.from_hdf(bk_file_ser)
    db_par = Database.from_hdf(bk_file_par)

    assert len(db_ser) == len(db_par)
    for func in ["z", "t", "s1", "s2"]:
        f_s = db_ser.get_function_history(func, with_x_vect=False)
        f_p = db_par.get_function_history(func, with_x_vect=False)
        assert_array_equal(f_p, f_s, strict=True)


@pytest.mark.parametrize("eval_func", [False, True])
@pytest.mark.parametrize("eval_jac", [False, True])
def test_eval_func_and_eval_jac(eval_func, eval_jac):
    """Check eval_func and eval_jac."""
    problem = Power2()
    SciPyDOE("MC").execute(
        problem,
        n_samples=1,
        eval_jac=eval_jac,
        eval_func=eval_func,
    )
    database = problem.database
    last_item = database.last_item
    name = "pow2"
    assert (name in last_item) is eval_func
    assert (database.get_gradient_name(name) in last_item) is eval_jac
