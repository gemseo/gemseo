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
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author:  François Gallard - minor improvements for integration
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import lexsort
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from gemseo import execute_algo
from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.evaluation_function import EvaluationFunction
from gemseo.core.problem.database import Database
from gemseo.optimization import Augmented_Lagrangian_Order_0_Settings
from gemseo.optimization.nlopt.settings.nlopt_slsqp_settings import NLOPT_SLSQP_Settings
from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.problem.multiobjective_optimization.binh_korn import BinhKorn
from gemseo.problem.multiobjective_optimization.fonseca_fleming import FonsecaFleming
from gemseo.problem.multiobjective_optimization.poloni import Poloni
from gemseo.problem.multiobjective_optimization.viennet import Viennet
from gemseo.problem.optimization.power_2 import Power2
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from numpy import ndarray


@pytest.fixture
def binh_korn():
    """Fixture that returns a BinhKorn problem instance."""
    return BinhKorn()


@pytest.mark.parametrize("n_sub_optim", [5, 10])
@pytest.mark.parametrize(
    "opt_problem", [FonsecaFleming(), Poloni(), BinhKorn(), Viennet()]
)
def test_mnbi(n_sub_optim, opt_problem):
    """Tests the MNBI algo on several benchmark problems."""
    result = execute_algo(
        opt_problem,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=n_sub_optim,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
    )
    assert len(result.pareto_front.f_optima) >= n_sub_optim + 2


def test_min_n_sub_optim(snapshot):
    """Test that an exception is raised when the `n_sub_optim` is too low."""
    with assert_exception(ValueError, snapshot):
        execute_algo(
            Viennet(),
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=3,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        )


def identity(
    x_dv: ndarray,
) -> ndarray:
    """A function that returns its inputs.

    Args:
        x_dv: The design variable vector.

    Returns:
        The output values.
    """
    return x_dv


def test_mnbi_parallel(binh_korn):
    """Test the MNBI algo on the BinhKorn problem in parallel.

    Check that observables are stored as well.
    """
    observable = ArrayFunction(
        identity,
        name="identity",
        f_type=ArrayFunction.FunctionType.OBS,
        input_names=["x", "y"],
        dim=2,
    )
    binh_korn.add_observable(observable)
    n_sub_optim = 10
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=n_sub_optim,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        n_processes=2,
        xtol_abs=0.0,
    )
    assert_array_equal(
        binh_korn.database.get_function_value("identity", 1),
        binh_korn.database.get_x_vect(1),
    )
    assert len(result.pareto_front.f_optima) >= n_sub_optim + 2


def test_mnbi_parallel_with_an_observable_the_sub_optimizations_ignore(binh_korn):
    """Check a parallel run with an observable no sub-optimization evaluates.

    The history of a sub-optimization answers
    for some of the functions of the problem being solved only,
    and an observable outside the new-iteration ones is absent from it;
    reading it unguarded raised a `KeyError`.
    """
    binh_korn.add_observable(
        ArrayFunction(
            identity,
            name="identity",
            f_type=ArrayFunction.FunctionType.OBS,
            input_names=["x", "y"],
            dim=2,
        ),
        new_iter=False,
    )

    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=5,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        n_processes=2,
        xtol_abs=0.0,
    )

    assert len(result.pareto_front.f_optima) >= 7


def test_objective_values_of_a_parallel_run(binh_korn):
    """Check the objective values a parallel run brings back.

    A sub-optimization minimizing a component of the objective
    evaluates the objective of the problem being solved,
    which records in the database of that problem;
    returning the database of the sub-problem only
    left the individual optima without an objective value
    and the Pareto front read from that history was smaller.
    """
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=5,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        n_processes=2,
        xtol_abs=0.0,
    )

    database = binh_korn.database
    f_hist, x_hist = database.get_function_history(
        binh_korn.objective.name, with_x_vect=True
    )
    assert_array_equal(x_hist, database.get_x_vect_history())
    assert f_hist.shape == (len(database), binh_korn.objective.dim)


def test_serial_and_parallel_runs_store_the_same_database():
    """Check that a serial and a parallel run store the same evaluations.

    A sub-optimization evaluates the objective and the constraints of the
    problem being solved through their bound wrappers, which already record
    every evaluated point in the database of that problem;
    a parallel run only has to merge back the copy of that database a
    worker returns.
    Both modes should therefore end up with databases of the same size and
    the same objective history.

    Viennet has three objectives, so the beta sub-optimizations always
    restart from the initial design value instead of the previous
    sub-optimum, and `skip_betas` is disabled;
    with both sources of a run order dependency removed, the two modes are
    expected to evaluate the exact same points regardless of how the
    sub-optimizations are dispatched to worker processes.
    """
    common_settings = {
        "algo_name": "MNBI",
        "max_iter": 10000,
        "n_sub_optim": 8,
        "sub_optim_algo_settings": NLOPT_SLSQP_Settings(max_iter=50),
        "skip_betas": False,
        "xtol_abs": 0.0,
    }

    serial_problem = Viennet()
    execute_algo(serial_problem, n_processes=1, **common_settings)
    serial_database = serial_problem.database

    parallel_problem = Viennet()
    execute_algo(parallel_problem, n_processes=2, **common_settings)
    parallel_database = parallel_problem.database

    assert len(serial_database) == len(parallel_database)

    name = serial_problem.objective.name
    serial_f, serial_x = serial_database.get_function_history(name, with_x_vect=True)
    parallel_f, parallel_x = parallel_database.get_function_history(
        name, with_x_vect=True
    )
    # The two runs may store their points in a different order.
    serial_order = lexsort(serial_x.T)
    parallel_order = lexsort(parallel_x.T)
    assert_allclose(serial_x[serial_order], parallel_x[parallel_order])
    assert_allclose(serial_f[serial_order], parallel_f[parallel_order])


def test_no_redundant_merge_in_a_serial_run(binh_korn, monkeypatch):
    """Check that a serial run never merges a database by function name.

    The sub-optimizations evaluate the objective and the constraints of the
    problem being solved through their bound wrappers, which already record
    in its database; a serial run therefore never has to merge one back by
    function name, only the relaxed variable names, through an empty tuple
    of function names.
    """
    calls_with_names = []
    merge_function_histories = Database.merge_function_histories

    def count_merge(self, database, function_names):
        function_names = tuple(function_names)
        if function_names:
            calls_with_names.append(function_names)
        return merge_function_histories(self, database, function_names)

    monkeypatch.setattr(Database, "merge_function_histories", count_merge)

    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=5,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        n_processes=1,
    )

    assert not calls_with_names


def test_n_calls_of_a_serial_run(binh_korn, enable_function_statistics, monkeypatch):
    """Check the number of calls a serial run reports.

    A sub-optimization running in this process
    evaluates the objective of the problem being solved,
    which counts those calls itself;
    adding the count reported by the sub-optimization on top of it
    counted them twice.
    """
    name = binh_korn.objective.name
    n_evaluations = []
    evaluate = EvaluationFunction.evaluate

    def count_evaluation(self, input_value):
        """Count the evaluations asked of the objective of the problem."""
        if self.name == name:
            n_evaluations.append(input_value)

        return evaluate(self, input_value)

    monkeypatch.setattr(EvaluationFunction, "evaluate", count_evaluation)

    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=5,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=20),
        n_processes=1,
    )

    assert binh_korn.objective.n_calls == len(n_evaluations)


def test_mnbi_with_an_augmented_lagrangian_sub_algorithm_records_into_the_top_database(
    binh_korn,
):
    """Check that an augmented Lagrangian sub-algorithm records in the top database.

    It builds its inner problem from the constraints of the MNBI sub-problem,
    which are bound to the database of the top-level problem.
    """
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=3,
        sub_optim_algo_settings=Augmented_Lagrangian_Order_0_Settings(
            max_iter=3,
            sub_algorithm_settings=SLSQP_Settings(max_iter=10),
        ),
    )

    n_entries_with_a_constraint = sum(
        1
        for values in binh_korn.database.values()
        if "ineq1" in values or "ineq2" in values
    )
    # Without the inner evaluations,
    # the top-level database holds 15 points evaluating a constraint.
    assert n_entries_with_a_constraint > 18


def test_mono_objective_error(snapshot):
    """Check that an exception is raised for single objective problems."""
    with assert_exception(ValueError, snapshot):
        execute_algo(
            Power2(),
            algo_name="MNBI",
            max_iter=100,
            n_sub_optim=5,
            sub_optim_algo_settings=SLSQP_Settings(),
        )


def test_protected_const(binh_korn, snapshot):
    """Test that an exception is raised for a protected constraint name."""
    from gemseo.optimization.mnbi.mnbi import MNBI

    protected_constraint = ArrayFunction(
        lambda x: x,
        name=MNBI._MNBI__sub_optim_constraint_name,
        f_type=ArrayFunction.ConstraintType.INEQ,
    )
    binh_korn.add_constraint(protected_constraint)
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=5,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        )


@pytest.mark.parametrize("kwargs", [{}, {"debug_file_path": "foo.h5"}])
def test_debug_mode(tmp_wd, binh_korn, kwargs):
    """Test the creation of a debug file when the setting is enabled."""
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=3,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        debug=True,
        **kwargs,
    )
    file_name = kwargs.get("debug_file_path", "debug_history.h5")
    debug_database = Database.from_hdf(tmp_wd / file_name)
    assert len(debug_database) == 3
    assert "obj" in debug_database.last_item


def test_maximize_objective(binh_korn, enable_function_statistics):
    """Test the result of a maximized multi objective problem."""
    binh_korn.use_standardized_objective = False
    binh_korn.minimize_objective = False
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=100,
        n_sub_optim=5,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(),
    )
    assert len(result.pareto_front.f_optima) >= 7


def test_unfeasible_solution(binh_korn, snapshot):
    """Test the result of a maximized multi objective problem."""
    binh_korn.design_space.set_current_value(array([3, 3]))
    with assert_exception(RuntimeError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=1,
            n_sub_optim=3,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=1),
        )


def test_skippable_points(caplog):
    """Test the mechanism that allows to skip sub-optimizations."""
    execute_algo(
        Poloni(),
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=30,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=5),
    )
    assert "Skipping sub-optimization for phi_beta =" in caplog.text


def test_exclusive_settings_error(binh_korn, snapshot):
    """Test that an exception is raised when mutually exclusive settings are set.

    Settings custom_anchor_points and custom_phi_betas are not compatible with each
    other.
    """
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=10,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
            custom_anchor_points=[array([44.5, 14]), array([29.4, 19])],
            custom_phi_betas=[array([38, 17]), array([60, 10])],
        )


def test_custom_anchor_points_error(binh_korn, snapshot):
    """Test that exceptions are raised when custom_anchor_points has incorrect values.

    The length of the custom_anchor_points list must be the same as the number of
    objectives. The length of all custom_anchor_points arrays must be the same as the
    number of objectives.
    """
    custom_anchor_points = [array([44.5, 14])]
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=10,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
            custom_anchor_points=custom_anchor_points,
        )

    custom_anchor_points = [array([44.5, 14]), array([29.4, 19, 12])]
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=10,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
            custom_anchor_points=custom_anchor_points,
        )


def test_custom_phi_betas_warning(binh_korn, caplog):
    """Test that a warning is issued when custom_phi_betas has the wrong length."""
    custom_phi_betas = [array([38, 17]), array([60, 10])]
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=10,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        custom_phi_betas=custom_phi_betas,
    )
    assert (
        "The requested number of sub-optimizations "
        "does not match the number of custom phi_beta values; "
        f"keeping the latter ({len(custom_phi_betas)})." in caplog.text
    )


def test_custom_phi_betas_error(binh_korn, snapshot):
    """Test that an exception is raised for incorrect values of custom_phi_betas.

    The length of all custom_phi_betas arrays must be the same as the number of
    objectives.
    """
    custom_phi_betas = [array([38, 17]), array([60, 10, 28])]
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            n_sub_optim=10,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
            custom_phi_betas=custom_phi_betas,
        )


def test_mnbi_custom_anchor_points(binh_korn):
    """Tests the MNBI algo restart with custom anchor points."""
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=10,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
    )
    result_restart = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=10,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        custom_anchor_points=[array([44.5, 14]), array([29.4, 19])],
    )

    assert (
        len(result_restart.pareto_front.f_optima)
        >= len(result.pareto_front.f_optima) + 10
    )


def test_mnbi_custom_phi_betas(binh_korn):
    """Tests the MNBI algo restart with custom values of phi_beta."""
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=10,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
    )
    result_restart = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=2,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
        custom_phi_betas=[array([38, 17]), array([60, 10])],
    )

    assert (
        len(result_restart.pareto_front.f_optima)
        >= len(result.pareto_front.f_optima) + 2
    )


@pytest.mark.parametrize("normalize_design_space", [True, False])
def test_mnbi_normalize_design_space(binh_korn, normalize_design_space):
    """Tests that the setting `normalize_design_space` is correctly handled."""
    utopia_neighbor = (
        [17.01259261, 25.0875926]
        if normalize_design_space
        else [14.89156056, 26.43593304]
    )

    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        n_sub_optim=10,
        sub_optim_algo_settings=NLOPT_SLSQP_Settings(
            max_iter=100,
            normalize_design_space=normalize_design_space,
            ftol_abs=1e-14,
            xtol_abs=1e-14,
            ftol_rel=1e-8,
            xtol_rel=1e-8,
            ineq_tolerance=1e-4,
        ),
        xtol_abs=0.0,
    )
    assert_allclose(result.pareto_front.f_utopia, [0, 4], atol=1e-7)

    assert_allclose(result.pareto_front.f_utopia_neighbors.flatten(), utopia_neighbor)


def test_normalize_exception(binh_korn, snapshot):
    """Check that an exception is raised when the top problem is normalized."""
    with assert_exception(ValueError, snapshot):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            n_sub_optim=5,
            sub_optim_algo_settings=NLOPT_SLSQP_Settings(max_iter=100),
            normalize_design_space=True,
        )
