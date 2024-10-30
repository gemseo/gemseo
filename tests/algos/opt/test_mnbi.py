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

import re

import pytest
from numpy import array
from numpy import ndarray
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal

from gemseo import execute_algo
from gemseo.algos.database import Database
from gemseo.algos.opt.mnbi.mnbi import MNBI
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn
from gemseo.problems.multiobjective_optimization.fonseca_fleming import FonsecaFleming
from gemseo.problems.multiobjective_optimization.poloni import Poloni
from gemseo.problems.multiobjective_optimization.viennet import Viennet
from gemseo.problems.optimization.power_2 import Power2


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
        sub_optim_max_iter=100,
        n_sub_optim=n_sub_optim,
        sub_optim_algo="NLOPT_SLSQP",
    )
    assert len(result.pareto_front.f_optima) >= n_sub_optim + 2


def test_min_n_sub_optim():
    """Test that an exception is raised when the `n_sub_optim` is too low."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of sub-optimization problems must be "
            "strictly greater than the number of objectives 3; got 3."
        ),
    ):
        execute_algo(
            Viennet(),
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=3,
            sub_optim_algo="NLOPT_SLSQP",
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
    observable = MDOFunction(
        identity,
        name="identity",
        f_type=MDOFunction.FunctionType.OBS,
        input_names=["x", "y"],
        dim=2,
    )
    binh_korn.add_observable(observable)
    n_sub_optim = 10
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=n_sub_optim,
        sub_optim_algo="NLOPT_SLSQP",
        n_processes=2,
        xtol_abs=0.0,
    )
    assert_array_equal(
        binh_korn.database.get_function_value("identity", 1),
        binh_korn.database.get_x_vect(1),
    )
    assert len(result.pareto_front.f_optima) >= n_sub_optim + 2


def test_mono_objective_error():
    """Check that an exception is raised for single objective problems."""
    with pytest.raises(
        ValueError,
        match=re.escape("MNBI optimizer is not suitable for mono-objective problems."),
    ):
        execute_algo(
            Power2(),
            algo_name="MNBI",
            max_iter=100,
            n_sub_optim=5,
            sub_optim_algo="SLSQP",
        )


def test_protected_const(binh_korn):
    """Test that an exception is raised for a protected constraint name."""
    protected_constraint = MDOFunction(
        lambda x: x,
        MNBI._MNBI__SUB_OPTIM_CONSTRAINT_NAME,
        f_type=MDOFunction.ConstraintType.INEQ,
    )
    binh_korn.add_constraint(protected_constraint)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The constraint name {MNBI._MNBI__SUB_OPTIM_CONSTRAINT_NAME} is "
            f"protected when using MNBI optimizer"
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=5,
            sub_optim_algo="NLOPT_SLSQP",
        )


@pytest.mark.parametrize("kwargs", [{}, {"debug_file_path": "foo.h5"}])
def test_debug_mode(tmp_wd, binh_korn, kwargs):
    """Test the creation of a debug file when the setting is enabled."""
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=3,
        sub_optim_algo="NLOPT_SLSQP",
        debug=True,
        **kwargs,
    )
    file_name = kwargs.get("debug_file_path", "debug_history.h5")
    debug_database = Database.from_hdf(tmp_wd / file_name)
    assert len(debug_database) == 3
    assert "obj" in debug_database.last_item


def test_maximize_objective(binh_korn):
    """Test the result of a maximized multi objective problem."""
    binh_korn.use_standardized_objective = False
    binh_korn.minimize_objective = False
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=100,
        n_sub_optim=5,
        sub_optim_algo="SLSQP",
    )
    assert len(result.pareto_front.f_optima) >= 7


def test_unfeasible_solution(binh_korn):
    """Test the result of a maximized multi objective problem."""
    binh_korn.design_space.set_current_value(array([3, 3]))
    with pytest.raises(
        RuntimeError,
        match=re.escape("No feasible optimum found for the 0-th objective function."),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=1,
            sub_optim_max_iter=1,
            n_sub_optim=3,
            sub_optim_algo="SLSQP",
        )


def test_skippable_points(caplog):
    """Test the mechanism that allows to skip sub-optimizations."""
    execute_algo(
        Poloni(),
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=5,
        n_sub_optim=30,
        sub_optim_algo="SLSQP",
    )
    assert "Skipping sub-optimization for phi_beta =" in caplog.text


def test_exclusive_settings_error(binh_korn):
    """Test that an exception is raised when mutually exclusive settings are set.

    Settings custom_anchor_points and custom_phi_betas are not compatible with each
    other.
    """
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The custom_anchor_points and custom_phi_betas settings "
            "cannot be set at the same time."
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=10,
            sub_optim_algo="NLOPT_SLSQP",
            custom_anchor_points=[array([44.5, 14]), array([29.4, 19])],
            custom_phi_betas=[array([38, 17]), array([60, 10])],
        )


def test_custom_anchor_points_error(binh_korn):
    """Test that exceptions are raised when custom_anchor_points has incorrect values.

    The length of the custom_anchor_points list must be the same as the number of
    objectives. The length of all custom_anchor_points arrays must be the same as the
    number of objectives.
    """
    custom_anchor_points = [array([44.5, 14])]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of custom anchor points must be "
            f"the same as the number of objectives {binh_korn.objective.dim}; "
            f"got {len(custom_anchor_points)}."
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=10,
            sub_optim_algo="NLOPT_SLSQP",
            custom_anchor_points=custom_anchor_points,
        )

    custom_anchor_points = [array([44.5, 14]), array([29.4, 19, 12])]
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The custom anchor points must be of dimension {binh_korn.objective.dim}; "
            f"got {[len(p) for p in custom_anchor_points]}"
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=10,
            sub_optim_algo="NLOPT_SLSQP",
            custom_anchor_points=custom_anchor_points,
        )


def test_custom_phi_betas_warning(binh_korn, caplog):
    """Test that a warning is issued when custom_phi_betas has the wrong length."""
    custom_phi_betas = [array([38, 17]), array([60, 10])]
    execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=10,
        sub_optim_algo="NLOPT_SLSQP",
        custom_phi_betas=custom_phi_betas,
    )
    assert (
        "The requested number of sub-optimizations "
        "does not match the number of custom phi_beta values; "
        f"keeping the latter ({len(custom_phi_betas)})." in caplog.text
    )


def test_custom_phi_betas_error(binh_korn):
    """Test that an exception is raised for incorrect values of custom_phi_betas.

    The length of all custom_phi_betas arrays must be the same as the number of
    objectives.
    """
    custom_phi_betas = [array([38, 17]), array([60, 10, 28])]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The custom phi_beta values "
            f"must be of dimension {binh_korn.objective.dim}; "
            f"got {[len(p) for p in custom_phi_betas]}"
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=10,
            sub_optim_algo="NLOPT_SLSQP",
            custom_phi_betas=custom_phi_betas,
        )


def test_mnbi_custom_anchor_points(binh_korn):
    """Tests the MNBI algo restart with custom anchor points."""
    result = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=10,
        sub_optim_algo="NLOPT_SLSQP",
    )
    result_restart = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=10,
        sub_optim_algo="NLOPT_SLSQP",
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
        sub_optim_max_iter=100,
        n_sub_optim=10,
        sub_optim_algo="NLOPT_SLSQP",
    )
    result_restart = execute_algo(
        binh_korn,
        algo_name="MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=2,
        sub_optim_algo="NLOPT_SLSQP",
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
        sub_optim_max_iter=100,
        n_sub_optim=10,
        sub_optim_algo="NLOPT_SLSQP",
        sub_optim_algo_settings={
            "normalize_design_space": normalize_design_space,
            "ftol_abs": 1e-14,
            "xtol_abs": 1e-14,
            "ftol_rel": 1e-8,
            "xtol_rel": 1e-8,
            "ineq_tolerance": 1e-4,
        },
        xtol_abs=0.0,
    )
    assert_allclose(result.pareto_front.f_utopia, [0, 4], atol=1e-7)

    assert_allclose(result.pareto_front.f_utopia_neighbors.flatten(), utopia_neighbor)


def test_normalize_exception(binh_korn):
    """Check that an exception is raised when the top problem is normalized."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The mNBI algo does not allow to normalize the design space at"
            " the top level"
        ),
    ):
        execute_algo(
            binh_korn,
            algo_name="MNBI",
            max_iter=100,
            n_sub_optim=5,
            sub_optim_algo="SLSQP",
            normalize_design_space=True,
        )
