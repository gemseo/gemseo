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
from numpy.testing import assert_array_equal

from gemseo import execute_algo
from gemseo.algos.database import Database
from gemseo.algos.opt.mnbi import MNBI
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.analytical.binh_korn import BinhKorn
from gemseo.problems.analytical.fonseca_fleming import FonsecaFleming
from gemseo.problems.analytical.poloni import Poloni
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.analytical.viennet import Viennet


@pytest.fixture()
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
        "MNBI",
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
            "MNBI",
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
        "MNBI",
        max_iter=10000,
        sub_optim_max_iter=100,
        n_sub_optim=n_sub_optim,
        sub_optim_algo="NLOPT_SLSQP",
        n_processes=2,
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
        lambda x: x, MNBI._MNBI__SUB_OPTIM_CONSTRAINT_NAME, f_type="ineq"
    )
    binh_korn.add_ineq_constraint(protected_constraint)
    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The constraint name {MNBI._MNBI__SUB_OPTIM_CONSTRAINT_NAME} is "
            f"protected when using MNBI optimizer"
        ),
    ):
        execute_algo(
            binh_korn,
            "MNBI",
            max_iter=10000,
            sub_optim_max_iter=100,
            n_sub_optim=5,
            sub_optim_algo="NLOPT_SLSQP",
        )


@pytest.mark.parametrize("kwargs", [{}, {"debug_file_path": "foo.h5"}])
def test_debug_mode(tmp_wd, binh_korn, kwargs):
    """Test the creation of a debug file when the option is enabled."""
    execute_algo(
        binh_korn,
        "MNBI",
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


def test_skippable_betas(caplog):
    """Test the mechanism that allows to skip betas."""
    execute_algo(
        Poloni(),
        "MNBI",
        max_iter=10000,
        sub_optim_max_iter=5,
        n_sub_optim=30,
        sub_optim_algo="SLSQP",
    )
    assert "Skipping beta =" in caplog.text
