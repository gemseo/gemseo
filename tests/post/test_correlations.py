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
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import random
from functools import partial
from pathlib import Path

import pytest

from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post import Correlations_Settings
from gemseo.post.correlations import Correlations
from gemseo.post.factory import POST_FACTORY
from gemseo.problems.optimization.rosenbrock import Rosenbrock
from gemseo.utils.testing.helpers import assert_exception

PARENT_PATH = Path(__file__).parent
POWER_HDF5_PATH = PARENT_PATH / "power2_opt_pb.h5"
MOD_SELLAR_HDF5_PATH = PARENT_PATH / "modified_sellar_opt_pb.h5"


@pytest.fixture(scope="module")
def factory():
    return POST_FACTORY


def test_correlations(tmp_wd, factory, snapshot_matplotlib) -> None:
    """Test correlations with the Rosenbrock problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
    """
    problem = Rosenbrock(20)
    OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=L_BFGS_B_Settings())

    post = factory.execute(
        problem,
        Correlations_Settings(
            save=True,
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.95,
            file_path="correlations_1",
        ),
    )
    assert len(post.output_file_paths) == 2
    for outf in post.output_file_paths:
        assert Path(outf).exists()


def test_correlations_import(tmp_wd, factory) -> None:
    """Test correlations with imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
    """
    problem = OptimizationProblem.from_hdf(POWER_HDF5_PATH)
    post = factory.execute(
        problem,
        Correlations_Settings(
            save=True,
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.999,
            file_path="correlations_2",
        ),
    )
    assert len(post.output_file_paths) == 1
    for outf in post.output_file_paths:
        assert Path(outf).exists()


def test_correlations_func_name_error(factory, snapshot) -> None:
    """Test ValueError for non-existent function.

    Args:
        factory: Fixture that returns a post-processing factory.
    """
    problem = Rosenbrock(20)
    OPTIMIZATION_LIBRARY_FACTORY.execute(problem, settings=L_BFGS_B_Settings())

    with assert_exception(ValueError, snapshot):
        factory.execute(
            problem, Correlations_Settings(save=False, show=False, func_names=["toto"])
        )


@pytest.mark.parametrize(
    ("func_names"),
    [["pow2", "ineq1"], []],
)
def test_correlations_func_names(factory, func_names, snapshot_matplotlib) -> None:
    """Test func_names filter.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        func_names: The function names subset for which the correlations
            are computed. If None, all functions are considered.
    """
    problem = OptimizationProblem.from_hdf(POWER_HDF5_PATH)
    factory.execute(
        problem,
        Correlations_Settings(
            func_names=func_names,
            save=False,
            file_extension="png",
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.99,
            file_path="correlations",
        ),
    )


def test_func_name_sorting(factory, snapshot_matplotlib) -> None:
    """Test that the function names sorting.

    Use a database from a modified Sellar problem
    with function and variable names that are similar
    i.e. `obj`, `obj_const`, `c_1`, `c_1_y`.
    The subplot labels must be correct.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
    """
    problem = OptimizationProblem.from_hdf(MOD_SELLAR_HDF5_PATH)
    factory.execute(
        problem,
        Correlations_Settings(
            func_names=["obj", "c_1", "obj_constr"],
            save=False,
            file_extension="png",
            n_plots_x=4,
            n_plots_y=4,
            coeff_limit=0.99,
            file_path="correlations",
        ),
    )


def test_func_order() -> None:
    """Test the func_order static method used to sort the function names.

    When the variables (functions and design variables) are sorted using `func_order` as
    the key, the output should have all the elements ordered following the pattern of
    `func_names`. Design variables are to be sent to the end of the list, their order is
    not important.

    In this test, the variables are shuffled randomly to simulate the way a user enters
    the data.
    """
    variables = [
        "y_1_2",
        "x_1",
        "y_Final",
        "y_final_10",
        "y_1_1",
        "-[pos-0.5][0]",
        "-[pos-0.5][1]",
        "Cruise_Speed_1_4",
        "pressure_1",
        "x_a_23",
        "x_a_1",
        "DesignVariable1",
        "x_42300",
        "[neg-0.5][0]",
        "pressure",
        "[neg-0.5][1]",
        "y_1",
        "CruiseSpeed_1",
        "design_variable_2",
        "sym_*",
        "pressure_empty",
        "eq[0]",
        "eq[1]",
    ]

    random.shuffle(variables)
    variables.sort(
        key=partial(
            Correlations.func_order,
            [
                "x_a",
                "Cruise_Speed_1",
                "y_1",
                "pressure_empty",
                "x_1",
                "eq",
                "CruiseSpeed",
                "[neg-0.5]",
                "y_Final",
                "x",
                "y_final",
                "-[pos-0.5]",
                "pressure",
                "sym",
            ],
        )
    )

    variables_expected = [
        "x_a_1",
        "x_a_23",
        "Cruise_Speed_1_4",
        "y_1",
        "y_1_1",
        "y_1_2",
        "pressure_empty",
        "x_1",
        "eq[0]",
        "eq[1]",
        "CruiseSpeed_1",
        "[neg-0.5][0]",
        "[neg-0.5][1]",
        "y_Final",
        "x_42300",
        "y_final_10",
        "-[pos-0.5][0]",
        "-[pos-0.5][1]",
        "pressure",
        "pressure_1",
        "DesignVariable1",
        "design_variable_2",
        "sym_*",
    ]

    assert variables == variables_expected


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective,
    common_problem,
    snapshot_matplotlib,
) -> None:
    """Check Correlations with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = Correlations(common_problem)
    maximum_correlation_coefficient = opt.MAXIMUM_CORRELATION_COEFFICIENT
    opt.MAXIMUM_CORRELATION_COEFFICIENT = 1.0
    opt.execute(
        Correlations_Settings(func_names=["obj", "eq", "neg", "pos"], save=False)
    )
    opt.MAXIMUM_CORRELATION_COEFFICIENT = maximum_correlation_coefficient
