# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import random
from functools import partial

import pytest
from matplotlib.testing.decorators import image_comparison

from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.post.correlations import Correlations
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.rosenbrock import Rosenbrock
from gemseo.utils.py23_compat import PY2, Path

PARENT_PATH = Path(__file__).parent
POWER_HDF5_PATH = PARENT_PATH / "power2_opt_pb.h5"
MOD_SELLAR_HDF5_PATH = PARENT_PATH / "modified_sellar_opt_pb.h5"


@pytest.fixture(scope="module")
def factory():
    return PostFactory()


def test_correlations(tmp_wd, factory, pyplot_close_all):
    """Test correlations with the Rosenbrock problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = Rosenbrock(20)
    OptimizersFactory().execute(problem, "L-BFGS-B")

    post = factory.execute(
        problem,
        "Correlations",
        save=True,
        n_plots_x=4,
        n_plots_y=4,
        coeff_limit=0.95,
        file_path="correlations_1",
    )
    assert len(post.output_files) == 2
    for outf in post.output_files:
        assert Path(outf).exists()


def test_correlations_import(tmp_wd, factory, pyplot_close_all):
    """Test correlations with imported problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(str(POWER_HDF5_PATH))
    post = factory.execute(
        problem,
        "Correlations",
        save=True,
        n_plots_x=4,
        n_plots_y=4,
        coeff_limit=0.999,
        file_path="correlations_2",
    )
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_correlations_func_name_error(factory):
    """Test ValueError for non-existent function.

    Args:
        factory: Fixture that returns a post-processing factory.
    """
    problem = Rosenbrock(20)
    OptimizersFactory().execute(problem, "L-BFGS-B")

    with pytest.raises(
        ValueError, match=r"The following elements are not" r" functions: .*toto.*"
    ):
        factory.execute(
            problem, "Correlations", save=False, show=False, func_names=["toto"]
        )


@pytest.mark.skipif(PY2, reason="image comparison does not work with python 2")
@pytest.mark.parametrize(
    "func_names,baseline_images",
    [(["pow2", "ineq1"], ["pow2_ineq1"]), ([], ["all_func"])],
)
@image_comparison(None, extensions=["png"])
def test_correlations_func_names(
    tmp_wd, factory, baseline_images, func_names, pyplot_close_all
):
    """Test func_names filter.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        baseline_images: The reference images to be compared.
        func_names: The function names subset for which the correlations
            are computed. If None, all functions are considered.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(str(POWER_HDF5_PATH))
    post = factory.execute(
        problem,
        "Correlations",
        func_names=func_names,
        save=False,
        file_extension="png",
        n_plots_x=4,
        n_plots_y=4,
        coeff_limit=0.99,
        file_path="correlations",
    )
    post.figures


@pytest.mark.skipif(PY2, reason="image comparison does not work with python 2")
@image_comparison(baseline_images=["modified_sellar"], extensions=["png"])
def test_func_name_sorting(tmp_wd, factory, pyplot_close_all):
    """Test that the function names sorting.

    Use a database from a modified Sellar problem
    with function and variable names that are similar
    i.e. `obj`, `obj_const`, `c_1`, `c_1_y`.
    The subplot labels must be correct.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        factory: Fixture that returns a post-processing factory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = OptimizationProblem.import_hdf(str(MOD_SELLAR_HDF5_PATH))
    post = factory.execute(
        problem,
        "Correlations",
        func_names=["obj", "c_1", "obj_constr"],
        save=False,
        file_extension="png",
        n_plots_x=4,
        n_plots_y=4,
        coeff_limit=0.99,
        file_path="correlations",
    )
    post.figures


def test_func_order():
    """Test the func_order static method used to sort the function names.

    When the variables (functions and design variables) are sorted using
    `func_order` as the key, the output should have all the elements ordered
    following the pattern of `func_names`. Design variables are to be sent
    to the end of the list, their order is not important.

    In this test, the variables are shuffled randomly to simulate the way a user
    enters the data.
    """
    variables = [
        "y_1_2",
        "x_1",
        "y_Final",
        "y_final_10",
        "y_1_1",
        "Cruise_Speed_1_4",
        "pressure_1",
        "x_a_23",
        "x_a_1",
        "DesignVariable1",
        "x_42300",
        "pressure",
        "y_1",
        "CruiseSpeed_1",
        "design_variable_2",
        "sym_*",
        "pressure_empty",
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
                "CruiseSpeed",
                "y_Final",
                "x",
                "y_final",
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
        "CruiseSpeed_1",
        "y_Final",
        "x_42300",
        "y_final_10",
        "pressure",
        "pressure_1",
        "DesignVariable1",
        "design_variable_2",
        "sym_*",
    ]

    assert variables == variables_expected
