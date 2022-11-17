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

from unittest import mock  # noqa: F401

import pytest
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.post.post_factory import PostFactory
from gemseo.problems.analytical.binh_korn import BinhKorn
from gemseo.problems.analytical.power_2 import Power2
from gemseo.utils.testing import image_comparison

# - the kwargs to be passed to ParetoFront._plot
# - the expected file names without extension to be compared
TEST_PARAMETERS = {
    "default": ({}, ["ParetoFront"]),
}


TEST_PARAMETERS_BINHKORN = {
    "show_non_feasible_True": (
        {"show_non_feasible": True},
        ["ParetoFront_BinhKorn_NonFeasible_True"],
    ),
    "show_non_feasible_False": (
        {"show_non_feasible": False},
        ["ParetoFront_BinhKorn_NonFeasible_False"],
    ),
}

pytestmark = pytest.mark.skipif(
    not PostFactory().is_available("ScatterPlotMatrix"),
    reason="ScatterPlotMatrix plot is not available.",
)


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_pareto(
    tmp_wd,
    kwargs,
    baseline_images,
    pyplot_close_all,
):
    """Test the generation of Pareto front plots.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        kwargs: The parametrized keyword arguments.
        baseline_images: The reference images to be compared.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = Power2()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=50)
    post = PostFactory().execute(
        problem,
        "ParetoFront",
        save=False,
        file_path="power",
        objectives=problem.get_all_functions_names(),
        **kwargs,
    )
    post.figures


def test_pareto_minimize(
    tmp_wd,
):
    """Test the generation of Pareto front plots.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    problem = Power2()
    problem.change_objective_sign()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=50)
    PostFactory().execute(
        problem, "ParetoFront", file_path="power", objectives=["pow2", "ineq1"]
    )


def test_pareto_incorrect_objective_list():
    """Test that an error is raised if the objective labels len is not consistent."""
    problem = Power2()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=50)
    msg = "objective_labels shall have the same dimension as the number of objectives to plot."
    with pytest.raises(ValueError, match=msg):
        PostFactory().execute(
            problem,
            "ParetoFront",
            save=False,
            objectives=problem.get_all_functions_names(),
            objectives_labels=["fake_label"],
            file_path="power",
        )


def test_pareto_incorrect_objective_names():
    """Test that an error is raised if the objective labels len is not consistent."""
    problem = Power2()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=50)
    msg = (
        "Cannot build Pareto front, Function \\w* is neither among"
        " optimization problem functions:.*\\.$"
    )
    with pytest.raises(ValueError, match=msg):
        PostFactory().execute(
            problem,
            "ParetoFront",
            save=False,
            objectives=["fake_obj"],
            file_path="power",
        )


@pytest.mark.parametrize(
    "kwargs, baseline_images",
    TEST_PARAMETERS_BINHKORN.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS_BINHKORN.keys(),
)
@image_comparison(None)
def test_pareto_binhkorn(
    tmp_wd,
    kwargs,
    baseline_images,
    pyplot_close_all,
):
    """Test the generation of Pareto front plots using the Binh-Korn problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        kwargs: The parametrized keyword arguments.
        baseline_images: The reference images to be compared.
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = BinhKorn()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=100)
    post = PostFactory().execute(
        problem,
        "ParetoFront",
        save=False,
        file_path="binh_korn",
        objectives=["compute_binhkorn"],
        **kwargs,
    )
    post.figures


@image_comparison(["binh_korn_design_variable"])
def test_pareto_binhkorn_design_variable(pyplot_close_all):
    """Test the generation of Pareto front plots using the Binh-Korn problem.

    Args:
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = BinhKorn()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=100)
    post = PostFactory().execute(
        problem,
        "ParetoFront",
        save=False,
        file_path="binh_korn_design_variable",
        objectives=["x", "compute_binhkorn"],
        objectives_labels=["xx", "compute_binhkorn1", "compute_binhkorn2"],
    )
    post.figures


@image_comparison(["binh_korn_no_obj"])
def test_pareto_binhkorn_no_obj(pyplot_close_all):
    """Test the generation of Pareto front plots using the Binh-Korn problem.

    Args:
        pyplot_close_all: Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    problem = BinhKorn()
    DOEFactory().execute(problem, algo_name="fullfact", n_samples=100)
    post = PostFactory().execute(
        problem,
        "ParetoFront",
        save=False,
        file_path="binh_korn_no_obj",
    )
    post.figures
