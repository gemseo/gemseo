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

import pytest

from gemseo.algos.doe.factory import DOE_LIBRARY_FACTORY
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.post import ParetoFront_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.testing.helpers import assert_exception
from gemseo.utils.testing.helpers import image_comparison

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
    not POST_FACTORY.is_available("ScatterPlotMatrix"),
    reason="ScatterPlotMatrix plot is not available.",
)


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_pareto(tmp_wd, kwargs, baseline_images) -> None:
    """Test the generation of Pareto front plots.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        kwargs: The parametrized keyword arguments.
        baseline_images: The reference images to be compared.
    """
    problem = Power2()
    DOE_LIBRARY_FACTORY.execute(problem, settings=PYDOE_FULLFACT_Settings(n_samples=50))
    POST_FACTORY.execute(
        problem,
        ParetoFront_Settings(
            save=False, file_path="power", objectives=problem.function_names, **kwargs
        ),
    )


def test_pareto_minimize(
    tmp_wd,
) -> None:
    """Test the generation of Pareto front plots.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    problem = Power2()
    problem.minimize_objective = False
    DOE_LIBRARY_FACTORY.execute(problem, settings=PYDOE_FULLFACT_Settings(n_samples=50))
    POST_FACTORY.execute(
        problem, ParetoFront_Settings(file_path="power", objectives=["pow2", "ineq1"])
    )


def test_pareto_incorrect_objective_list(snapshot) -> None:
    """Test that an error is raised if the objective labels len is not consistent."""
    problem = Power2()
    DOE_LIBRARY_FACTORY.execute(problem, settings=PYDOE_FULLFACT_Settings(n_samples=50))
    with assert_exception(ValueError, snapshot):
        POST_FACTORY.execute(
            problem,
            ParetoFront_Settings(
                save=False,
                objectives=problem.function_names,
                objectives_labels=["fake_label"],
                file_path="power",
            ),
        )


def test_pareto_incorrect_objective_names(snapshot) -> None:
    """Test that an error is raised if the objective labels len is not consistent."""
    problem = Power2()
    DOE_LIBRARY_FACTORY.execute(problem, settings=PYDOE_FULLFACT_Settings(n_samples=50))
    with assert_exception(ValueError, snapshot):
        POST_FACTORY.execute(
            problem,
            ParetoFront_Settings(
                save=False, objectives=["fake_obj"], file_path="power"
            ),
        )


@pytest.mark.parametrize(
    ("kwargs", "baseline_images"),
    TEST_PARAMETERS_BINHKORN.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS_BINHKORN.keys(),
)
@image_comparison(None)
def test_pareto_binhkorn(tmp_wd, kwargs, baseline_images) -> None:
    """Test the generation of Pareto front plots using the Binh-Korn problem.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        kwargs: The parametrized keyword arguments.
        baseline_images: The reference images to be compared.
    """
    problem = BinhKorn()
    DOE_LIBRARY_FACTORY.execute(
        problem, settings=PYDOE_FULLFACT_Settings(n_samples=100)
    )
    POST_FACTORY.execute(
        problem,
        ParetoFront_Settings(
            save=False, file_path="binh_korn", objectives=["compute_binhkorn"], **kwargs
        ),
    )


@image_comparison(["binh_korn_design_variable"])
def test_pareto_binhkorn_design_variable() -> None:
    """Test the generation of Pareto front plots using the Binh-Korn problem."""
    problem = BinhKorn()
    DOE_LIBRARY_FACTORY.execute(
        problem, settings=PYDOE_FULLFACT_Settings(n_samples=100)
    )
    POST_FACTORY.execute(
        problem,
        ParetoFront_Settings(
            save=False,
            file_path="binh_korn_design_variable",
            objectives=["x", "compute_binhkorn"],
            objectives_labels=["xx", "compute_binhkorn1", "compute_binhkorn2"],
        ),
    )


@image_comparison(["binh_korn_no_obj"])
def test_pareto_binhkorn_no_obj() -> None:
    """Test the generation of Pareto front plots using the Binh-Korn problem."""
    problem = BinhKorn()
    DOE_LIBRARY_FACTORY.execute(
        problem, settings=PYDOE_FULLFACT_Settings(n_samples=100)
    )
    POST_FACTORY.execute(
        problem,
        ParetoFront_Settings(save=False, file_path="binh_korn_no_obj"),
    )
