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

from os.path import exists

import pytest
from gemseo.algos.pareto_front import compute_pareto_optimal_points
from gemseo.algos.pareto_front import generate_pareto_plots
from matplotlib import pyplot as plt
from numpy import array
from numpy import ndarray
from numpy.random import rand
from numpy.random import seed
from numpy.testing import assert_array_equal


@pytest.fixture()
def objective_points() -> ndarray:
    """Return points.

    Returns:
         The objective points.
    """
    return array([[1, 2], [1.4, 1.7], [1.6, 1.6], [2, 1], [2, 2], [1.5, 1.5], [2, 0.5]])


@pytest.fixture()
def non_feasible_points() -> ndarray:
    """Return a non-feasible point mask.

    Returns:
         The non-feasible points.
    """
    return array([False, True, False, True, False, True, False])


def test_select_pareto_optimal(
    tmp_wd,
    objective_points: objective_points,
):
    """Test the selection of Pareto optimal points.

    Args:
        objective_points: Points fixture on which the test shall be applied.
    """
    inds = compute_pareto_optimal_points(objective_points)
    assert_array_equal(inds, array([True, True, False, False, False, True, True]))


def test_select_pareto_optimal_w_non_feasible_points(
    objective_points: objective_points,
    non_feasible_points: non_feasible_points,
):
    """Test the selection of Pareto optimal points, with non-feasible points.

    Args:
        objective_points: Points fixture on which the test shall be applied.
        non_feasible_points: Mask fixture of non-feasible points.
    """
    inds = compute_pareto_optimal_points(
        objective_points, feasible_points=~non_feasible_points
    )
    assert_array_equal(inds, array([True, False, True, False, False, False, True]))


def test_pareto_front(tmp_wd, objective_points):
    """Test the generation of Pareto fronts.

    Args:
        objective_points: points on which the test shall be applied
    """
    generate_pareto_plots(objective_points, ["0", "1"])
    outfile = "Pareto_2d.png"
    plt.savefig(outfile)
    plt.close()
    assert exists(outfile)


def test_raise_error_if_dimension_mismatch(tmp_wd, objective_points):
    """Check that a value error is raised if there is a mismatch between the objective
    values and the objective names.

    Args:
        objective_points: points on which the test shall be applied
    """
    expect_msg = (
        "^Inconsistent objective values size and objective names: \\d+ != \\d+$"
    )
    with pytest.raises(ValueError, match=expect_msg):
        generate_pareto_plots(objective_points, ["0", "1", "2"])


@pytest.mark.parametrize("show_non_feasible", (True, False))
def test_pareto_front_w_non_feasible(
    tmp_wd, objective_points, non_feasible_points, show_non_feasible
):
    """Generate Pareto fronts with non-feasible points.

    Args:
        objective_points: points on which the test shall be applied
        non_feasible_points: mask of non-feasible points
        show_non_feasible: if True, show the non-feasible points in the plot
    """
    generate_pareto_plots(
        objective_points,
        ["0", "1"],
        non_feasible_samples=non_feasible_points,
        show_non_feasible=show_non_feasible,
    )
    outfile = "Pareto_2d_non_feasible_not_shown.png"
    plt.savefig(outfile)
    plt.close()
    assert exists(outfile)


def test_5d(tmp_wd):
    """Generate a Pareto Front using random points."""
    seed(1)
    n_obj = 5
    objs = rand(100, n_obj)
    inds = compute_pareto_optimal_points(objs)
    assert sum(inds) > 0
    names = [str(i) for i in range(n_obj)]
    generate_pareto_plots(objs, names)
    outfile = "Pareto_5d.png"
    plt.savefig(outfile)
    plt.close()
    assert exists(outfile)
