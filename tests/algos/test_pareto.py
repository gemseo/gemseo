# Copyright 2022 Airbus SAS
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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Gabriel Max DE MENDONÇA ABRANTES
#                 Francois Gallard
"""Tests for the Pareto front."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import zeros
from numpy.testing import assert_allclose
from numpy.testing import assert_array_equal
from pandas import DataFrame
from pandas import MultiIndex

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.pareto.pareto_front import ParetoFront
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn

if TYPE_CHECKING:
    from gemseo.algos.optimization_problem import OptimizationProblem


@pytest.fixture
def problem_2obj() -> OptimizationProblem:
    """The Binh-Korn optimization problem ready to be post-processed."""
    problem = BinhKorn()
    OptimizationLibraryFactory().execute(
        problem,
        algo_name="MNBI",
        max_iter=100,
        n_sub_optim=5,
        sub_optim_algo="SLSQP",
        sub_optim_algo_settings={
            "normalize_design_space": False,
        },
        normalize_design_space=False,
    )
    return problem


def test_pareto(problem_2obj):
    """Test the :class:`ParetoFront` class.

    Args:
        problem_2obj: Fixture returning the multi-objective
            optimization problem to post-process.
    """
    obj_name = problem_2obj.objective.name
    database = problem_2obj.database
    keys = list(database.keys())
    # Take the objective evaluations out from database except for the two firsts.
    for key in keys[2:]:
        database.get(key).pop(problem_2obj.objective.name)
    pareto = ParetoFront.from_optimization_problem(problem_2obj)

    # Check problem property.
    assert pareto._problem == problem_2obj

    obj0 = database.get_function_value(obj_name, keys[0])
    obj1 = database.get_function_value(obj_name, keys[1])
    if len(pareto.f_optima) == 1:
        assert (obj0 in pareto.f_optima) or (obj1 in pareto.f_optima)
    else:
        assert obj0 in pareto.f_optima
        assert obj1 in pareto.f_optima

    # Check if the anchor points are in database.
    for x_anchor, f_anchor in zip(pareto.x_anchors, pareto.f_anchors):
        assert_array_equal(
            database.get_function_value(problem_2obj.objective.name, x_anchor),
            f_anchor,
        )

    # Check if the minimum norm point is in database.
    assert pareto.f_utopia_neighbors in database.get_function_history(
        problem_2obj.objective.name
    )


def test_get_utopia_nearest_neighbors():
    """Test the __get_utopia_nearest_neighbors method."""
    # 2 objectives, 3 design variables and 4 points on the pareto front.
    f_optima = array([
        [1.0, 4.0],
        [2.0, 3.0],
        [3.0, 2.0],
        [4.0, 1.0],
    ])
    x_optima = array([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
    ])

    f_utopia_neighbors, x_utopia_neighbors, min_norm = (
        ParetoFront._ParetoFront__get_utopia_nearest_neighbors(
            f_optima, x_optima, f_utopia=zeros(2)
        )
    )

    assert_array_equal(f_utopia_neighbors, f_optima[[1, 2]])
    assert_array_equal(x_utopia_neighbors, x_optima[[1, 2]])
    assert min_norm == (2**2 + 3**2) ** 0.5

    with pytest.raises(
        Exception,
        match="does not have the same amount of objectives as the pareto front",
    ):
        ParetoFront._ParetoFront__get_utopia_nearest_neighbors(
            f_optima,
            x_optima,
            f_utopia=array([1.0, 1.0, 1.0]),
        )


def test_pretty_table():
    """Test the creation of a PrettyTable."""
    # Create DataFrame with multiple index and column levels.
    dframe = DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]})
    indexes = [("i", "1"), ("i", "2"), ("j", "1"), ("j", "2")]
    dframe.index = MultiIndex.from_tuples(indexes)
    columns = [("c", "a"), ("c", "b")]
    dframe.columns = MultiIndex.from_tuples(columns)

    get_pretty_table_from_df = ParetoFront._ParetoFront__get_pretty_table_from_df
    p_table = str(get_pretty_table_from_df(dframe))
    for val in columns + indexes:
        assert f"{val[0]} ({val[1]})" in p_table

    # Drop multiple indexes.
    dframe.reset_index(level=[0, 1], drop=True, inplace=True)
    p_table = str(get_pretty_table_from_df(dframe))
    for val in range(4):
        assert str(val) in p_table


def test_get_lowest_norm_attribute(problem_2obj):
    """Test the lowest norm attributes from the Pareto result."""
    pareto_result = ParetoFront.from_optimization_problem(problem_2obj)
    x_utopia_neighbors = pareto_result.x_utopia_neighbors
    distance_from_utopia = pareto_result.distance_from_utopia
    x_optima = pareto_result.x_optima
    f_utopia = pareto_result.f_utopia
    f_anti_utopia = pareto_result.f_anti_utopia

    assert_allclose(x_utopia_neighbors, array([[1.4295, 1.4295]]), atol=1e-4)
    assert_allclose(distance_from_utopia, 27.0, atol=1e-2)
    assert array([5.0, 3.0]) in x_optima
    assert_allclose(f_utopia, array([0.0, 4.0]))
    assert_allclose(f_anti_utopia, array([136.0, 50.0]))
