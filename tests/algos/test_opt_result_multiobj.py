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
"""Tests for the multi-objective optimization result."""

from __future__ import annotations

import pytest

from gemseo.algos.multiobjective_optimization_result import (
    MultiObjectiveOptimizationResult,
)
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.problems.multiobjective_optimization.binh_korn import BinhKorn


@pytest.fixture(scope="module")
def problem() -> OptimizationProblem:
    """The Binh-Korn optimization problem ready to be post-processed."""
    binh_korn = BinhKorn()
    OptimizationLibraryFactory().execute(
        binh_korn, algo_name="MNBI", max_iter=100, n_sub_optim=5, sub_optim_algo="SLSQP"
    )
    return binh_korn


def test_export_hdf(problem: OptimizationProblem, tmpdir):
    """Test the export and the import of the multi-objective optimization history."""
    out_file = tmpdir / "output.hdf"
    problem.to_hdf(out_file)
    read_opt_pb = OptimizationProblem.from_hdf(out_file)
    read_solution = read_opt_pb.solution
    assert isinstance(problem.solution, MultiObjectiveOptimizationResult)
    assert isinstance(read_solution, MultiObjectiveOptimizationResult)
    assert not read_opt_pb.is_mono_objective
    assert len(read_solution.pareto_front.f_optima) > 7


@pytest.mark.parametrize("solved", [True, False])
def test_str(problem, solved):
    """Test the string representation of the multi objective result."""
    opt = OptimizationLibraryFactory().create("MNBI")
    opt.problem = problem if solved else BinhKorn()
    result = opt._get_result(opt.problem, None, None)
    if solved:
        s_res = str(result)
        assert "The solution is feasible." in s_res
        assert "Pareto efficient solutions:" in s_res
    else:
        assert "Pareto efficient solutions:" not in str(result)

    assert ("Pareto front:" in repr(result)) is solved
    assert ("pareto_front" in str(result.to_dict().keys())) is solved
