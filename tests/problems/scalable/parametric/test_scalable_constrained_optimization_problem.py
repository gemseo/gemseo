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

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import arange
from numpy import array
from numpy import atleast_2d
from numpy import diag
from numpy import mean as np_mean
from numpy import ones_like

from gemseo import MDODiscipline
from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdofunctions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Iterable


class ScalableDiscipline(MDODiscipline):
    def __init__(self, p: float):
        self.p = p
        super().__init__()
        self.input_grammar.update_from_names(["x"])
        self.output_grammar.update_from_names(["f", "g"])

    def _run(self) -> None:
        x = self.local_data["x"]
        self.local_data["f"] = array([np_mean(x)]) / (len(x) + 1) * 2 * 100
        self.local_data["g"] = ((arange(len(x)) + 1) / x) ** self.p - 1.0

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        x = self.local_data["x"]
        self._init_jacobian()
        self.jac["f"]["x"] = atleast_2d(ones_like(x)) / len(x) / (len(x) + 1) * 2 * 100
        self.jac["g"]["x"] = atleast_2d(
            diag(
                self.p
                * ((arange(len(x)) + 1) / x) ** (self.p - 1.0)
                * (-(arange(len(x)) + 1) / x**2)
            )
        )


@pytest.fixture(params=[10, 50, 100])
def n(request):
    return request.param


@pytest.fixture(params=[MDOFunction.ConstraintType.INEQ, MDOFunction.ConstraintType.EQ])
def constraint_kind(request):
    return request.param


@pytest.fixture(
    params=[
        "SLSQP",
        "NLOPT_SLSQP",
        "Augmented_Lagrangian_order_0",
        "Augmented_Lagrangian_order_1",
    ]
)
def algo(request):
    return request.param


@pytest.fixture(params=[1, 2])
def scalable_optimization_problem_scenario(request, n, constraint_kind):
    disc = ScalableDiscipline(request.param)
    ds = DesignSpace()
    ds.add_variable("x", size=n, l_b=0.1, u_b=n, value=n)
    scenario = create_scenario(
        disciplines=[disc],
        formulation="DisciplinaryOpt",
        design_space=ds,
        objective_name="f",
    )
    scenario.add_constraint("g", constraint_kind)
    return scenario


def test_resolution(scalable_optimization_problem_scenario, algo, n, constraint_kind):
    if constraint_kind == MDOFunction.ConstraintType.EQ and algo in [
        "SLSQP",
        "NLOPT_SLSQP",
    ]:
        pytest.skip("SLSQP is not well suited for non-linear equality constraints. ")
    options = {
        "max_iter": 1000,
        "algo": algo,
        "normalize_design_space": True,
        "algo_options": {"stop_crit_n_x": 2, "ftol_rel": 1e-6},
    }
    if "Augmented_Lagrangian" in algo:
        options["algo_options"]["sub_solver_algorithm"] = "L-BFGS-B"
        options["algo_options"]["sub_problem_options"] = {"max_iter": 300}
    scalable_optimization_problem_scenario.execute(options)
    assert pytest.approx(
        scalable_optimization_problem_scenario.formulation.opt_problem.solution.x_opt,
        rel=1e-2,
    ) == (arange(n) + 1)
