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

from gemseo import create_scenario
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt.factory import OPTIMIZATION_LIBRARY_FACTORY
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class ScalableDiscipline(Discipline):
    def __init__(self, p: float) -> None:
        """
        Args:
            p: The power used in the constraint.
        """
        self.p = p
        super().__init__()
        self.io.input_grammar.update_from_names(["x"])
        self.io.output_grammar.update_from_names(["f", "g"])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"]
        return {
            "f": array([np_mean(x)]) / (len(x) + 1) * 2 * 100,
            "g": ((arange(len(x)) + 1) / x) ** self.p - 1.0,
        }

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        x = self.io.data["x"]
        self._init_jacobian()
        self.jac["f"]["x"] = atleast_2d(ones_like(x)) / len(x) / (len(x) + 1) * 2 * 100
        self.jac["g"]["x"] = atleast_2d(
            diag(
                self.p
                * ((arange(len(x)) + 1) / x) ** (self.p - 1.0)
                * (-(arange(len(x)) + 1) / x**2)
            )
        )


@pytest.mark.parametrize("n", [10, 50, 100])
@pytest.mark.parametrize("p", [1, 2])
@pytest.mark.parametrize(
    "constraint_kind", [MDOFunction.ConstraintType.INEQ, MDOFunction.ConstraintType.EQ]
)
@pytest.mark.parametrize(
    "algo",
    [
        "SLSQP",
        "NLOPT_SLSQP",
        "Augmented_Lagrangian_Order_0",
        "Augmented_Lagrangian_Order_1",
    ],
)
def test_resolution(algo, n, p, constraint_kind) -> None:
    if constraint_kind == MDOFunction.ConstraintType.EQ and algo in {
        "SLSQP",
        "NLOPT_SLSQP",
    }:
        pytest.skip("SLSQP is not well suited for non-linear equality constraints")

    settings = OPTIMIZATION_LIBRARY_FACTORY.create_settings(
        algo,
        max_iter=100,
        normalize_design_space=True,
        stop_crit_n_x=2,
        ftol_rel=1e-6,
        **(
            {
                "sub_algorithm_settings": L_BFGS_B_Settings(max_iter=300),
            }
            if algo.startswith("Augmented_Lagrangian")
            else {}
        ),
    )

    ds = DesignSpace()
    ds.add_variable("x", size=n, lower_bound=0.1, upper_bound=n, value=n)

    scenario = create_scenario(
        [ScalableDiscipline(p)],
        "f",
        ds,
        formulation_name="DisciplinaryOpt",
    )
    scenario.add_constraint("g", constraint_type=constraint_kind)
    scenario.execute(settings)

    assert pytest.approx(
        scenario.formulation.problem.solution.x_opt,
        rel=3e-2,
    ) == (arange(n) + 1)
