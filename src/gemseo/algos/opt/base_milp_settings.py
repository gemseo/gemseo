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
"""Settings for the mixed-integer linear programming algorithms."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings


class BaseMILPSettings(BaseOptimizerSettings):
    """The mixed-integer linear programming library setting."""

    disp: bool = Field(
        default=False,
        description="""Whether to print optimization status during optimization.""",
    )

    mip_rel_gap: NonNegativeFloat = Field(
        default=0.0,
        description=(
            """The termination criterion for MIP solver.

            The solver will terminate when the gap between the primal objective value
            and the dual objective bound, scaled by the primal objective value, is <=
            mip_rel_gap.
            """
        ),
    )

    node_limit: PositiveInt = Field(
        default=1_000,
        description="""The maximum number of nodes to solve before stopping.""",
    )

    presolve: bool = Field(
        default=True,
        description=(
            """Whether to perform a preliminary analysis on the problem before solving.

            It attempts to detect infeasibility, unboundedness or problem
            simplifications.
            """
        ),
    )
