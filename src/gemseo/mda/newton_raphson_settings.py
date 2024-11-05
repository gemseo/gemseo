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
"""Settings for NewtonRaphson MDA."""

from __future__ import annotations

from pydantic import Field
from strenum import StrEnum

from gemseo.algos.linear_solvers.base_linear_solver_settings import (  # noqa: TCH001
    BaseLinearSolverSettings,
)
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.mda.base_mda_root_settings import BaseMDARootSettings
from gemseo.typing import StrKeyMapping  # noqa: TCH001
from gemseo.utils.pydantic import copy_field

LinearSolver = StrEnum("LinearSolver", names=LinearSolverLibraryFactory().algorithms)


class MDANewtonRaphson_Settings(BaseMDARootSettings):  # noqa: N801
    """The settings for :class:`.MDANewtonRaphson`."""

    execute_before_linearizing: bool = copy_field(
        "execute_before_linearizing",
        BaseMDARootSettings,
        default=False,
    )

    newton_linear_solver_name: LinearSolver = Field(
        default=LinearSolver.DEFAULT,
        description="""The name of the linear solver for the Newton method.""",
    )

    newton_linear_solver_settings: StrKeyMapping | BaseLinearSolverSettings = Field(
        default_factory=dict,
        description="""The settings for the Newton linear solver.""",
    )
