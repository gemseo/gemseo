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

from typing import ClassVar

from pydantic import Field
from strenum import StrEnum

from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,  # noqa: TC001
)
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.linear_solvers.scipy_linalg import LGMRES_Settings
from gemseo.mda.base_parallel_solver_settings import BaseMDAParallelSolverSettings
from gemseo.typing import StrKeyMapping  # noqa: TC001

LinearSolver = StrEnum("LinearSolver", names=LinearSolverLibraryFactory().algorithms)


class MDANewtonRaphson_Settings(BaseMDAParallelSolverSettings):  # noqa: N801
    """The settings for [MDANewtonRaphson][gemseo.mda.newton_raphson.MDANewtonRaphson]."""  # noqa: E501

    _INHERITED_FIELD_DEFAULTS: ClassVar[StrKeyMapping] = {
        "execute_before_linearizing": False
    }

    newton_linear_solver_settings: BaseLinearSolverSettings = Field(
        default_factory=LGMRES_Settings,
        description="""The settings for the Newton linear solver.""",
    )
