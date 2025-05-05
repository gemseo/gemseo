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

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import model_validator
from strenum import StrEnum

from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,  # noqa: TC001
)
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.mda.base_parallel_mda_settings import BaseParallelMDASettings
from gemseo.typing import StrKeyMapping  # noqa: TC001
from gemseo.utils.pydantic import copy_field

if TYPE_CHECKING:
    from typing_extensions import Self

LinearSolver = StrEnum("LinearSolver", names=LinearSolverLibraryFactory().algorithms)


class MDANewtonRaphson_Settings(BaseParallelMDASettings):  # noqa: N801
    """The settings for :class:`.MDANewtonRaphson`."""

    _TARGET_CLASS_NAME = "MDANewtonRaphson"

    execute_before_linearizing: bool = copy_field(
        "execute_before_linearizing",
        BaseParallelMDASettings,
        default=False,
    )

    newton_linear_solver_name: LinearSolver = Field(
        default=LinearSolver.DEFAULT,
        description="""The name of the linear solver for the Newton method.

This field is ignored when ``newton_linear_solver_settings`` is a Pydantic model.""",
    )

    newton_linear_solver_settings: StrKeyMapping | BaseLinearSolverSettings = Field(
        default_factory=dict,
        description="""The settings for the Newton linear solver.""",
    )

    @model_validator(mode="after")
    def __newton_linear_solver_settings_to_pydantic_model(self) -> Self:
        """Convert MDA settings into a Pydantic model."""
        if isinstance(self.newton_linear_solver_settings, Mapping):
            factory = LinearSolverLibraryFactory()
            library_name = factory.algo_names_to_libraries[
                self.newton_linear_solver_name
            ]
            settings_model = (
                factory.get_class(library_name)
                .ALGORITHM_INFOS[self.newton_linear_solver_name]
                .Settings
            )
            self.newton_linear_solver_settings = settings_model(
                **self.newton_linear_solver_settings
            )
        return self
