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
"""Settings for MDA algorithms."""

from __future__ import annotations

from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from strenum import StrEnum

from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.mda.base_mda_settings import BaseMDASettings

LinearSolver = StrEnum("LinearSolver", names=LinearSolverLibraryFactory().algorithms)


class BaseMDASolverSettings(BaseMDASettings):
    """The base settings class for MDA algorithms."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    acceleration_method: AccelerationMethod = Field(
        default=AccelerationMethod.NONE,
        description=(
            """The acceleration method used within the fixed point iterations."""
        ),
    )

    over_relaxation_factor: NonNegativeFloat = Field(
        default=1.0, le=2.0, description="""The over-relaxation factor."""
    )
