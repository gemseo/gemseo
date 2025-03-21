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

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import NonNegativeInt  # noqa: TC002
from strenum import StrEnum

from gemseo.algos.linear_solvers.base_linear_solver_settings import (  # noqa: TC001
    BaseLinearSolverSettings,
)
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.core.coupling_structure import CouplingStructure  # noqa: TC001
from gemseo.typing import StrKeyMapping  # noqa: TC001

LinearSolver = StrEnum("LinearSolver", names=LinearSolverLibraryFactory().algorithms)


class BaseMDASettings(BaseModel):
    """The base settings class for MDA algorithms."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )

    coupling_structure: CouplingStructure | None = Field(
        default=None,
        description="""The coupling structure to be used by the MDA.

If ``None``, the coupling structure is created from the disciplines.""",
    )

    linear_solver: LinearSolver = Field(
        default=LinearSolver.DEFAULT, description="""The name of the linear solver."""
    )

    linear_solver_settings: StrKeyMapping | BaseLinearSolverSettings = Field(
        default_factory=dict,
        description="""The settings passed to the linear solver factory.""",
    )

    linear_solver_tolerance: NonNegativeFloat = Field(
        default=1e-12,
        description="""The linear solver tolerance.

Linear solvers are used to compute the total derivatives.""",
    )

    log_convergence: bool = Field(
        default=False,
        description="""Whether to log the MDA convergence.

The log displays the normalized residual norm.""",
    )

    max_mda_iter: NonNegativeInt = Field(
        default=20,
        description="""The maximum number of iterations for the MDA algorithm.

If 0,
evaluate the coupling variables without trying to solve the coupling equations.""",
    )

    max_consecutive_unsuccessful_iterations: NonNegativeInt = Field(
        default=8,
        description="""The maximum number of consecutive unsuccessful iterations.

Iterations are considered unsuccessful if the normalized residual norm increases.""",
    )

    name: str = Field(
        default="",
        description="""The name to be given to the MDA.

If empty, use the name of the class.""",
    )

    tolerance: NonNegativeFloat = Field(
        default=1e-6,
        description="""The tolerance for the MDA algorithm residual.

The available normalization strategies for the residual
are described in :attr:`.BaseMDA.ResidualScaling`.""",
    )

    use_lu_fact: bool = Field(
        default=False,
        description="""Whether to perform an LU factorization.

The factorization is used to solve the linear systems arising in the computation
of the total derivatives. Since there are possibly several right-hand side, if
affordable, such a factorization may improve the solution time.""",
    )

    warm_start: bool = Field(
        default=False,
        description="""Whether to warm start the execution of the MDA algorithm.

The warm start strategy consists in using the last cached values for the
coupling variables as an initial guess. This is expected to reduce the number
of iteration of the MDA algorithm required to reach convergence.""",
    )
