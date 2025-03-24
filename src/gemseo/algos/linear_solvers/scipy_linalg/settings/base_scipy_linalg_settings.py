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
"""Settings for the SciPy linear solvers."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import Annotated

from numpy import ndarray  # noqa: TC002
from pydantic import AliasChoices
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import WithJsonSchema
from scipy.sparse import sparray  # noqa: TC002
from scipy.sparse.linalg import LinearOperator  # noqa: TC002

from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,
)
from gemseo.utils.compatibility.scipy import SCIPY_LOWER_THAN_1_12


class BaseSciPyLinalgSettingsBase(BaseLinearSolverSettings):
    """The settings of the SciPy GMRES algorithm."""

    _TARGET_CLASS_NAME = "TFQMR"

    atol: NonNegativeFloat = Field(
        default=0.0,
        description="""The absolute tolerance.

Algorithm stops if norm(b - A @ x) <= max(rtol*norm(b), atol).""",
    )

    if SCIPY_LOWER_THAN_1_12:
        tol: PositiveFloat = Field(
            alias="rtol",
            default=1e-12,
            description="""The relative tolerance.

Algorithm stops if norm(b - A @ x) <= max(rtol*norm(b), atol).""",
        )  # pragma: no cover
    else:
        rtol: PositiveFloat = Field(
            default=1e-12,
            description="""The relative tolerance.

Algorithm stops if norm(b - A @ x) <= max(rtol*norm(b), atol).""",
        )

    callback: Annotated[Callable, WithJsonSchema({})] | None = Field(
        default=None,
        description="""The user-supplied function to call after each iteration.

It is called as callback(xk), where xk is the current solution vector.
If ``None``, no function is called.""",
    )

    maxiter: PositiveInt = Field(
        default=1000,
        validation_alias=AliasChoices("max_iter", "maxiter"),
        description="Maximum number of iterations.",
    )

    x0: ndarray | None = Field(
        default=None,
        description="""Starting guess for the solution.

If ``None``, start from a matrix of zeros.""",
    )

    M: LinearOperator | ndarray | sparray | None = Field(
        default=None,
        validation_alias=AliasChoices("M", "preconditioner"),
        description="The preconditioner approximating the inverse of A.",
    )
