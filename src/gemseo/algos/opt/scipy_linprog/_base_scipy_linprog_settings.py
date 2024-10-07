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
"""Settings for the SciPy linear programming algorithms."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TCH002
from pydantic import PositiveInt  # noqa:TCH002
from pydantic import model_validator

from gemseo.algos.opt._base_optimization_library_settings import (
    BaseOptimizationLibrarySettings,
)


class BaseSciPyLinProgSettings(BaseOptimizationLibrarySettings):
    """The SciPy linear programming library setting."""

    autoscale: bool = Field(
        default=False,
        description="""Whether to perform auto-scaling of the constraints.""",
    )

    disp: bool = Field(
        default=False,
        description="""Whether to print convergence messages.""",
    )

    maxiter: PositiveInt = Field(
        default=1_000_000,
        description="""Maximum number of iterations to perform.""",
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

    rr: bool = Field(
        default=True,
        description="""Whether to remove linearly dependent equality-constraints.""",
    )

    rr_method: str | None = Field(
        default=None,
        description=(
            """The method to remove redundancy, either 'SVD', 'pivot' or 'ID'.

            If ``None``, use “svd” if the matrix is nearly full rank. If not, uses
            “pivot”. The behavior of this default is subject to change without prior
            notice.
            """
        ),
    )

    tol: NonNegativeFloat | None = Field(
        default=None,
        description=(
            """The tolerance below which a residual is considered to be exactly zero.
            If ``None``, a default value specific to each algorithm will be used.
            """
        ),
    )

    @model_validator(mode="after")
    def use_algorithm_default_tol(self):
        """Remove the ``tol`` attribute if ``None``.

        If the option is not provided, a default value specific to each algorithm will
        be used.
        """
        if self.tol is None:
            del self.tol
        return self
