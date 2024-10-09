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
"""Settings for the SciPy Nelder-Mead algorithm."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TCH003
from typing import Any

from numpy import asarray
from pydantic import Field
from pydantic import NonNegativeInt  # noqa:TCH002
from pydantic import model_validator

from gemseo.algos.opt.scipy_local._base_scipy_local_settings import (
    BaseScipyLocalSettings,
)
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TCH001


class NelderMeadSettings(BaseScipyLocalSettings):
    """Settings for the SciPy Nelder-Mead algorithm."""

    maxiter: NonNegativeInt = Field(
        default=1_000,
        description="""The maximum number of algorithm iterations.""",
    )

    maxfev: NonNegativeInt = Field(
        default=1_000,
        description="""The maximum number of function evaluations.""",
    )

    return_all: bool = Field(
        default=False,
        description=(
            "Whether to return a list of the best solution at each of the iterations."
        ),
    )

    initial_simplex: Sequence[float] | NDArrayPydantic[float] | None = Field(
        default=None,
        description=(
            """The initial simplex.

            If provided, the expected shape is `(N+1, N)` where `N` is the problem
            dimension.
            """
        ),
    )

    adaptive: bool = Field(
        default=False,
        description=(
            "Whether to adapt the algorithm parameters to dimensionality of problem."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_initial_simplex(cls, data: Any) -> Any:
        """Cast the initial simplex as a NumPy array if relevant."""
        if "initial_simplex" in data and data["initial_simplex"] is not None:
            data["initial_simplex"] = asarray(data["initial_simplex"])
        return data