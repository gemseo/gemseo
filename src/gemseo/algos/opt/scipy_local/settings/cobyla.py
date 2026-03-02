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
"""Settings for the SciPy COBYLA algorithm."""

from __future__ import annotations

from typing import ClassVar

from numpy import inf
from pydantic import Field
from pydantic import PositiveFloat  # noqa:TC002

from gemseo.algos.opt.scipy_local.settings.base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class COBYLA_Settings(BaseScipyLocalSettings):  # noqa: N801
    """Settings for the SciPy COBYLA algorithm."""

    f_target: float = Field(
        default=-inf,
        description="""The target value for the objective function.

The optimization procedure is terminated when the objective function value
of a feasible point is less than or equal to this target.""",
    )

    rhobeg: PositiveFloat = Field(
        default=1.0,
        description="""Reasonable initial changes to the variables.""",
    )

    catol: PositiveFloat | None = Field(
        default=None,
        description="""Tolerance (absolute) for constraint violations.""",
    )

    _redundant_settings: ClassVar[list[str]] = ["maxiter", "tol"]
