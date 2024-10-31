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
"""Settings for the linear solvers."""

from __future__ import annotations

from pydantic import Field

from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings


class BaseLinearSolverSettings(BaseAlgorithmSettings):
    """The settings common to all the linear solver libraries."""

    store_residuals: bool = Field(
        default=False,
        description="Whether to store the residual norms at each iterations.",
    )

    use_ilu_precond: bool = Field(
        default=False,
        description="Whether to use an incomplete LU factorization as preconditioner.",
    )

    save_when_fail: bool = Field(
        default=False,
        description="Whether to use an incomplete LU factorization as preconditioner.",
    )
