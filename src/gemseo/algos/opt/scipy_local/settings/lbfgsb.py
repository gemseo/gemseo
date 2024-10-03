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
"""Settings for the SciPy L-BFGS-B algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TCH002
from pydantic import NonNegativeInt  # noqa: TCH002
from pydantic import PositiveInt  # noqa: TCH002

from gemseo.algos.opt.gradient_based_algorithm_settings import (
    GradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.scipy_local._base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class LBFGSBSettings(BaseScipyLocalSettings, GradientBasedAlgorithmSettings):
    """Settings for the SciPy L-BFGS-B algorithm."""

    maxcor: PositiveInt = Field(
        default=20,
        description=(
            """The maximum number of corrections for the limited memory matrix."""
        ),
    )

    gtol: NonNegativeFloat = Field(
        default=1e-6,
        description=(
            "The precision goal for the projected gradient value to stop the algorithm."
        ),
    )

    eps: NonNegativeFloat = Field(
        default=1e-8,
        description=(
            """The absolute step size forforward differences.

            The forward differences is used to approximate the Jacobian if not provided.
            """
        ),
    )

    maxfun: NonNegativeInt = Field(
        default=1_000,
        description="""The maximum number of function evaluations.""",
    )

    maxiter: NonNegativeInt = Field(
        default=1_000,
        description="""The maximum number of algrotihm iterations.""",
    )

    iprint: int = Field(
        default=-1,
        description=(
            """The flag to control the frequency of output.

            Default is no output.
            """
        ),
    )

    maxls: PositiveInt = Field(
        default=20,
        description="""The maximum number of line search steps per iteration.""",
    )
