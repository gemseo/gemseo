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

from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa: TC002
from pydantic import PositiveInt  # noqa: TC002

from gemseo.algos.opt.base_gradient_based_algorithm_settings import (
    BaseGradientBasedAlgorithmSettings,
)
from gemseo.algos.opt.scipy_local.settings.base_scipy_local_settings import (
    BaseScipyLocalSettings,
)


class L_BFGS_B_Settings(BaseScipyLocalSettings, BaseGradientBasedAlgorithmSettings):  # noqa: N801
    """Settings for the SciPy L-BFGS-B algorithm."""

    _TARGET_CLASS_NAME = "L-BFGS-B"

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

    iprint: int = Field(
        default=-1,
        description="""The flag to control the frequency of output.

Default is no output.""",
    )

    maxls: PositiveInt = Field(
        default=20,
        description="""The maximum number of line search steps per iteration.""",
    )

    _redundant_settings: ClassVar[list[str]] = ["eps", "maxfun", "maxiter"]
