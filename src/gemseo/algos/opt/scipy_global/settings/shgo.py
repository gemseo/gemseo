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
"""Settings for the SciPy SHGO algorithm."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from gemseo.algos.opt.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.typing import StrKeyMapping  # noqa: TC001


class SHGO_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy differential evolution setting."""

    _TARGET_CLASS_NAME = "SHGO"

    n: NonNegativeInt = Field(
        default=100,
        description=(
            """The number of samples used to construct the simplicial complex."""
        ),
    )

    iters: NonNegativeInt = Field(
        default=1,
        description=(
            """The number of iterations used to construct the simplicial complex."""
        ),
    )

    options: StrKeyMapping = Field(
        default_factory=dict,
        description="""The options for the local optimization algorithm.""",
    )

    sampling_method: str = Field(
        default="simplicial",
        description="""The sampling method.""",
    )

    workers: PositiveInt = Field(
        default=1,
        description="""The number workers to parallelize on.""",
    )
