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
"""Settings for the SciPy differential evolution algorithm."""

from __future__ import annotations

from typing import ClassVar

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.algos.opt.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001
from gemseo.utils.seeder import SEED


class DIFFERENTIAL_EVOLUTION_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy differential evolution setting."""

    _TARGET_CLASS_NAME = "DIFFERENTIAL_EVOLUTION"

    strategy: str = Field(
        default="best1bin",
        description="""The differential evolution strategy to use.""",
    )

    popsize: PositiveInt = Field(
        default=15,
        description="""The multiplier for setting the total population size.""",
    )

    tol: NonNegativeFloat = Field(
        default=1e-2,
        description="The relative tolerance for convergence.",
    )

    mutation: float | tuple[float, float] = Field(
        default=(0.5, 1.0),
        description="""The mutation constant.

If specified as a float it should be in the range [0, 2]. If specified as a
tuple(min, max) dithering is employed.""",
    )

    recombination: NonNegativeFloat = Field(
        default=0.7,
        le=1.0,
        description="The recombination constant.",
    )

    seed: int = Field(
        default=SEED,
        description="""The random seed.""",
    )

    disp: bool = Field(
        default=False,
        description="""Whether to print convergence messages.""",
    )

    polish: bool = Field(
        default=True,
        description="""Whether to polish the best population member at the end.""",
    )

    init: str | NDArrayPydantic[float] = Field(
        default="latinhypercube",
        description="""The method to perform the population initialization as a string
         or the initial population as an array.""",
    )

    atol: NonNegativeFloat = Field(
        default=0.0,
        description="The absolute tolerance for convergence.",
    )

    updating: str = Field(
        default="immediate",
        description="""The best solution vector updating strategy.""",
    )

    workers: int = Field(
        default=1,
        description=(
            """The number of parallel workers the population is subdivided in."""
        ),
    )

    _redundant_settings: ClassVar[list[str]] = ["maxiter"]
