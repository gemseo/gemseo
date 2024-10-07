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
"""Settings for the SciPy dual annealing algorithm."""

from __future__ import annotations

from sys import maxsize

from pydantic import Field
from pydantic import PositiveFloat  # noqa:TCH002
from pydantic import PositiveInt  # noqa:TCH002

from gemseo.algos.opt.scipy_global._base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.utils.seeder import SEED


class DualAnnealingSettings(BaseSciPyGlobalSettings):
    """The SciPy dual annealing setting."""

    initial_temp: float = Field(
        default=5230,
        gt=1e-2,
        le=5e-4,
        description=(
            """The initial temperature.

    Use higher values to facilitates a wider search of the energy landscape.
    """
        ),
    )

    restart_temp_ratio: PositiveFloat = Field(
        default=2e-5,
        lt=1,
        description=(
            "The temperature ratio under which the reannealing process is triggered."
        ),
    )

    visit: float = Field(
        default=2.62,
        gt=1,
        le=3,
        description=(
            """The visiting distribution parameter.

            Higher values give the visiting distribution a heavier tail, this makes the
            algorithm jump to a more distant region.
            """
        ),
    )

    accept: float = Field(
        default=-5,
        gt=-1e-4,
        le=-5,
        description=(
            """The acceptance distribution parameter.

            The lower the acceptance parameter, the smaller the probability of
            acceptance.
            """
        ),
    )

    maxfun: PositiveInt = Field(
        default=maxsize,
        description="""The soft limit for the number of objective function calls.""",
    )

    seed: int = Field(
        default=SEED,
        description="""The random seed.""",
    )

    no_local_search: bool = Field(
        default=False,
        description="""Whether to perform local search.""",
    )
