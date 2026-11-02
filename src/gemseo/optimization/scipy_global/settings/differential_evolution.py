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

from enum import StrEnum
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    _check_database_with_processes,
)
from gemseo.util import seeder
from gemseo.util.pydantic_ndarray import NDArrayPydantic  # noqa: TC001

if TYPE_CHECKING:
    from typing import Self


class DifferentialEvolutionUpdating(StrEnum):
    """The best solution vector updating strategy of the differential evolution."""

    IMMEDIATE = "immediate"
    """The best solution vector is updated after each trial vector is evaluated."""

    DEFERRED = "deferred"
    """The best solution vector is updated once per generation."""


class DIFFERENTIAL_EVOLUTION_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy differential evolution setting."""

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
        default=seeder.seed,
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

    updating: DifferentialEvolutionUpdating | None = Field(
        default=None,
        description="""The best solution vector updating strategy.

If `None`, use `IMMEDIATE` when `workers` is 1 and `DEFERRED` otherwise.
`IMMEDIATE` cannot be used when `workers` is not 1,
even when the workers end up unused because of the constraints.""",
    )

    workers: PositiveInt | Literal[-1] = Field(
        default=1,
        description=(
            """The number of parallel workers the population is subdivided in.

If -1, use all the CPUs.
The evaluations made by the worker processes are stored in the database
and count towards the termination criteria,
as if they were made in the main process.
SciPy evaluates the constraints in the main process,
one point after the other,
outside `workers`.
So with constraints,
the workers are not used and a warning is logged.
When the run stops in the middle of a batch, e.g. on `max_iter`,
the evaluations already started finish in the background.
The workers cannot be used when `use_database` is `False`."""
        ),
    )

    _redundant_settings: ClassVar[list[str]] = ["maxiter"]

    @model_validator(mode="after")
    def __check_updating_with_workers(self) -> Self:
        """Check that the updating strategy is not immediate with several workers.

        SciPy evaluates the population in parallel
        only with the deferred updating strategy.

        Raises:
            ValueError: When `updating` is `IMMEDIATE` and `workers` is not 1.
        """
        if (
            self.updating == DifferentialEvolutionUpdating.IMMEDIATE
            and self.workers != 1
        ):
            workers = (
                "all the CPUs" if self.workers == -1 else f"{self.workers} workers"
            )
            msg = (
                'The updating strategy "immediate" cannot be used '
                f'with {workers}; use "deferred" or None.'
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def __check_database_with_workers(self) -> Self:
        """Check that the database is used when there are several workers.

        Raises:
            ValueError: When `use_database` is `False` and `workers` is not 1.
        """
        _check_database_with_processes(self.use_database, self.workers)
        return self
