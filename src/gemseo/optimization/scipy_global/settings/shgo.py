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

from typing import TYPE_CHECKING
from typing import Literal

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    BaseSciPyGlobalSettings,
)
from gemseo.optimization.scipy_global.settings.base_scipy_global_settings import (
    _check_database_with_processes,
)
from gemseo.util.typing import StrKeyMapping  # noqa: TC001

if TYPE_CHECKING:
    from typing import Self


class SHGO_Settings(BaseSciPyGlobalSettings):  # noqa: N801
    """The SciPy differential evolution setting."""

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

    workers: PositiveInt | Literal[-1] = Field(
        default=1,
        description="""The number of workers to parallelize on.

If -1, use all the CPUs.
The evaluations made by the worker processes are stored in the database
and count towards the termination criteria,
as if they were made in the main process.
When the run stops in the middle of a batch, e.g. on `max_iter`,
the evaluations already started finish in the background.
The workers cannot be used when `use_database` is `False`.""",
    )

    @model_validator(mode="after")
    def __check_database_with_workers(self) -> Self:
        """Check that the database is used when there are several workers.

        Raises:
            ValueError: When `use_database` is `False` and `workers` is not 1.
        """
        _check_database_with_processes(self.use_database, self.workers)
        return self
