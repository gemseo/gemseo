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
"""Settings for the DOE library."""

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa:TC003
from collections.abc import Sequence  # noqa:TC003
from typing import TYPE_CHECKING  # noqa:TC003
from typing import Annotated
from typing import Any  # noqa:TC003

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002
from pydantic import WithJsonSchema
from pydantic import model_validator

from gemseo.algos.base_driver_settings import BaseDriverSettings
from gemseo.algos.evaluation_problem import EvaluationType  # noqa:TC001

if TYPE_CHECKING:
    from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class BaseDOESettings(BaseDriverSettings):
    """The settings for the ``DOELibrary``."""

    eval_func: bool = Field(
        default=True,
        description="Whether to sample the function computing the output values.",
    )

    eval_jac: bool = Field(
        default=False,
        description="Whether to sample the function computing the Jacobian data.",
    )

    n_processes: PositiveInt = Field(
        default=1,
        description="The maximum number of processes to parallelize the execution.",
    )

    wait_time_between_samples: NonNegativeFloat = Field(
        default=0.0,
        description="The time to wait between each sample evaluation, in seconds.",
    )

    callbacks: Sequence[
        Annotated[Callable[[int, EvaluationType], Any], WithJsonSchema({})]
    ] = Field(
        default=(),
        description="""The functions called after evaluating the function of interest.

A callback must be called as ``callback(sample_index, (output, Jacobian))``.""",
    )

    preprocessors: Sequence[Annotated[Callable[[int], Any], WithJsonSchema({})]] = (
        Field(
            default=(),
            description="""The functions called
before evaluating the function of interest.

A preprocessor must be called as ``preprocessor(sample_index)``.

This option is not compatible with the vectorization of functions evaluations.
""",
        )
    )

    normalize_design_space: bool = Field(
        default=False,
        description="Whether to normalize the design space variables between 0 and 1.",
    )

    vectorize: bool = Field(
        default=False,
        description="Whether to vectorize the functions evaluations.",
    )

    @model_validator(mode="after")
    def __check(self) -> Self:
        """Check that field values are compatible.

        Raises:
            NotImplementedError: When combining parallelization and vectorization.
        """
        if self.wait_time_between_samples > 0 and self.n_processes == 1:
            LOGGER.warning(
                "The option 'wait_time_between_samples' is ignored "
                "when the option 'n_processes' is 1 (serial mode)."
            )

        if self.vectorize and self.n_processes > 1:
            msg = "Vectorization in parallel is not supported."
            raise NotImplementedError(msg)

        if self.preprocessors and self.vectorize:
            msg = "Combining preprocessors and vectorization is not supported."
            raise NotImplementedError(msg)

        return self
