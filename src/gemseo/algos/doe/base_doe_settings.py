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

from collections.abc import Callable  # noqa:TC003
from collections.abc import Sequence  # noqa:TC003
from typing import Annotated
from typing import Any  # noqa:TC003

from pydantic import Field
from pydantic import NonNegativeFloat  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002
from pydantic import WithJsonSchema

from gemseo.algos.base_driver_settings import BaseDriverSettings
from gemseo.algos.evaluation_problem import EvaluationType  # noqa:TC001


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
        description="""The functions to be evaluated after each functions evaluation.

The functions evaluation is done by
:meth:`.OptimizationProblem.evaluate_functions` and the callback must be
called as ``callback(index, (output, Jacobian))``.""",
    )

    normalize_design_space: bool = Field(
        default=False,
        description="Whether to normalize the design space variables between 0 and 1.",
    )
