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
"""Settings for the driver library."""

from __future__ import annotations

from pydantic import Field
from pydantic.types import NonNegativeFloat  # noqa: TC002

from gemseo.algos.base_algorithm_settings import BaseAlgorithmSettings


class BaseDriverSettings(BaseAlgorithmSettings):
    """The common parameters for all driver libraries."""

    enable_progress_bar: bool | None = Field(
        default=None,
        description=(
            """Whether to enable the progress bar in the optimization log.

If ``None``, use the global value of ``enable_progress_bar`` (see the
``configure`` function to change it globally)."""
        ),
    )

    eq_tolerance: NonNegativeFloat = Field(
        default=1e-2,
        description="""The tolerance on the equality constraints.""",
    )

    ineq_tolerance: NonNegativeFloat = Field(
        default=1e-4,
        description="""The tolerance on the inequality constraints.""",
    )

    log_problem: bool = Field(
        default=True,
        description="""Whether to log the definition and result of the problem.""",
    )

    max_time: NonNegativeFloat = Field(
        default=0.0,
        description="""The maximum runtime in seconds, disabled if 0.""",
    )

    normalize_design_space: bool = Field(
        default=True,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    reset_iteration_counters: bool = Field(
        default=True,
        description=(
            """Whether to reset the iteration counters before each execution."""
        ),
    )

    round_ints: bool = Field(
        default=True,
        description="""Whether to round the integer variables.""",
    )

    use_database: bool = Field(
        default=True,
        description="""Whether to wrap the functions in the database.""",
    )

    use_one_line_progress_bar: bool = Field(
        default=False,
        description="""Whether to log the progress bar on a single line.""",
    )

    store_jacobian: bool = Field(
        default=True,
        description="""Whether to store the Jacobian matrices in the database.

This argument is ignored when the ``use_database`` option is ``False``.
If a gradient-based algorithm is used,
this option cannot be set along with kkt options.""",
    )
