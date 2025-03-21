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
"""Settings for the multi-start algorithm."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.typing import StrKeyMapping  # noqa: TC001


class MultiStart_Settings(BaseOptimizerSettings):  # noqa: N801
    """The multi-start algorithm settings."""

    _TARGET_CLASS_NAME = "MultiStart"

    normalize_design_space: bool = Field(
        default=False,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    n_start: PositiveInt = Field(
        default=5,
        description="""The number of sub-optimizations.""",
    )

    opt_algo_max_iter: NonNegativeInt = Field(
        default=0,
        description="""The maximum number of iterations for each sub-optimization.

If 0, this number is ``int(max_iter/n_start)``.""",
    )

    opt_algo_name: str = Field(
        default="SLSQP",
        description="""The name of the sub-optimization algorithm.""",
    )

    opt_algo_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings of the sub-optimization algorithm.""",
    )

    doe_algo_name: str = Field(
        default="LHS",
        description="""The name of the DOE algorithm.

The DOE algorthm is used to generate the sub-optimizations starting points.""",
    )

    doe_algo_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings of the DOE algorithm.""",
    )

    multistart_file_path: str | Path = Field(
        default="",
        description="""The database file path to save the local optima.

If empty, do not save the local optima.""",
    )

    n_processes: PositiveInt = Field(
        default=1,
        description=(
            "The maximum number of processes used to parallelize the sub-optimizations."
        ),
    )
