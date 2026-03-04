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
from pydantic import PositiveInt

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings


class MultiStart_Settings(BaseOptimizerSettings):  # noqa: N801
    """The multi-start algorithm settings."""

    normalize_design_space: bool = Field(
        default=False,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    opt_algo_settings: BaseOptimizerSettings = Field(
        default_factory=SLSQP_Settings,
        description="""The settings of the sub-optimization algorithm.

If its field `max_iter` is not set explicitly,
this number is `int(max_iter/n_samples)`
where `n_samples` is deduced from `doe_algo_settings`.
""",
    )

    doe_algo_settings: BaseDOESettings = Field(
        default_factory=LHS_Settings,
        description="""The settings of the DOE algorithm.

The DOE algorithm is used to generate the sub-optimizations starting points.""",
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
