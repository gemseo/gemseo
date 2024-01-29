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
"""Settings for post-processing."""

from typing import Optional

from pydantic import Field

from gemseo.post.base_post_settings import BasePostSettings
from gemseo.utils.pydantic import update_field


class Settings(BasePostSettings):  # noqa: D101
    iteration: Optional[int] = Field(
        None,
        description="The iteration to plot the sensitivities. "
        "Can use either positive or negative indexing, "
        "e.g. ``5`` for the 5-th iteration or ``-2`` for the penultimate one. "
        "If ``None``, use the iteration of the optimum.",
    )
    scale_gradients: bool = Field(
        False,
        description="Whether to normalize each gradient w.r.t. the design variables.",
    )
    compute_missing_gradients: bool = Field(
        False,
        description="Whether to compute the gradients at the selected iteration "
        "if they were not computed by the algorithm."
        "\\n\\n.. warning::\\n"
        "Activating this option may add considerable computation time depending "
        "on the cost of the gradient evaluation. "
        "This option will not compute the gradients if the "
        ":class:`.OptimizationProblem` instance was imported from an HDF5 "
        "file. This option requires an :class:`.OptimizationProblem` with a "
        "gradient-based algorithm.",
    )


update_field(Settings, "fig_size", default=(10.0, 10.0))
