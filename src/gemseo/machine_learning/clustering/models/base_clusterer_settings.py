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
"""Settings for the clustering models."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeInt  # noqa:TC002
from pydantic import PositiveInt  # noqa:TC002

from gemseo.machine_learning.core.models.unsupervised_settings import (
    BaseMLUnsupervisedModelSettings,
)
from gemseo.utils.seeder import SEED


class BaseClustererSettings(BaseMLUnsupervisedModelSettings):
    """The settings common to all the clustering models."""

    n_clusters: PositiveInt = Field(
        default=5, description="The number of clusters of the clustering model."
    )

    random_state: NonNegativeInt | None = Field(
        default=SEED,
        description="""The random state parameter.

If `None`, use the global random state instance from `numpy.random`.
Creating the model multiple times will produce different results.
If `int`, use a new random number generator seeded by this integer.
This will produce the same results.""",
    )
