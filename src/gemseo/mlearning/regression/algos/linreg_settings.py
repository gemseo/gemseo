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
"""Settings of the linear regressor."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import NonNegativeInt

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from gemseo.utils.seeder import SEED


class LinearRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the linear regressor."""

    fit_intercept: bool = Field(
        default=True, description="Whether to fit the intercept."
    )

    penalty_level: NonNegativeFloat = Field(
        default=0.0,
        description="""The penalty level greater or equal to 0.

If zero, there is no penalty.""",
    )

    l2_penalty_ratio: NonNegativeFloat = Field(
        default=1.0,
        description="""The penalty ratio related to the l2 regularization.

If 1, use the Ridge penalty.
If 0, use the Lasso penalty.
Between 0 and 1, use the ElasticNet penalty.""",
    )

    random_state: NonNegativeInt | None = Field(
        default=SEED,
        description="""The random state parameter in the case of a penalty.

If ``None``, use the global random state instance from ``numpy.random``.
Creating the model multiple times will produce different results.
If ``int``, use a new random number generator seeded by this integer.
This will produce the same results.""",
    )
