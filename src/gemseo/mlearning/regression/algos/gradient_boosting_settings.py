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
"""Settings of the gradient boosting for regression."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveInt

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)


class GradientBoostingRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the gradient boosting for regression."""

    _TARGET_CLASS_NAME = "GradientBoostingRegressor"

    n_estimators: PositiveInt = Field(
        default=100, description="The number of boosting stages to perform."
    )
