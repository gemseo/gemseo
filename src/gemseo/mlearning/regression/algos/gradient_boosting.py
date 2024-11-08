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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Gradient boosting for regression."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from sklearn.ensemble import GradientBoostingRegressor as SKLGradientBoosting

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.gradient_boosting_settings import (
    GradientBoostingRegressor_Settings,
)

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class GradientBoostingRegressor(BaseRegressor):
    """Gradient boosting for regression."""

    LIBRARY: ClassVar[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "GradientBoostingRegressor"

    Settings: ClassVar[type[GradientBoostingRegressor_Settings]] = (
        GradientBoostingRegressor_Settings
    )

    def _post_init(self):
        super()._post_init()
        self.__algo = {
            "n_estimators": self._settings.n_estimators,
            "parameters": self._settings.parameters,
        }
        self.algo = []

    def _fit(
        self,
        input_data: NumberArray,
        output_data: NumberArray,
    ) -> None:
        for _output_data in output_data.T:
            self.algo.append(
                SKLGradientBoosting(
                    n_estimators=self.__algo["n_estimators"],
                    **self.__algo["parameters"],
                )
            )
            self.algo[-1].fit(input_data, _output_data)

    def _predict(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        return array([algo.predict(input_data) for algo in self.algo]).T
