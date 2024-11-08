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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Support vector machine for regression."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from sklearn.svm import SVR

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.svm_settings import SVMRegressor_Settings

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class SVMRegressor(BaseRegressor):
    """Support vector machine for regression."""

    LIBRARY: ClassVar[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "SVMRegression"

    Settings: ClassVar[type[SVMRegressor_Settings]] = SVMRegressor_Settings

    def _post_init(self):
        super()._post_init()
        self.__algo = {
            "kernel": self._settings.kernel,
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
                SVR(
                    kernel=self.__algo["kernel"],
                    **self.__algo["parameters"],
                )
            )
            self.algo[-1].fit(input_data, _output_data)

    def _predict(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        return array([algo.predict(input_data) for algo in self.algo]).T
