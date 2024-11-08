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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Random forest regression model.

Use an ensemble of decision trees.

Dependence
----------
The regression model relies on the ``RandomForestRegressor`` class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.RandomForestRegressor.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from sklearn.ensemble import RandomForestRegressor as SKLRandForest

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.random_forest_settings import (
    RandomForestRegressor_Settings,
)

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class RandomForestRegressor(BaseRegressor):
    """Random forest regression."""

    SHORT_ALGO_NAME: ClassVar[str] = "RF"
    LIBRARY: ClassVar[str] = "scikit-learn"

    Settings: ClassVar[type[RandomForestRegressor_Settings]] = (
        RandomForestRegressor_Settings
    )

    def _post_init(self):
        super()._post_init()
        self.algo = SKLRandForest(
            n_estimators=self._settings.n_estimators,
            random_state=self._settings.random_state,
            **self._settings.parameters,
        )

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        # SKLearn RandomForestRegressor does not like output
        # shape (n_samples, 1), prefers (n_samples,).
        # The shape (n_samples, n_outputs) with n_outputs >= 2 is fine.
        if output_data.shape[1] == 1:
            output_data = output_data[:, 0]
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return self.algo.predict(input_data).reshape((len(input_data), -1))
