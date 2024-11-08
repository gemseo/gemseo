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
"""Multilayer perceptron (MLP)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

import sklearn.neural_network
from numpy import newaxis

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.mlp_settings import MLPRegressor_Settings

if TYPE_CHECKING:
    from gemseo.typing import NumberArray


class MLPRegressor(BaseRegressor):
    """MultiLayer perceptron (MLP)."""

    LIBRARY: ClassVar[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "MLP"

    Settings: ClassVar[type[MLPRegressor_Settings]] = MLPRegressor_Settings

    def _post_init(self):
        super()._post_init()
        self.algo = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=self._settings.hidden_layer_sizes,
            random_state=self._settings.random_state,
            **self._settings.parameters,
        )

    def _fit(
        self,
        input_data: NumberArray,
        output_data: NumberArray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: NumberArray,
    ) -> NumberArray:
        output_data = self.algo.predict(input_data)
        if output_data.ndim == 1:
            return output_data[:, newaxis]

        return output_data
