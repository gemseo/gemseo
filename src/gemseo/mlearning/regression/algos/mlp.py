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
from typing import Any
from typing import ClassVar

import sklearn.neural_network
from numpy import newaxis

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.typing import NumberArray


class MLPRegressor(BaseRegressor):
    """MultiLayer perceptron (MLP)."""

    LIBRARY: ClassVar[str] = "scikit-learn"
    SHORT_ALGO_NAME: ClassVar[str] = "MLP"

    def __init__(
        self,
        data: IODataset,
        transformer: TransformerType = BaseRegressor.IDENTITY,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
        hidden_layer_sizes: tuple[int] = (100,),
        **parameters: Any,
    ) -> None:
        """
        Args:
            hidden_layer_sizes: The number of neurons per hidden layer.
        """  # noqa: D205 D212 D415
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            hidden_layer_sizes=hidden_layer_sizes,
            **parameters,
        )
        self.algo = sklearn.neural_network.MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, **parameters
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
