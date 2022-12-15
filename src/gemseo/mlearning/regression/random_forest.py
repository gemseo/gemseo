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

from typing import ClassVar
from typing import Iterable

from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor as SKLRandForest

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.python_compatibility import Final


class RandomForestRegressor(MLRegressionAlgo):
    """Random forest regression."""

    SHORT_ALGO_NAME: ClassVar[str] = "RF"
    LIBRARY: Final[str] = "scikit-learn"

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        n_estimators: int = 100,
        **parameters,
    ) -> None:
        """
        Args:
            n_estimators: The number of trees in the forest.
        """
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_estimators=n_estimators,
            **parameters,
        )
        self.algo = SKLRandForest(n_estimators=n_estimators, **parameters)

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        # SKLearn RandomForestReressor does not like output
        # shape (n_samples, 1), prefers (n_samples,).
        # The shape (n_samples, n_outputs) with n_outputs >= 2 is fine.
        if output_data.shape[1] == 1:
            output_data = output_data[:, 0]
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        output_data = self.algo.predict(input_data)

        # n_outputs=1 => output_shape=(n_samples,). Convert to (n_samples, 1).
        if len(output_data.shape) == 1:
            output_data = output_data[:, None]

        return output_data
