# -*- coding: utf-8 -*-
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
"""The random forest for regression.

The random forest regression uses averaging methods on an ensemble of decision trees.

Dependence
----------
The regression model relies on the RandomForestRegressor class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.RandomForestRegressor.html>`_.
"""
from __future__ import division, unicode_literals

import logging
from typing import Iterable, Optional, Union

from numpy import ndarray
from sklearn.ensemble import RandomForestRegressor as SKLRandForest

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo

LOGGER = logging.getLogger(__name__)


class RandomForestRegressor(MLRegressionAlgo):
    """Random forest regression."""

    LIBRARY = "scikit-learn"
    ABBR = "RandomForestRegressor"

    def __init__(
        self,
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        n_estimators=100,  # type: int
        **parameters
    ):  # type: (...) -> None
        """
        Args:
            n_estimators: The number of trees in the forest.
        """
        super(RandomForestRegressor, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_estimators=n_estimators,
            **parameters  # type: Optional[Union[bool,int,float,str]]
        )
        self.algo = SKLRandForest(n_estimators=n_estimators, **parameters)

    def _fit(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> None
        # SKLearn RandomForestReressor does not like output
        # shape (n_samples, 1), prefers (n_samples,).
        # The shape (n_samples, n_outputs) with n_outputs >= 2 is fine.
        if output_data.shape[1] == 1:
            output_data = output_data[:, 0]
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        output_data = self.algo.predict(input_data)

        # n_outputs=1 => output_shape=(n_samples,). Convert to (n_samples, 1).
        if len(output_data.shape) == 1:
            output_data = output_data[:, None]

        return output_data
