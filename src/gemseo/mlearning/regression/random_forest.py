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
"""
Random forest regression
========================

The random forest regression uses averaging methods on an ensemble
of decision trees.

Dependence
----------
The regression model relies on the RandomForestRegressor class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.RandomForestRegressor.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from sklearn.ensemble import RandomForestRegressor as SKLRandForest

from gemseo.mlearning.regression.regression import MLRegressionAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class RandomForestRegressor(MLRegressionAlgo):
    """ Random forest regression """

    LIBRARY = "scikit-learn"
    ABBR = "RandomForestRegressor"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        n_estimators=100,
        **parameters
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param n_estimators: number of trees in the forest.
        :type n_estimators: int
        :param parameters: other keyword arguments for the sklearn algo.
        """
        super(RandomForestRegressor, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_estimators=n_estimators,
            **parameters
        )
        self.algo = SKLRandForest(n_estimators=n_estimators, **parameters)

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D)
        :param ndarray output_data: output data (2D)
        """
        # SKLearn RandomForestReressor does not like output
        # shape (n_samples, 1), prefers (n_samples,).
        # The shape (n_samples, n_outputs) with n_outputs >= 2 is fine.
        if output_data.shape[1] == 1:
            output_data = output_data[:, 0]
        self.algo.fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output for given input data.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray
        """
        output_data = self.algo.predict(input_data)

        # n_outputs=1 => output_shape=(n_samples,). Convert to (n_samples, 1).
        if len(output_data.shape) == 1:
            output_data = output_data[:, None]

        return output_data
