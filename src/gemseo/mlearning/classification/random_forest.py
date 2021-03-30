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
#        :author: Francois Gallard, Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Random forest classification model
==================================

The random forest classification model uses averaging methods on an ensemble
of decision trees.

Dependence
----------
The classifier relies on the RandomForestClassifier class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.ensemble.RandomForestClassifier.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import stack
from sklearn.ensemble import RandomForestClassifier as SKLRandForest

from gemseo.mlearning.classification.classification import MLClassificationAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class RandomForestClassifier(MLClassificationAlgo):
    """ Random forest classification algorithm. """

    LIBRARY = "scikit-learn"
    ABBR = "RandomForestClassifier"

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
        :param parameters: other keyword arguments for sklearn rand. forest.
        """
        super(RandomForestClassifier, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_estimators=n_estimators,
            **parameters
        )
        self.algo = SKLRandForest(n_estimators=n_estimators, **parameters)

    def _fit(self, input_data, output_data):
        """Fit the classification model.

        :param ndarray input_data: input data (2D).
        :param ndarray(int) output_data: output data.
        """
        if output_data.shape[1] == 1:
            output_data = output_data.ravel()
        self.algo.fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output data from input data.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: output data (n_samples, n_outputs).
        :rtype: ndarray(int)
        """
        output_data = self.algo.predict(input_data).astype(int)
        if len(output_data.shape) == 1:
            output_data = output_data[:, None]
        return output_data

    def _predict_proba_soft(self, input_data):
        """Predict probability of belonging to each class.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: probabilities of belonging to each class
            (n_samples, n_outputs, n_classes). For a given sample and output
            variable, the sum of probabilities is one.
        :rtype: ndarray
        """
        probas = self.algo.predict_proba(input_data)
        if len(probas[0].shape) == 1:
            probas = probas[..., None]
        else:
            probas = stack(probas, axis=-1)
        return probas
