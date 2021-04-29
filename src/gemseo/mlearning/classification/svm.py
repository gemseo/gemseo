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
SVM Classifier
==============

This module implements the SVMClassifier class. A support vector machine (SVM) passes
the data through a kernel in order to increase its dimension and thereby make the
classes linearly separable.

Dependence
----------
The classifier relies on the SVC class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.svm.SVC.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

import logging

from numpy import stack
from sklearn.svm import SVC

from gemseo.mlearning.classification.classification import MLClassificationAlgo

LOGGER = logging.getLogger(__name__)


class SVMClassifier(MLClassificationAlgo):
    """K nearest neighbors classification algorithm."""

    LIBRARY = "scikit-learn"
    ABBR = "SVM"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        c=1.0,
        kernel="rbf",
        probability=False,
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
        :param C: inverse L2 regularization parameter. Higher values give less
            regularization. Default: 1.0.
        :type C: float
        :param kernel: kernel for SVM. Examples: "linear", "poly", "rbf", "sigmoid",
            "precomputed" or a callable. Default: "rbf".
        :type kernel: str or callable
        :param probability: toggles the availability of the predict_proba(x, hard=False)
            method. The algorithm is faster if set to False. Default: False.
        :type probability: bool
        :param parameters: other keyword arguments for sklearn SVC.
        """
        super(SVMClassifier, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            c=c,
            kernel=kernel,
            probability=probability,
            **parameters
        )
        self.algo = SVC(C=c, kernel=kernel, probability=probability, **parameters)

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
        if not self.parameters["probability"]:
            raise NotImplementedError(
                "SVMClassifier soft probability prediction is only available if the "
                "parameter 'probability' is set to True."
            )
        probas = self.algo.predict_proba(input_data)
        if len(probas[0].shape) == 1:
            probas = probas[..., None]
        else:
            probas = stack(probas, axis=-1)
        return probas
