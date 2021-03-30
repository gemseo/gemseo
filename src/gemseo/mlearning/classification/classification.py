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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Classification model
====================

The :mod:`~gemseo.mlearning.classification.classification` module
implements classification algorithms, whose goal is to find relationships
between input data and output classes.

Classification algorithms provide methods for predicting classes of new input
data, as well as predicting the probabilities of belonging to each of the
classes wherever possible.

This concept is implemented through the :class:`.MLClassificationAlgo` class
which inherits from the :class:`.MLSupervisedAlgo` class.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import unique, zeros

from gemseo.mlearning.core.supervised import MLSupervisedAlgo

standard_library.install_aliases()


class MLClassificationAlgo(MLSupervisedAlgo):
    """Classification Algorithm.

    Inheriting classes should implement the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLClassificationAlgo._predict` methods, and
    :meth:`!MLClassificationAlgo._predict_proba_soft` method if possible.
    """

    def __init__(
        self, data, transformer=None, input_names=None, output_names=None, **parameters
    ):
        """Constructor.

        :param Dataset data: learning dataset.
        :param transformer: transformation strategy for data groups.
            If None, do not scale data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param parameters: algorithm parameters.
        """
        super(MLClassificationAlgo, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            **parameters
        )
        self.n_classes = None

    def learn(self, samples=None):
        """Train machine learning algorithm on learning set, possibly filtered
        using the given parameters. Determine the number of classes.
        :param list(int) samples: indices of training samples.
        """
        output_data = self.learning_set.get_data_by_names(self.output_names, False)
        self.n_classes = unique(output_data).shape[0]
        super(MLClassificationAlgo, self).learn(samples)

    @MLSupervisedAlgo.DataFormatters.format_input_output
    def predict_proba(self, input_data, hard=True):
        """Predict probability of belonging to each class.

        :param input_data: input data (n_inputs,) or (n_samples, n_inputs).
        :type input_data: dict(ndarray) or ndarray
        :param bool hard: indicator for hard or soft classification.
            Default: True.
        :return: probabilities of belonging to each class
            (n_outputs, n_classes) or (n_samples, n_outputs, n_classes).
        :rtype: dict(ndarray) or ndarray
        """
        return self._predict_proba(input_data, hard)

    def _predict_proba(self, input_data, hard=True):
        """Predict probability of belonging to each class.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :param bool hard: indicator for hard or soft classification.
            Default: True.
        :return: probabilities of belonging to each class
            (n_samples, n_outputs, n_classes). For a given sample and output
            variable, the sum of probabilities is one.
        :rtype: ndarray
        """
        if hard:
            probas = self._predict_proba_hard(input_data)
        else:
            probas = self._predict_proba_soft(input_data)
        return probas

    def _predict_proba_hard(self, input_data):
        """Create class indicator of input data.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: probabilities of belonging to each class
            (n_samples, n_outputs, n_classes). For a given sample and output
            variable, the sum of probabilities is one.
        :rtype: ndarray
        """
        n_samples = input_data.shape[0]
        prediction = self._predict(input_data).astype(int)
        n_outputs = prediction.shape[1]
        probas = zeros((n_samples, self.n_classes, n_outputs))
        for n_sample in range(prediction.shape[0]):
            for n_output in range(n_outputs):
                probas[n_sample, prediction[n_sample, n_output], n_output] = 1
        return probas

    def _predict_proba_soft(self, input_data):
        """Predict probability of belonging to each class.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: probabilities of belonging to each class
            (n_samples, outputs, n_classes). For a given sample and output
            variable, the sum of probabilities is one.
        :rtype: ndarray
        """
        raise NotImplementedError

    def _get_objects_to_save(self):
        """Get objects to save.
        :return: objects to save.
        :rtype: dict
        """
        objects = super(MLClassificationAlgo, self)._get_objects_to_save()
        objects["n_classes"] = self.n_classes
        return objects
