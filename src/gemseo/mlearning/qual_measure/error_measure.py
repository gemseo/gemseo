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
Error measure
=============

The :mod:`~gemseo.mlearning.qual_measure.error_measure` module implements
the concept of error measures for machine learning algorithms.

This concept is implemented through the :class:`.MLErrorMeasure` class
and implements the different evaluation methods.

The error measure class is adapted for supervised machine learning algorithms,
as it measures the error of a predicted value to some reference value.
"""
from __future__ import absolute_import, division, unicode_literals

import numpy.random as npr
from future import standard_library
from numpy import arange, array_split
from numpy import delete as npdelete
from numpy import vstack

from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure

standard_library.install_aliases()


class MLErrorMeasure(MLQualityMeasure):
    """ Error measure for machine learning. """

    def evaluate_learn(self, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        """
        if not self.algo.is_trained:
            self.algo.learn()
        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_test(self, test_data, multioutput=True):
        """Evaluate quality measure using a test dataset.

        :param Dataset test_data: test data.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        """
        if not self.algo.is_trained:
            self.algo.learn()
        in_grp = test_data.INPUT_GROUP
        out_grp = test_data.OUTPUT_GROUP
        inputs = test_data.get_data_by_group(in_grp)
        outputs = test_data.get_data_by_group(out_grp)
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_kfolds(self, n_folds=5, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        :param int n_folds: number of folds. Default: 5.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        """
        n_samples = self.algo.learning_set.n_samples
        inds = arange(n_samples)
        folds = array_split(inds, n_folds)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        expected = []
        predicted = []
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            train = npdelete(inds, fold)
            self.algo.learn(samples=train)

            predicted.append(self.algo.predict(inputs[fold]))
            expected.append(outputs[fold])

        expected = vstack(expected)
        predicted = vstack(predicted)

        return self._compute_measure(expected, predicted, multioutput)

    def evaluate_bootstrap(self, n_replicates=100, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        """
        n_samples = self.algo.learning_set.n_samples
        inds = arange(n_samples)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        expected = []
        predicted = []
        for _ in range(n_replicates):
            train = npr.choice(n_samples, n_samples)
            test = npdelete(inds, train)
            self.algo.learn(samples=train)
            predicted.append(self.algo.predict(inputs[test]))
            expected.append(outputs[test])

        expected = vstack(expected)
        predicted = vstack(predicted)

        return self._compute_measure(expected, predicted, multioutput)

    def _compute_measure(self, outputs, predictions, multioutput=True):
        """Compute error measure.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: measure value.
        """
        raise NotImplementedError
