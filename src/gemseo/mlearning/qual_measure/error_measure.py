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

from numpy import arange, array_split
from numpy import delete as npdelete
from numpy import unique
from numpy.random import choice

from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLErrorMeasure(MLQualityMeasure):
    """Error measure for machine learning."""

    def evaluate_learn(self, samples=None, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        self.algo.learn(samples)
        inputs = self.algo.input_data[samples]
        outputs = self.algo.output_data[samples]
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_test(self, test_data, samples=None, multioutput=True):
        """Evaluate quality measure using a test dataset.

        :param Dataset test_data: test data.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        self.algo.learn(samples)
        in_grp = test_data.INPUT_GROUP
        out_grp = test_data.OUTPUT_GROUP
        inputs = test_data.get_data_by_group(in_grp)
        outputs = test_data.get_data_by_group(out_grp)
        predictions = self.algo.predict(inputs)
        measure = self._compute_measure(outputs, predictions, multioutput)
        return measure

    def evaluate_kfolds(self, n_folds=5, samples=None, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        :param int n_folds: number of folds. Default: 5.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        inds = samples
        folds = array_split(inds, n_folds)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        qualities = []
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            train = npdelete(inds, fold)
            self.algo.learn(samples=train)
            expected = outputs[fold]
            predicted = self.algo.predict(inputs[fold])
            quality = self._compute_measure(expected, predicted, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def evaluate_bootstrap(self, n_replicates=100, samples=None, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        if isinstance(samples, list):
            n_samples = len(samples)
        else:
            n_samples = samples.size
        inds = arange(n_samples)

        in_grp = self.algo.learning_set.INPUT_GROUP
        out_grp = self.algo.learning_set.OUTPUT_GROUP
        inputs = self.algo.learning_set.get_data_by_group(in_grp)
        outputs = self.algo.learning_set.get_data_by_group(out_grp)

        qualities = []
        for _ in range(n_replicates):
            train = unique(choice(n_samples, n_samples))
            test = npdelete(inds, train)
            self.algo.learn(samples[train])
            expected = outputs[samples[test]]
            predicted = self.algo.predict(inputs[samples[test]])
            quality = self._compute_measure(expected, predicted, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def _compute_measure(self, outputs, predictions, multioutput=True):
        """Compute error measure.

        :param ndarray outputs: reference outputs.
        :param ndarray predictions: predicted outputs.
        :param bool multioutput: if True, return the error measure for each
            output component. Otherwise, average these errors. Default: True.
        :return: measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError
