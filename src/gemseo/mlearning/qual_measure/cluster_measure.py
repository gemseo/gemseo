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
Clustering measure
==================

The :mod:`~gemseo.mlearning.qual_measure.cluster_measure` module implements
the concept of clustering measures for machine learning algorithms.

This concept is implemented through the :class:`.MLClusteringMeasure` class
and implements the different evaluation methods.
"""
from __future__ import absolute_import, division, unicode_literals

from numpy import arange, array_split
from numpy import delete as npdelete
from numpy.random import choice

from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLClusteringMeasure(MLQualityMeasure):
    """Clustering measure for machine learning."""

    def evaluate_learn(self, samples=None, multioutput=True):
        """Evaluate quality measure using the learning dataset.

        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        if not self.algo.is_trained:
            self.algo.learn(samples)
        data = self._get_data()[samples]
        labels = self.algo.labels
        measure = self._compute_measure(data, labels, multioutput)
        return measure

    def evaluate_test(self, test_data, samples=None, multioutput=True):
        """Evaluate quality measure using a test dataset.

        Only works if clustering algorithm has a predict method.

        :param Dataset test_data: test data.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        self._check_predict()
        if not self.algo.is_trained:
            self.algo.learn(samples)
        names = self.algo.var_names
        data = test_data.get_data_by_names(names, False)
        predictions = self.algo.predict(data)
        measure = self._compute_measure(data, predictions, multioutput)
        return measure

    def evaluate_kfolds(self, n_folds=5, samples=None, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        Only works if clustering algorithm has a predict method.

        :param int n_folds: number of folds. Default: 5.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        self._check_predict()
        inds = samples
        folds = array_split(inds, n_folds)

        data = self._get_data()

        qualities = []
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            train = npdelete(inds, fold)
            self.algo.learn(samples=train)
            testdata = data[fold]
            predictions = self.algo.predict(testdata)

            quality = self._compute_measure(testdata, predictions, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def evaluate_bootstrap(self, n_replicates=100, samples=None, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        Only works if clustering algorithm has a predict method.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        :rtype: float or ndarray(float)
        """
        samples = self._assure_samples(samples)
        self._check_predict()
        if isinstance(samples, list):
            n_samples = len(samples)
        else:
            n_samples = samples.size
        inds = arange(n_samples)

        data = self._get_data()

        qualities = []
        for _ in range(n_replicates):
            train = choice(n_samples, n_samples)
            test = npdelete(inds, train)
            self.algo.learn(samples=samples[train])
            testdata = data[samples[test]]
            predictions = self.algo.predict(testdata)

            quality = self._compute_measure(testdata, predictions, multioutput)
            qualities.append(quality)

        quality = sum(qualities) / len(qualities)

        return quality

    def _compute_measure(self, data, labels, multioutput=True):
        """Compute error measure.

        :param ndarray data: reference data.
        :param ndarray labels: predicted labels.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: measure value.
        :rtype: float or ndarray(float)
        """
        raise NotImplementedError

    def _get_data(self):
        """Get data."""
        names = self.algo.var_names
        data = self.algo.learning_set.get_data_by_names(names, False)
        return data

    def _check_predict(self):
        """Check if clustering algorithm has predict method."""
        if not hasattr(self.algo, "predict"):
            raise NotImplementedError(
                "Clustering algorithm does not provide " "a predict method."
            )
