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
Clustering algorithm
====================

The :mod:`~gemseo.mlearning.cluster.cluster` module
implements the concept of clustering models,
a kind of unsupervised machine learning algorithm where the goal is
to group data into clusters.
Wherever it is possible, these methods should be able to predict the class of
new data, as well as the probability of belonging to each class.

This concept is implemented through the :class:`.MLClusteringAlgo` class
which inherits from the :class:`.MLUnsupervisedAlgo` class.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import atleast_2d, unique, zeros

from gemseo.mlearning.core.unsupervised import MLUnsupervisedAlgo
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


class MLClusteringAlgo(MLUnsupervisedAlgo):
    """Clustering algorithm.

    Inheriting class should overload the
    :meth:`!MLUnsupervisedAlgo._fit` method, and the
    :meth:`!MLClusteringAlgo._predict` and
    :meth:`!MLClusteringAlgo._predict_proba` methods if possible.
    """

    def __init__(self, data, transformer=None, var_names=None, **parameters):
        """Constructor.

        :param Dataset data: learning dataset.
        :param transformer: transformation strategy for data groups.
            If None, do not scale data. Default: None.
        :type transformer: dict(str)
        :param var_names: names of the variables to consider.
        :type var_names: list(str)
        :param parameters: algorithm parameters.
        """
        super(MLClusteringAlgo, self).__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.labels = None
        self.n_clusters = None

    def learn(self, samples=None):
        """Overriding learn function for assuring that labels are defined.
        Identify number of clusters.
        """
        super(MLClusteringAlgo, self).learn(samples=samples)
        if self.labels is None:
            raise ValueError("self._fit() should assign labels.")
        self.n_clusters = unique(self.labels).shape[0]

    def predict(self, data):
        """Predict cluster of data.

        :param data: data (1D or 2D).
        :type data: dict(ndarray) or ndarray
        :return: clusters of data ("0D" or 1D).
        :rtype: int or ndarray(int)
        """
        as_dict = isinstance(data, dict)
        if as_dict:
            data = DataConversion.dict_to_array(data, self.var_names)
        single_sample = len(data.shape) == 1
        data = atleast_2d(data)
        parameters = self.learning_set.DEFAULT_GROUP
        if parameters in self.transformer:
            data = self.transformer[parameters].transform(data)
        clusters = self._predict(atleast_2d(data)).astype(int)
        if single_sample:
            clusters = clusters[0]
        return clusters

    def _predict(self, data):
        """Predict cluster of data.

        :param ndarray data: data (2D).
        :return: clusters of data (1D).
        :rtype: ndarray(int)
        """
        raise NotImplementedError

    def predict_proba(self, data, hard=True):
        """Predict probability of belonging to each cluster.

        :param data: data (1D or 2D).
        :type data: dict(ndarray) or ndarray
        :param bool hard: indicator for hard or soft clustering. Default: True.
        :return: probabilities of belonging to each cluster (1D or 2D, same
            as data).
        :rtype: ndarray
        """
        as_dict = isinstance(data, dict)
        if as_dict:
            data = DataConversion.dict_to_array(data, self.var_names)
        single_sample = len(data.shape) == 1
        data = atleast_2d(data)
        probas = self._predict_proba(atleast_2d(data), hard)
        if single_sample:
            probas = probas.ravel()
        return probas

    def _predict_proba(self, data, hard=True):
        """Predict probability of belonging to each cluster.

        :param ndarray data: data (2D).
        :param bool hard: indicator for hard or soft clustering. Default: True.
        :return: probabilities of belonging to each cluster (2D). The sum of
            each row is one.
        :rtype: ndarray
        """
        if hard:
            probas = self._predict_proba_hard(data)
        else:
            probas = self._predict_proba_soft(data)
        return probas

    def _predict_proba_hard(self, data):
        """Create cluster indicator of input data.

        :param ndarray input_data: input data (2D).
        :return: cluster indicators for each sample (2D). The sum of
            each row is one.
        :rtype: ndarray
        """
        prediction = self._predict(data)
        probas = zeros((data.shape[0], self.n_clusters))
        for i, pred in enumerate(prediction):
            probas[i, pred] = 1
        return probas

    def _predict_proba_soft(self, data):
        """Predict probability of belonging to each cluster.

        :param ndarray data: data (2D).
        :return: probabilities for each cluster for each sample (2D). The sum
            of each row is one.
        :rtype: ndarray
        """
        raise NotImplementedError

    def _get_objects_to_save(self):
        """Get objects to save.
        :return: objects to save.
        :rtype: dict
        """
        objects = super(MLClusteringAlgo, self)._get_objects_to_save()
        objects["labels"] = self.labels
        objects["n_clusters"] = self.n_clusters
        return objects
