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
Silhouette coefficient clustering measure
=========================================

The :mod:`~gemseo.mlearning.qual_measure.silhouette` module implements the
concept of silhouette coefficient measure for machine learning algorithms.

This concept is implemented through the
:class:`.SilhouetteMeasure` class and
overloads the :meth:`!MLClusteringMeasure._compute_measure` method.

The silhouette coefficient is defined for each point as the difference between
the average distance from the point to each of the other points in its cluster
and the average distance from the point to each of the points in the the
nearest cluster different from its own.

More formally, the silhouette coefficient :math:`s_i` of a point :math:`x_i`
is given by

.. math::

    a_i = \\frac{1}{|C_{k_i}|} \\sum_{j\\in\\C_{k_i}} \\|x_i-x_j\\|\\\\
    b_i = \\underset{\\ell=1,\\cdots,K\\atop{\\ell\\neq k_i}}{\\min}\\
        \\frac{1}{|C_\\ell|} \\sum_{j\\in\\C_\\ell} \\|x_i-x_j\\|\\\\
    s_i = \\frac{b_i-a_i}{\\max(b_i,a_i)}

where
:math:`K` is the number of clusters,
:math:`C_k` is the set of indices of points belonging to
cluster :math:`k\\ k=1,\\cdots,K` and
:math:`|C_k| = \\sum_{j\\in C_i} 1` is the number of points in
cluster :math:`k\\ k=1,\\cdots,K`.
"""
from __future__ import absolute_import, division, unicode_literals

from sklearn.metrics import silhouette_samples, silhouette_score

from gemseo.mlearning.qual_measure.cluster_measure import MLClusteringMeasure


class SilhouetteMeasure(MLClusteringMeasure):
    """Silhouette coefficient measure for machine learning."""

    SMALLER_IS_BETTER = False

    def evaluate_test(self, test_data, samples=None, multioutput=True):
        """Evaluate quality measure using a test dataset.

        Only works if clustering algorithm has a predict method.

        :param Dataset test_data: test data.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def evaluate_kfolds(self, n_folds=5, samples=None, multioutput=True):
        """Evaluate quality measure using the k-folds technique.

        Only works if clustering algorithm has a predict method.

        :param int n_folds: number of folds. Default: 5.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def evaluate_bootstrap(self, n_replicates=100, samples=None, multioutput=True):
        """Evaluate quality measure using the bootstrap technique.

        Only works if clustering algorithm has a predict method.

        :param int n_replicates: number of bootstrap replicates. Default: 100.
        :param list(int) samples: samples to consider for training.
            If None, use all samples. Default: None.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: quality measure value.
        """
        raise NotImplementedError

    def _compute_measure(self, data, labels, multioutput=True):
        """Compute Silhouette coefficient(s).

        :param ndarray data: reference data.
        :param ndarray labels: predicted labels.
        :param bool multioutput: if True, return the quality measure for each
            output component. Otherwise, average these measures. Default: True.
        :return: measure value.
        """
        if multioutput:
            measure = silhouette_samples(data, labels)
        else:
            measure = silhouette_score(data, labels)

        return measure
