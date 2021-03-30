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
K-means clustering algorithm
============================

The k-means algorithm groups the data into clusters, where the number of
clusters :math:`k` is fixed. This is done by initializing :math:`k` centroids
in the design space. The points are grouped into clusters according to their
nearest centroid.

When fitting the algorithm, each centroid is successively moved to the mean of
its corresponding cluster, and the cluster value of each point is then reset to
the cluster value of the closest centroid. This process is repeated until
convergence.

Cluster values of new points may be predicted by returning the value of the
closest centroid. Denoting
:math:`(c_1, \\cdots, c_k) \\in \\mathbb{R}^{n \\times k}` the centroids, and
assuming no overlap between the centroids, we may compute the prediction

.. math::

        \\operatorname{cluster}(x) =
            \\underset{i=1,\\cdots,k}{\\operatorname{argmin}} \\|x-c_i\\|.

A probability measure may also be provided, using the distances from the point
to each of the centroids:

.. math::

    \\mathbb{P}(x \\in C_i) = \\begin{cases}
        1 & \\operatorname{if} x = c_i\\\\
        0 & \\operatorname{if} x = c_j,\\ j \\neq i\\\\
        \\frac{\\frac{1}{\\|x-c_i\\|}}{\\sum_{j=1}^k \\frac{1}{\\|x-c_j\\|}}
            & \\operatorname{if} x \\neq c_j\\, \\forall j=1,\\cdots,k
    \\end{cases},

where :math:`C_i = \\{x\\, | \\, \\operatorname{cluster}(x) = i \\}`. Here,
:math:`\\mathbb{P}(x \\in C_i)` represents the probability of cluster :math:`i`
given the point :math:`x`.


This concept is implemented through
the :class:`.KMeans` class which inherits from
the :class:`.MLClusteringAlgo` class.

Dependence
----------
This clustering algorithm relies on the KMeans class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cluster.KMeans.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import finfo
from numpy.linalg import norm
from sklearn.cluster import KMeans as SKLKmeans

from gemseo.mlearning.cluster.cluster import MLClusteringAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class KMeans(MLClusteringAlgo):
    """ KMeans clustering algorithm. """

    ABBR = "KMeans"

    EPS = finfo(float).eps

    def __init__(
        self,
        data,
        transformer=None,
        var_names=None,
        n_clusters=5,
        random_state=0,
        **parameters
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param var_names: names of the variables to consider.
        :type var_names: list(str)
        :param n_clusters: number of clusters. Default: 5.
        :type n_clusters: int
        :param random_state: If None, use a random generation
            of the initial centroids.
            Use an int to make the randomness deterministic. Default: 0.
        :type random_state: int
        :param parameters: Scikit-learn algorithm parameters.
        """
        super(KMeans, self).__init__(
            data,
            transformer=transformer,
            var_names=var_names,
            n_clusters=n_clusters,
            random_state=random_state,
            **parameters
        )
        self.algo = SKLKmeans(n_clusters, random_state=random_state, **parameters)

    def _fit(self, data):
        """Fit the clustering model to the data and store labels.

        :param ndarray data: training data (2D).
        """
        self.labels = self.algo.fit_predict(data)

    def _predict(self, data):
        """Predict cluster of data.

        :param ndarray data: data (2D).
        :return: clusters of data (1D).
        :rtype: ndarray(int).
        """
        return self.algo.predict(data)

    def _predict_proba_soft(self, data):
        """Predict probability of belonging to each cluster.

        :param ndarray data: data (2D).
        :return: probabilities for each cluster for each sample (2D). The sum
            of each row is one.
        :rtype: ndarray
        """
        centers = self.algo.cluster_centers_
        distances = norm(data[:, None] - centers, axis=2)
        inverse_distances = 1 / (distances + self.EPS)
        probas = inverse_distances / inverse_distances.sum(axis=1)[:, None]
        return probas
