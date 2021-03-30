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
Gaussian mixture clustering algorithm
=====================================

The Gaussian mixture algorithm groups the data into clusters.
The number of clusters is fixed. Each cluster :math:`i=1, \\cdots, k` is
defined by a mean :math:`\\mu_i` and a covariance matrix :math:`\\Sigma_i`.

The prediction of the cluster value of a point is simply the cluster where the
probability density from the Gaussian distribution defined by the given mean
and covariance matrix is the highest:

.. math::

        \\operatorname{cluster}(x) =
            \\underset{i=1,\\cdots,k}{\\operatorname{argmax}}
            \\mathcal{N}(x; \\mu_i, \\Sigma_i) =
            \\underset{i=1,\\cdots,k}{\\operatorname{argmin}}
            \\|x-\\mu_i\\|_{\\Sigma_i^{-1}},

where :math:`\\mathcal{N}(x; \\mu_i, \\Sigma_i)` is the value of the
probability density function of a Gaussian random variable
:math:`X \\sim \\mathcal{N}(\\mu_i, \\Sigma_i)` at the point :math:`x` and
:math:`\\|x-\\mu_i\\|_{\\Sigma_i^{-1}} =
\\sqrt{(x-\\mu_i)^T \\Sigma_i^{-1} (x-\\mu_i)}`
is the Mahalanobis distance between :math:`x` and :math:`\\mu_i` weighted by
:math:`\\Sigma_i`.
Likewise, the probability of belonging to a cluster :math:`i=1, \\cdots, k` may
be determined through

.. math::

        \\mathbb{P}(x \\in C_i) = \\frac{\\mathcal{N}(x; \\mu_i, \\Sigma_i)}
            {\\sum_{j=1}^k \\mathcal{N}(x; \\mu_j, \\Sigma_j)},

where :math:`C_i = \\{x\\, | \\, \\operatorname{cluster}(x) = i \\}`.

When fitting the algorithm, the cluster centers :math:`\\mu_i` and the
covariance matrices :math:`\\Sigma_i` are computed using the
expectation-maximization algorithm.

This concept is implemented through the :class:`.GaussianMixture`
class which inherits from the :class:`.MLClusteringAlgo` class.

Dependence
----------
This clustering algorithm relies on the GaussianMixture class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.mixture.GaussianMixture.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from sklearn.mixture import GaussianMixture as SKLGaussianMixture

from gemseo.mlearning.cluster.cluster import MLClusteringAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class GaussianMixture(MLClusteringAlgo):
    """ Gaussian mixture clustering algorithm. """

    ABBR = "GaussMix"

    def __init__(
        self, data, transformer=None, var_names=None, n_components=5, **parameters
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param var_names: names of the variables to consider.
        :type var_names: list(str)
        :param n_components: number of Gaussian mixture components.
            Default: 5.
        :type n_components: int
        :param parameters: Scikit-learn algorithm parameters.
        """
        super(GaussianMixture, self).__init__(
            data,
            transformer=transformer,
            var_names=var_names,
            n_components=n_components,
            **parameters
        )
        self.algo = SKLGaussianMixture(n_components, **parameters)

    def _fit(self, data):
        """Fit the clustering model to the data and store labels.

        :param ndarray data: training data (2D).
        """
        self.algo.fit(data)
        self.labels = self.algo.predict(data)

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
        :rtype: ndarray.
        """
        return self.algo.predict_proba(data)
