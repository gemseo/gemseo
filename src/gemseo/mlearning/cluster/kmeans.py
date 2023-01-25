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
"""The k-means algorithm for clustering.

The k-means algorithm groups the data into clusters,
where the number of clusters :math:`k` is fixed.
This is done by initializing :math:`k` centroids in the design space.
The points are grouped into clusters according to their nearest centroid.

When fitting the algorithm,
each centroid is successively moved to the mean of its corresponding cluster,
and the cluster value of each point is then reset
to the cluster value of the closest centroid.
This process is repeated until convergence.

Cluster values of new points may be predicted
by returning the value of the closest centroid.
Denoting :math:`(c_1, \\cdots, c_k) \\in \\mathbb{R}^{n \\times k}` the centroids,
and assuming no overlap between the centroids,
we may compute the prediction

.. math::

        \\operatorname{cluster}(x) =
            \\underset{i=1,\\cdots,k}{\\operatorname{argmin}} \\|x-c_i\\|.

A probability measure may also be provided,
using the distances from the point to each of the centroids:

.. math::

    \\mathbb{P}(x \\in C_i) = \\begin{cases}
        1 & \\operatorname{if} x = c_i\\\\
        0 & \\operatorname{if} x = c_j,\\ j \\neq i\\\\
        \\frac{\\frac{1}{\\|x-c_i\\|}}{\\sum_{j=1}^k \\frac{1}{\\|x-c_j\\|}}
            & \\operatorname{if} x \\neq c_j\\, \\forall j=1,\\cdots,k
    \\end{cases},

where :math:`C_i = \\{x\\, | \\, \\operatorname{cluster}(x) = i \\}`.
Here,
:math:`\\mathbb{P}(x \\in C_i)` represents the probability of cluster :math:`i`
given the point :math:`x`.


This concept is implemented through the :class:`.KMeans` class
which inherits from the :class:`.MLClusteringAlgo` class.

Dependence
----------
This clustering algorithm relies on the KMeans class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cluster.KMeans.html>`_.
"""
from __future__ import annotations

from typing import ClassVar
from typing import Iterable

from numpy import finfo
from numpy import ndarray
from numpy import newaxis
from numpy.linalg import norm
from sklearn.cluster import KMeans as SKLKmeans

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.utils.python_compatibility import Final


class KMeans(MLPredictiveClusteringAlgo):
    """The k-means clustering algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "KMeans"
    LIBRARY: Final[str] = "scikit-learn"

    EPS = finfo(float).eps

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLPredictiveClusteringAlgo.IDENTITY,
        var_names: Iterable[str] | None = None,
        n_clusters: int = 5,
        random_state: int | None = 0,
        **parameters: int | float | bool | str | None,
    ) -> None:
        """
        Args:
            n_clusters: The number of clusters of the K-means algorithm.
            random_state: If ``None``, use a random generation of the initial centroids.
                Otherwise,
                the integer is used to make the initialization deterministic.
        """
        super().__init__(
            data,
            transformer=transformer,
            var_names=var_names,
            n_clusters=n_clusters,
            random_state=random_state,
            **parameters,
        )
        self.algo = SKLKmeans(n_clusters, random_state=random_state, **parameters)

    def _fit(
        self,
        data: ndarray,
    ) -> None:
        self.labels = self.algo.fit_predict(data)

    def _predict(
        self,
        data: ndarray,
    ) -> ndarray:
        return self.algo.predict(data)

    def _predict_proba_soft(
        self,
        data: ndarray,
    ) -> ndarray:
        centers = self.algo.cluster_centers_
        distances = norm(data[:, newaxis] - centers, axis=2)
        inverse_distances = 1 / (distances + self.EPS)
        probas = inverse_distances / inverse_distances.sum(axis=1)[:, newaxis]
        return probas
