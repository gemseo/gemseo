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
r"""The k-means algorithm for clustering.

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
Denoting :math:`(c_1, \cdots, c_k) \in \mathbb{R}^{n \times k}` the centroids,
and assuming no overlap between the centroids,
we may compute the prediction

.. math::

        \operatorname{cluster}(x) =
            \underset{i=1,\cdots,k}{\operatorname{argmin}} \|x-c_i\|.

A probability measure may also be provided,
using the distances from the point to each of the centroids:

.. math::

    \mathbb{P}(x \in C_i) = \begin{cases}
        1 & \operatorname{if} x = c_i\\
        0 & \operatorname{if} x = c_j,\ j \neq i\\
        \frac{\frac{1}{\|x-c_i\|}}{\sum_{j=1}^k \frac{1}{\|x-c_j\|}}
            & \operatorname{if} x \neq c_j\, \forall j=1,\cdots,k
    \end{cases},

where :math:`C_i = \{x\, | \, \operatorname{cluster}(x) = i \}`.
Here,
:math:`\mathbb{P}(x \in C_i)` represents the probability of cluster :math:`i`
given the point :math:`x`.


This concept is implemented through the :class:`.KMeans` class
which inherits from the :class:`.BaseClusterer` class.

Dependence
----------
This clustering algorithm relies on the KMeans class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cluster.KMeans.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import finfo
from numpy import ndarray
from numpy import newaxis
from numpy.linalg import norm
from sklearn.cluster import KMeans as SKLKmeans

from gemseo.mlearning.clustering.algos.base_predictive_clusterer import (
    BasePredictiveClusterer,
)
from gemseo.mlearning.clustering.algos.kmeans_settings import KMeans_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class KMeans(BasePredictiveClusterer):
    """The k-means clustering algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "KMeans"
    LIBRARY: ClassVar[str] = "scikit-learn"

    EPS = finfo(float).eps

    Settings: ClassVar[type[KMeans_Settings]] = KMeans_Settings

    def _post_init(self):
        super()._post_init()
        self.algo = SKLKmeans(
            self._settings.n_clusters,
            random_state=self._settings.random_state,
            n_init=self._settings.parameters.pop("n_init", "auto"),
            **self._settings.parameters,
        )

    def _fit(
        self,
        data: RealArray,
    ) -> None:
        self.labels = self.algo.fit_predict(data)

    def _predict(
        self,
        data: RealArray,
    ) -> ndarray:
        return self.algo.predict(data)

    def _predict_proba_soft(
        self,
        data: RealArray,
    ) -> RealArray:
        inverse_distances = 1 / (
            norm(data[:, newaxis] - self.algo.cluster_centers_, axis=2) + self.EPS
        )
        return inverse_distances / inverse_distances.sum(axis=1)[:, newaxis]
