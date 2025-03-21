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
r"""The Gaussian mixture algorithm for clustering.

The Gaussian mixture algorithm groups the data into clusters.
The number of clusters is fixed.
Each cluster :math:`i=1, \cdots, k` is defined
by a mean :math:`\mu_i` and a covariance matrix :math:`\Sigma_i`.

The prediction of the cluster value of a point is simply the cluster
where the probability density of the Gaussian distribution
defined by the given mean and covariance matrix
is the highest:

.. math::

        \operatorname{cluster}(x) =
            \underset{i=1,\cdots,k}{\operatorname{argmax}}
            \mathcal{N}(x; \mu_i, \Sigma_i)

where :math:`\mathcal{N}(x; \mu_i, \Sigma_i)` is the value
of the probability density function
of a Gaussian random variable :math:`X \sim \mathcal{N}(\mu_i, \Sigma_i)`
at the point :math:`x`
and :math:`\|x-\mu_i\|_{\Sigma_i^{-1}} =
\sqrt{(x-\mu_i)^T \Sigma_i^{-1} (x-\mu_i)}`
is the Mahalanobis distance between :math:`x`
and :math:`\mu_i` weighted by :math:`\Sigma_i`.
Likewise,
the probability of belonging to a cluster :math:`i=1, \cdots, k`
may be determined through

.. math::

        \mathbb{P}(x \in C_i) = \frac{\mathcal{N}(x; \mu_i, \Sigma_i)}
            {\sum_{j=1}^k \mathcal{N}(x; \mu_j, \Sigma_j)},

where :math:`C_i = \{x\, | \, \operatorname{cluster}(x) = i \}`.

When fitting the algorithm,
the cluster centers :math:`\mu_i` and the covariance matrices :math:`\Sigma_i`
are computed using the expectation-maximization algorithm.

This concept is implemented through the :class:`.GaussianMixture` class
which inherits from the :class:`.BaseClusterer` class.

Dependence
----------
This clustering algorithm relies on the GaussianMixture class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.mixture.GaussianMixture.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from sklearn.mixture import GaussianMixture as SKLGaussianMixture

from gemseo.mlearning.clustering.algos.base_predictive_clusterer import (
    BasePredictiveClusterer,
)
from gemseo.mlearning.clustering.algos.gaussian_mixture_settings import (
    GaussianMixture_Settings,
)

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.typing import RealArray


class GaussianMixture(BasePredictiveClusterer):
    """The Gaussian mixture clustering algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "GMM"
    LIBRARY: ClassVar[str] = "scikit-learn"

    Settings: ClassVar[type[GaussianMixture_Settings]] = GaussianMixture_Settings

    def _post_init(self):
        super()._post_init()
        self.algo = SKLGaussianMixture(
            n_components=self._settings.n_clusters,
            random_state=self._settings.random_state,
            **self._settings.parameters,
        )

    def _fit(
        self,
        data: RealArray,
    ) -> None:
        self.algo.fit(data)
        self.labels = self.algo.predict(data)

    def _predict(
        self,
        data: RealArray,
    ) -> ndarray:
        return self.algo.predict(data)

    def _predict_proba_soft(
        self,
        data: RealArray,
    ) -> RealArray:
        return self.algo.predict_proba(data)
