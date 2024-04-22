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
r"""The silhouette score to assess the quality of a clusterer.

The `silhouette coefficient <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`__
:math:`s_i` is a measure of
how similar a point :math:`x_i` is to its own cluster :math:`C_{k_i}` (cohesion)
compared to other clusters (separation):

.. math::

   s_i = \frac{b_i-a_i}{\max(a_i,b_i)}

with :math:`a_i=\frac{1}{|C_{k_i}|-1} \sum_{j\in C_{k_i}\setminus\{i\} } \|x_i-x_j\|`
and :math:`b_i = \underset{\ell=1,\cdots,K\atop{\ell\neq k_i}}{\min}
\frac{1}{|C_\ell|} \sum_{j\in C_\ell} \|x_i-x_j\|`

where

- :math:`K` is the number of clusters,
- :math:`C_k` are the indices of the points belonging to the cluster :math:`k`,
- :math:`|C_k|` is the size of :math:`C_k`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import silhouette_score

from gemseo.mlearning.clustering.quality.base_predictive_clusterer_quality import (
    BasePredictiveClustererQuality,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.clustering.algos.base_predictive_clusterer import (
        BasePredictiveClusterer,
    )
    from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType


class SilhouetteMeasure(BasePredictiveClustererQuality):
    """The silhouette score to assess the quality of a clusterer."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo: BasePredictiveClusterer,
        fit_transformers: bool = BasePredictiveClustererQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A clustering algorithm.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_test_measure(  # noqa: D102
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        raise NotImplementedError

    def compute_cross_validation_measure(  # noqa: D102
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = BasePredictiveClustererQuality._RANDOMIZE,
        seed: int | None = None,
    ) -> MeasureType:
        raise NotImplementedError

    def compute_bootstrap_measure(  # noqa: D102
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
    ) -> MeasureType:
        raise NotImplementedError

    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        if multioutput:
            msg = (
                f"The {self.__class__.__name__} does not support the multioutput case."
            )
            raise NotImplementedError(msg)
        return silhouette_score(data, labels)
