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
r"""The silhouette coefficient to assess a clustering.

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

from typing import Sequence

from numpy import ndarray
from sklearn.metrics import silhouette_score

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.qual_measure.cluster_measure import MLPredictiveClusteringMeasure
from gemseo.mlearning.qual_measure.quality_measure import MeasureType


class SilhouetteMeasure(MLPredictiveClusteringMeasure):
    """The silhouette coefficient to assess a clustering."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo: MLPredictiveClusteringAlgo,
        fit_transformers: bool = MLPredictiveClusteringMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A clustering algorithm.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def evaluate_test(
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        raise NotImplementedError

    def evaluate_kfolds(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MLPredictiveClusteringMeasure._RANDOMIZE,
        seed: int | None = None,
    ) -> MeasureType:
        raise NotImplementedError

    def evaluate_bootstrap(
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
            raise NotImplementedError(
                f"The {self.__class__.__name__} does not support the multioutput case."
            )
        return silhouette_score(data, labels)
