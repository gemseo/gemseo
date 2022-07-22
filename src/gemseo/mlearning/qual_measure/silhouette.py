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
r"""The silhouette coefficient to measure the quality of a clustering algorithm.

The :mod:`~gemseo.mlearning.qual_measure.silhouette` module implements the
concept of silhouette coefficient measure for machine learning algorithms.

This concept is implemented through the
:class:`.SilhouetteMeasure` class and
overloads the :meth:`!MLClusteringMeasure._compute_measure` method.

The silhouette coefficient is defined for each point as the difference
between the average distance from the point to each of the other points in its cluster
and the average distance from the point to each of the points
in the nearest cluster different from its own.

More formally,
the silhouette coefficient :math:`s_i` of a point :math:`x_i` is given by

.. math::

    a_i = \\frac{1}{|C_{k_i}| - 1} \\sum_{j\\in C_{k_i}\setminus\{i\} } \\|x_i-x_j\\|\\\\
    b_i = \\underset{\\ell=1,\\cdots,K\\atop{\\ell\\neq k_i}}{\\min}\\
        \\frac{1}{|C_\\ell|} \\sum_{j\\in C_\\ell} \\|x_i-x_j\\|\\\\
    s_i = \\frac{b_i-a_i}{\\max(b_i,a_i)}

where
:math:`k_i` is the index of the cluster to which :math:`x_i` belongs,
:math:`K` is the number of clusters,
:math:`C_k` is the set of indices of points
belonging to the cluster :math:`k` (:math:`k=1,\\cdots,K`),
and :math:`|C_k| = \\sum_{j\\in C_k} 1` is the number of points
in the cluster :math:`k`, :math:`k=1,\\cdots,K`.
"""
from __future__ import annotations

from typing import Sequence

from numpy import ndarray
from sklearn.metrics import silhouette_score

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.qual_measure.cluster_measure import MLPredictiveClusteringMeasure


class SilhouetteMeasure(MLPredictiveClusteringMeasure):
    """The silhouette coefficient measure for machine learning."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo: MLPredictiveClusteringAlgo,
        fit_transformers: bool = False,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for clustering.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def evaluate_test(
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> float | ndarray:
        raise NotImplementedError

    def evaluate_kfolds(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = False,
        seed: int | None = None,
    ) -> float | ndarray:
        raise NotImplementedError

    def evaluate_bootstrap(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
    ) -> float | ndarray:
        raise NotImplementedError

    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> float | ndarray:
        if multioutput:
            raise NotImplementedError(
                "The SilhouetteMeasure does not support the multioutput case."
            )
        return silhouette_score(data, labels)
