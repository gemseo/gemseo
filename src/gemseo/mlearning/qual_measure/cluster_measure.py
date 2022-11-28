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
"""Here is the baseclass to measure the quality of machine learning algorithms.

The concept of clustering quality measure is implemented with the
:class:`.MLClusteringMeasure` class and proposes different evaluation methods.
"""
from __future__ import annotations

from copy import deepcopy
from typing import NoReturn
from typing import Sequence

from numpy import arange
from numpy import delete as npdelete
from numpy import ndarray
from numpy import unique

from gemseo.core.dataset import Dataset
from gemseo.mlearning.cluster.cluster import MLClusteringAlgo
from gemseo.mlearning.cluster.cluster import MLPredictiveClusteringAlgo
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLClusteringMeasure(MLQualityMeasure):
    """An abstract clustering measure for clustering algorithms."""

    def __init__(
        self,
        algo: MLClusteringAlgo,
        fit_transformers: bool = MLQualityMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for clustering.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def evaluate_learn(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> float | ndarray:
        self._train_algo(samples)
        samples = self._assure_samples(samples)
        return self._compute_measure(
            self._get_data()[samples], self.algo.labels, multioutput
        )

    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> float | ndarray:
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component; if not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def _get_data(self) -> dict[str, ndarray]:
        """Get data.

        Returns:
            The learning data indexed by the names of the variables.
        """
        return self.algo.learning_set.get_data_by_names(self.algo.var_names, False)


class MLPredictiveClusteringMeasure(MLClusteringMeasure):
    """An abstract clustering measure for predictive clustering algorithms."""

    def __init__(
        self,
        algo: MLPredictiveClusteringAlgo,
        fit_transformers: bool = MLQualityMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for predictive clustering.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def evaluate_test(
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> float | ndarray:
        self._train_algo(samples)
        data = test_data.get_data_by_names(self.algo.var_names, False)
        return self._compute_measure(data, self.algo.predict(data), multioutput)

    def evaluate_kfolds(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MLClusteringMeasure._RANDOMIZE,
        seed: int | None = None,
    ) -> float | ndarray:
        self._train_algo(samples)
        data = self._get_data()
        algo = deepcopy(self.algo)
        qualities = []
        folds, samples = self._compute_folds(samples, n_folds, randomize, seed)
        for fold in folds:
            algo.learn(
                samples=npdelete(samples, fold),
                fit_transformers=self._fit_transformers,
            )
            test_data = data[fold]
            qualities.append(
                self._compute_measure(test_data, algo.predict(test_data), multioutput)
            )

        return sum(qualities) / len(qualities)

    def evaluate_bootstrap(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
    ) -> float | ndarray:
        self._train_algo(samples)
        samples = self._assure_samples(samples)
        n_samples = samples.size
        indices = arange(n_samples)

        data = self._get_data()

        algo = deepcopy(self.algo)

        qualities = []
        generator = self._get_rng(seed)
        for _ in range(n_replicates):
            train_indices = unique(generator.choice(n_samples, n_samples))
            test_indices = npdelete(indices, train_indices)
            algo.learn(
                samples=[samples[index] for index in train_indices],
                fit_transformers=self._fit_transformers,
            )
            test_data = data[[samples[index] for index in test_indices]]
            predictions = algo.predict(test_data)

            quality = self._compute_measure(test_data, predictions, multioutput)
            qualities.append(quality)

        return sum(qualities) / len(qualities)

    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> NoReturn:
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component. If not, average these measures.

        Returns:
            The value of the quality measure.
        """
        raise NotImplementedError

    def _get_data(self) -> dict[str, ndarray]:
        """Get data.

        Returns:
            The learning data indexed by the names of the variables.
        """
        return self.algo.learning_set.get_data_by_names(self.algo.var_names, False)
