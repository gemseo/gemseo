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

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.mlearning.quality_measures.quality_measure import MeasureType
from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasure
from gemseo.mlearning.resampling.bootstrap import Bootstrap
from gemseo.mlearning.resampling.cross_validation import CrossValidation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy import ndarray

    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.clustering.clustering import MLClusteringAlgo
    from gemseo.mlearning.clustering.clustering import MLPredictiveClusteringAlgo


class MLClusteringMeasure(MLQualityMeasure):
    """An abstract clustering measure for clustering algorithms."""

    algo: MLClusteringAlgo

    def __init__(
        self,
        algo: MLClusteringAlgo,
        fit_transformers: bool = MLQualityMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for clustering.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_learning_measure(  # noqa: D102
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        return self._compute_measure(
            self._get_data()[self._pre_process(samples)[0]],
            self.algo.labels,
            multioutput,
        )

    @abstractmethod
    def _compute_measure(
        self,
        data: ndarray,
        labels: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        """Compute the quality measure.

        Args:
            data: The reference data.
            labels: The predicted labels.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.

        Returns:
            The value of the quality measure.
        """

    def _get_data(self) -> ndarray:
        """Get data.

        Returns:
            The learning data.
        """
        return self.algo.learning_set.get_view(
            variable_names=self.algo.var_names
        ).to_numpy()

    # TODO: API: remove this alias in the next major release.
    evaluate_learn = compute_learning_measure


class MLPredictiveClusteringMeasure(MLClusteringMeasure):
    """An abstract clustering measure for predictive clustering algorithms."""

    algo: MLPredictiveClusteringAlgo

    def __init__(
        self,
        algo: MLPredictiveClusteringAlgo,
        fit_transformers: bool = MLQualityMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for predictive clustering.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_test_measure(  # noqa: D102
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
    ) -> MeasureType:
        self._pre_process(samples)
        data = test_data.get_view(variable_names=self.algo.var_names).to_numpy()
        return self._compute_measure(data, self.algo.predict(data), multioutput)

    def compute_cross_validation_measure(  # noqa: D102
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MLClusteringMeasure._RANDOMIZE,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        samples, seed = self._pre_process(samples, seed, randomize)
        cross_validation = CrossValidation(samples, n_folds, randomize, seed)
        data = self._get_data()
        _, predictions = cross_validation.execute(
            self.algo,
            store_resampling_result,
            True,
            True,
            self._fit_transformers,
            store_resampling_result,
            data,
            (len(data),),
        )
        return self._compute_measure(data, predictions, multioutput)

    def compute_bootstrap_measure(  # noqa: D102
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        samples, seed = self._pre_process(samples, seed, True)
        bootstrap = Bootstrap(samples, n_replicates, seed)
        data = self._get_data()
        _, predictions = bootstrap.execute(
            self.algo,
            store_resampling_result,
            True,
            False,
            self._fit_transformers,
            store_resampling_result,
            data,
            (len(data),),
        )
        measure = 0
        for prediction, split in zip(predictions, bootstrap.splits):
            measure += self._compute_measure(data[split.test], prediction, multioutput)
        return measure / n_replicates

    # TODO: API: remove these aliases in the next major release.
    evaluate_test = compute_test_measure
    evaluate_kfolds = compute_cross_validation_measure
    evaluate_bootstrap = compute_bootstrap_measure
