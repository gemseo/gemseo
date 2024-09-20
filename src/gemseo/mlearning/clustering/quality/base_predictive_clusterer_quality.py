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
"""The base class to assess the quality of a predictive clusterer."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.clustering.quality.base_clusterer_quality import (
    BaseClustererQuality,
)
from gemseo.mlearning.core.quality.base_ml_algo_quality import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType
from gemseo.mlearning.resampling.bootstrap import Bootstrap
from gemseo.mlearning.resampling.cross_validation import CrossValidation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo import Dataset
    from gemseo.mlearning.clustering.algos.base_predictive_clusterer import (
        BasePredictiveClusterer,
    )


class BasePredictiveClustererQuality(BaseClustererQuality):
    """The base class to assess the quality of a predictive clusterer."""

    algo: BasePredictiveClusterer

    def __init__(
        self,
        algo: BasePredictiveClusterer,
        fit_transformers: bool = BaseMLAlgoQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for predictive clustering.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_test_measure(  # noqa: D102
        self,
        test_data: Dataset,
        samples: Sequence[int] = (),
        multioutput: bool = True,
    ) -> MeasureType:
        self._pre_process(samples)
        data = test_data.get_view(variable_names=self.algo.var_names).to_numpy()
        return self._compute_measure(data, self.algo.predict(data), multioutput)

    def compute_cross_validation_measure(  # noqa: D102
        self,
        n_folds: int = 5,
        samples: Sequence[int] = (),
        multioutput: bool = True,
        randomize: bool = BaseClustererQuality._RANDOMIZE,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        samples, seed = self._pre_process(samples, seed, randomize)
        data = self._get_data()
        cross_validation = CrossValidation(samples, n_folds, randomize, seed)
        _, predictions = cross_validation.execute(
            self.algo,
            return_models=store_resampling_result,
            input_data=data,
            fit_transformers=self._fit_transformers,
            store_sampling_result=store_resampling_result,
        )
        return self._compute_measure(data, predictions, multioutput)

    def compute_bootstrap_measure(  # noqa: D102
        self,
        n_replicates: int = 100,
        samples: Sequence[int] = (),
        multioutput: bool = True,
        seed: int | None = None,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        samples, seed = self._pre_process(samples, seed, True)
        data = self._get_data()
        bootstrap = Bootstrap(samples, n_replicates, seed)
        _, predictions = bootstrap.execute(
            self.algo,
            return_models=store_resampling_result,
            input_data=data,
            stack_predictions=False,
            fit_transformers=self._fit_transformers,
            store_sampling_result=store_resampling_result,
        )
        measure = 0
        for prediction, split in zip(predictions, bootstrap.splits):
            measure += self._compute_measure(data[split.test], prediction, multioutput)
        return measure / n_replicates
