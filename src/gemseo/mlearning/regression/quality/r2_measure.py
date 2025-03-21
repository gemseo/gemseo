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
r"""The R2 score to assess the quality of a regressor.

The R2 score s defined by

.. math::

    R_2(\hat{y}) = 1 - \frac{\sum_i (\hat{y}_i - y_i)^2}
                              {\sum_i (y_i-\bar{y})^2},

where
:math:`\hat{y}` are the predictions,
:math:`y` are the data points and
:math:`\bar{y}` is the mean of :math:`y`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from gemseo.mlearning.regression.quality.base_regressor_quality import (
    BaseRegressorQuality,
)
from gemseo.mlearning.resampling.bootstrap import Bootstrap
from gemseo.mlearning.resampling.cross_validation import CrossValidation

if TYPE_CHECKING:
    from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import RealArray


class R2Measure(BaseRegressorQuality):
    """The R2 score to assess the quality of a regressor."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo: BaseRegressor,
        fit_transformers: bool = BaseRegressorQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for regression.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers)

    def _compute_measure(
        self,
        outputs: RealArray,
        predictions: RealArray,
        multioutput: bool = True,
    ) -> MeasureType:
        return r2_score(
            outputs,
            predictions,
            multioutput=self._GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT[multioutput],
        )

    def compute_cross_validation_measure(  # noqa: D102
        self,
        n_folds: int = 5,
        samples: list[int] = (),
        multioutput: bool = True,
        randomize: bool = BaseRegressorQuality._RANDOMIZE,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        return self.__evaluate_by_resampling(
            as_dict,
            multioutput,
            randomize,
            CrossValidation,
            samples,
            seed,
            store_resampling_result,
            n_folds=n_folds,
            randomize=randomize,
        )

    def __evaluate_by_resampling(
        self,
        as_dict: bool,
        multioutput: bool,
        update_seed: bool,
        resampler_class,
        samples: list[int],
        seed: int | None,
        store_resampling_result: bool,
        **kwargs: Any,
    ) -> MeasureType:
        """Evaluate the quality measure with a resampler.

        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.
            update_seed: Whether to update the seed before resampling.
            resampler_class: The class of the resampler.
            samples: The indices of the learning samples.
                If ``None``, use the whole training dataset.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.
            **kwargs: The options to instantiate the resampler.

        Returns:
            The estimation of the quality measure by resampling.
        """
        samples, seed = self._pre_process(samples, seed, update_seed)
        stacked_predictions = resampler_class == CrossValidation
        resampler = resampler_class(samples, seed=seed, **kwargs)
        _, predictions = resampler.execute(
            self.algo,
            return_models=store_resampling_result,
            input_data=self.algo.input_data,
            stack_predictions=stacked_predictions,
            fit_transformers=self._fit_transformers,
            store_sampling_result=store_resampling_result,
        )
        output_data = self.algo.output_data
        var = output_data.var(0)
        if stacked_predictions:
            mse = ((output_data - predictions) ** 2).mean(0)
            if not multioutput:
                mse = mse.mean()
                var = var.mean()
        else:
            mse = 0
            for prediction, split in zip(predictions, resampler.splits):
                mse += mean_squared_error(
                    output_data[split.test],
                    prediction,
                    multioutput=self._GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT[
                        multioutput
                    ],
                )
            mse /= len(resampler.splits)

        return self._post_process_measure(1 - mse / var, multioutput, as_dict)

    def compute_bootstrap_measure(  # noqa: D102
        self,
        n_replicates: int = 100,
        samples: list[int] = (),
        multioutput: bool = True,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        return self.__evaluate_by_resampling(
            as_dict,
            multioutput,
            False,
            Bootstrap,
            samples,
            seed,
            store_resampling_result,
            n_replicates=n_replicates,
        )
