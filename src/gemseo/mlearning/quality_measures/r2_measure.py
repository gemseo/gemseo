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
r"""The R2 to measure the quality of a regression algorithm.

The :mod:`~gemseo.mlearning.quality_measures.r2_measure` module
implements the concept of R2 measures for machine learning algorithms.

This concept is implemented through the :class:`.R2Measure` class
and overloads the :meth:`!MLErrorMeasure._compute_measure` method.

The R2 is defined by

.. math::

    R_2(\\hat{y}) = 1 - \\frac{\\sum_i (\\hat{y}_i - y_i)^2}
                              {\\sum_i (y_i-\\bar{y})^2},

where
:math:`\\hat{y}` are the predictions,
:math:`y` are the data points and
:math:`\\bar{y}` is the mean of :math:`y`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NoReturn

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from gemseo.mlearning.quality_measures.error_measure import MLErrorMeasure
from gemseo.mlearning.resampling.bootstrap import Bootstrap
from gemseo.mlearning.resampling.cross_validation import CrossValidation

if TYPE_CHECKING:
    from numpy import ndarray

    from gemseo.mlearning.quality_measures.quality_measure import MeasureType
    from gemseo.mlearning.regression.regression import MLRegressionAlgo


class R2Measure(MLErrorMeasure):
    """The R2 measure for machine learning."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo: MLRegressionAlgo,
        fit_transformers: bool = MLErrorMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for regression.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers)

    def _compute_measure(
        self,
        outputs: ndarray,
        predictions: ndarray,
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
        samples: list[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MLErrorMeasure._RANDOMIZE,
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
        as_dict,
        multioutput,
        update_seed,
        resampler_class,
        samples,
        seed,
        store_resampling_result,
        **kwargs,
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
                If ``None``, use the whole learning dataset.
            seed: The seed of the pseudo-random number generator.
                If ``None``,
                then an unpredictable generator will be used.
            **kwargs: The options to instantiate the resampler.

        Returns:
            The estimation of the quality measure by resampling.
        """
        samples, seed = self._pre_process(samples, seed, update_seed)
        resampler = resampler_class(samples, seed=seed, **kwargs)
        stacked_predictions = resampler_class == CrossValidation
        output_data = self.algo.output_data
        _, predictions = resampler.execute(
            self.algo,
            store_resampling_result,
            True,
            stacked_predictions,
            self._fit_transformers,
            store_resampling_result,
            self.algo.input_data,
            output_data.shape,
        )
        var = self.algo.output_data.var(0)
        if stacked_predictions:
            mse = ((self.algo.output_data - predictions) ** 2).mean(0)
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
        samples: list[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> NoReturn:
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

    # TODO: API: remove these aliases in the next major release.
    evaluate_kfolds = compute_cross_validation_measure
    evaluate_bootstrap = compute_bootstrap_measure
