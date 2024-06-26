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
"""A cross-validation tool for resampling and surrogate modeling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array_split
from numpy import concatenate
from numpy import empty
from numpy import ndarray
from numpy import setdiff1d
from numpy import vstack
from numpy.random import default_rng

from gemseo.mlearning.resampling.base_resampler import BaseResampler
from gemseo.mlearning.resampling.split import Split
from gemseo.mlearning.resampling.splits import Splits
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.mlearning import MLAlgo


class CrossValidation(BaseResampler):
    """A cross-validation tool for resampling and surrogate modeling."""

    __randomize: bool
    """Whether the sample indices are shuffled before splitting."""

    __shuffled_sample_indices: NDArray[int]
    """The indices of the samples after shuffling."""

    def __init__(
        self,
        sample_indices: NDArray[int],
        n_folds: int = 5,
        randomize: bool = False,
        seed: int | None = SEED,
    ) -> None:
        """
        Args:
            n_folds: The number of folds.
            randomize: Whether the sample indices are shuffled before splitting.
        """  # noqa: D205 D212
        self.__randomize = randomize
        self.__shuffled_sample_indices = sample_indices.copy()
        if randomize:
            default_rng(seed).shuffle(self.__shuffled_sample_indices)
        super().__init__(sample_indices, n_splits=n_folds, seed=seed)
        if len(sample_indices) == n_folds:
            self.name = "LeaveOneOut"

    def execute(
        self,
        model: MLAlgo,
        return_models: bool = False,
        input_data: ndarray | None = None,
        stack_predictions: bool = True,
        fit_transformers: bool = True,
        store_sampling_result: bool = False,
    ) -> tuple[list[MLAlgo], list[ndarray] | ndarray]:
        """
        Args:
            stack_predictions: Whether the sub-predictions are stacked
                in the order of the ``sample_indices`` passed at instantiation
                (first the prediction at index ``sample_indices[0]``,
                then the prediction at index ``sample_indices[1]``,
                etc.).
                This argument is ignored when ``input_data`` is ``None``.
        """  # noqa: D205, D212, D415
        return super().execute(
            model,
            return_models=return_models,
            input_data=input_data,
            stack_predictions=stack_predictions,
            fit_transformers=fit_transformers,
            store_sampling_result=store_sampling_result,
        )

    def _create_splits(self) -> Splits:
        return Splits(*[
            Split(
                setdiff1d(self.__shuffled_sample_indices, test_indices),
                test_indices,
            )
            for test_indices in array_split(
                self.__shuffled_sample_indices, self._n_splits
            )
        ])

    @property
    def shuffled_sample_indices(self) -> NDArray[int]:
        """The original indices of the samples."""
        return self.__shuffled_sample_indices

    @property
    def n_folds(self) -> int:
        """The number of folds."""
        return self._n_splits

    @property
    def randomize(self) -> bool:
        """Whether the sample indices are shuffled before splitting."""
        return self.__randomize

    def _post_process_predictions(
        self,
        predictions: list[ndarray],
        stack_predictions: bool,
    ) -> ndarray | list[ndarray]:
        if stack_predictions:
            n_predictions = sum(len(prediction) for prediction in predictions)
            predictions_0 = predictions[0]
            if predictions_0.ndim == 1:
                final_predictions = empty((n_predictions,))
                function = concatenate
            else:
                final_predictions = empty((n_predictions, predictions_0.shape[1]))
                function = vstack

            final_predictions[self.__shuffled_sample_indices] = function(predictions)
            return final_predictions
        return predictions
