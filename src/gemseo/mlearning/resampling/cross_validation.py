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

from gemseo import SEED
from gemseo.mlearning.resampling.resampler import Resampler
from gemseo.mlearning.resampling.split import Split
from gemseo.mlearning.resampling.splits import Splits

if TYPE_CHECKING:
    from numpy.typing import NDArray


class CrossValidation(Resampler):
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
        output_data_shape: tuple[int, ...],
        stack_predictions: bool,
    ) -> ndarray | list[ndarray]:
        if stack_predictions:
            function = concatenate if len(output_data_shape) == 1 else vstack
            final_predictions = empty(output_data_shape)
            final_predictions[self.__shuffled_sample_indices] = function(predictions)
            return final_predictions
        return predictions
