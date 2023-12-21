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
"""A base class for resampling and surrogate modeling."""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from numpy import concatenate
from numpy import ndarray
from numpy import vstack

from gemseo import SEED
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from gemseo.mlearning.core.ml_algo import MLAlgo
    from gemseo.mlearning.resampling.splits import Splits


class Resampler(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A base class for resampling and surrogate modeling."""

    name: str
    """The name of the resampler.

    Use the class name by default.
    """

    _sample_indices: NDArray[int]
    """The original indices of the samples."""

    _seed: int | None
    """The seed to initialize the random generator.

    If ``None``,
    then fresh, unpredictable entropy will be pulled from the OS.
    """

    _splits: Splits
    """The train-test splits resulting from the splitting of the samples.

    A train-test split is a partition whose first component contains the indices of the
    learning samples and the second one the indices of the test samples.
    """

    _n_splits: int
    """The number of train-test splits."""

    def __init__(
        self,
        sample_indices: NDArray[int],
        n_splits: int,
        seed: int | None = SEED,
    ) -> None:
        """
        Args:
            sample_indices: The original indices of the samples.
            n_splits: The number of train-test splits.
            seed: The seed to initialize the random generator.
                If ``None``,
                then fresh, unpredictable entropy will be pulled from the OS.
        """  # noqa: D205 D212
        self._n_splits = n_splits
        self._seed = seed
        self._sample_indices = sample_indices
        self._splits = self._create_splits()
        self.name = self.__class__.__name__

    @abstractmethod
    def _create_splits(self) -> Splits:
        """Create the train-test splits."""

    @property
    def sample_indices(self) -> NDArray[int]:
        """The indices of the samples after shuffling."""
        return self._sample_indices

    @property
    def seed(self) -> int:
        """The seed to initialize the random generator."""
        return self._seed

    @property
    def splits(self) -> Splits:
        """The train-test splits resulting from the splitting of the samples.

        A train-test split is a partition whose first component contains the indices of
        the learning samples and the second one the indices of the test samples.
        """
        return self._splits

    def __eq__(self, other: Resampler) -> bool:
        return self._splits == other._splits

    def execute(
        self,
        model: MLAlgo,
        return_models: bool,
        predict: bool,
        stack_predictions: bool,
        fit_transformers: bool,
        store_sampling_result: bool,
        input_data: ndarray,
        output_data_shape: tuple[int, ...],
    ) -> tuple[list[MLAlgo], list[ndarray] | ndarray]:
        """Apply the resampling technique to a machine learning model.

        Args:
            model: The machine learning model.
            return_models: Whether the sub-models resulting
                from resampling are returned.
            predict: Whether the sub-models resulting from sampling do prediction
                on their corresponding learning data.
            stack_predictions: Whether the sub-predictions are stacked.
            fit_transformers: Whether to re-fit the transformers.
            store_sampling_result: Whether to store the sampling results
                in the attribute :class:`~.MLAlgo.resampling_results`
                of the original model.
            input_data: The input data.
            output_data_shape: The shape of the output data array.

        Returns:
            First the sub-models resulting from resampling
            if ``return_models`` is ``True``
            then the predictions, either per fold or stacked.

        Raises:
            ValueError: When the model is
                neither a supervised algorithm nor a clustering one.
        """
        if self.name in model.resampling_results:
            (resampler, sub_models, predictions) = model.resampling_results[self.name]
            if self == resampler:
                return sub_models, predictions

        if not return_models:
            sub_model = deepcopy(model)

        predictions = []
        sub_models = []
        for split in self._splits:
            if return_models:
                sub_model = deepcopy(model)
                sub_models.append(sub_model)

            sub_model.learn(samples=split.train, fit_transformers=fit_transformers)
            if predict:
                predictions.append(sub_model.predict(input_data[split.test]))

        if predict:
            predictions = self._post_process_predictions(
                predictions, output_data_shape, stack_predictions
            )

        if store_sampling_result:
            model.resampling_results[self.name] = (self, sub_models, predictions)

        return sub_models, predictions

    def _post_process_predictions(
        self,
        predictions: list[ndarray],
        output_data_shape: tuple[int, ...],
        stack_predictions: bool,
    ) -> ndarray | list[ndarray]:
        """Stack the predictions if required.

        Args:
            predictions: The predictions per fold.
            output_data_shape: The shape of the full learning output data.
            stack_predictions: Whether to stack the predictions.

        Returns:
            The predictions, either stacked or as is.
        """
        if stack_predictions:
            function = concatenate if len(output_data_shape) == 1 else vstack
            return function(predictions)
        return predictions
