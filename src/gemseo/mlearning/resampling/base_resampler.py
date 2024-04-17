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

from numpy import array
from numpy import concatenate
from numpy import ndarray
from numpy import vstack

from gemseo.datasets.dataset import Dataset
from gemseo.post.dataset.scatter import Scatter
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
    from gemseo.mlearning.resampling.splits import Splits


class BaseResampler(metaclass=ABCGoogleDocstringInheritanceMeta):
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

    def __eq__(self, other: BaseResampler) -> bool:
        return self._splits == other._splits

    def execute(
        self,
        model: BaseMLAlgo,
        return_models: bool = False,
        input_data: ndarray | None = None,
        stack_predictions: bool = True,
        fit_transformers: bool = True,
        store_sampling_result: bool = False,
    ) -> tuple[list[BaseMLAlgo], list[ndarray] | ndarray]:
        """Apply the resampling technique to a machine learning model.

        Args:
            model: The machine learning model.
            return_models: Whether the sub-models resulting
                from resampling are returned.
            input_data: The input data for the prediction, if any.
            stack_predictions: Whether the sub-predictions are stacked per sub-model
                (first the predictions of the first sub-model,
                then the prediction of the second sub-model,
                etc.).
                This argument is ignored when ``input_data`` is ``None``.
            fit_transformers: Whether to re-fit the transformers.
            store_sampling_result: Whether to store the sampling results
                in the attribute :class:`~.BaseMLAlgo.resampling_results`
                of the original model.

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
        predict = input_data is not None
        for split in self._splits:
            if return_models:
                sub_model = deepcopy(model)
                sub_models.append(sub_model)

            sub_model.learn(samples=split.train, fit_transformers=fit_transformers)
            if predict:
                predictions.append(sub_model.predict(input_data[split.test]))

        if predict:
            predictions = self._post_process_predictions(predictions, stack_predictions)

        if store_sampling_result:
            model.resampling_results[self.name] = (self, sub_models, predictions)

        return sub_models, predictions

    def _post_process_predictions(
        self, predictions: list[ndarray], stack_predictions: bool
    ) -> ndarray | list[ndarray]:
        """Stack the predictions if required.

        Args:
            predictions: The predictions per fold.
            stack_predictions: Whether to stack the predictions.

        Returns:
            The predictions, either stacked or as is.
        """
        if stack_predictions:
            return (concatenate if predictions[0].ndim == 1 else vstack)(predictions)
        return predictions

    def plot(
        self,
        file_path: str | Path = "",
        show: bool = True,
        colors: tuple[str, str] = ("b", "r"),
    ) -> Scatter:
        """Plot the train-test splits.

        Args:
            file_path: The file path to save the figure.
                If empty, do not save the figure.
            show: Whether to display the figure.
            colors: The colors for training and test points.

        Returns:
            The visualization.
        """
        index = []
        color = []
        split = []
        training_point_color, test_point_color = colors
        for i, _split in enumerate(self._splits):
            train = _split.train
            test = _split.test
            split.extend([i] * train.size)
            index.extend(train)
            color.extend([training_point_color] * train.size)
            split.extend([i] * test.size)
            index.extend(test)
            color.extend([test_point_color] * test.size)

        dataset = Dataset.from_array(
            array([split, index]).T,
            variable_names=["Split", "Index"],
        )
        scatter = Scatter(dataset, "Index", "Split")
        scatter.color = color
        scatter.execute(save=file_path != "", show=show, file_path=file_path)
        return scatter
