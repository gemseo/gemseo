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
"""Here is the baseclass to measure the error of machine learning algorithms.

The concept of error measure is implemented with the :class:`.MLErrorMeasure` class and
proposes different evaluation methods.
"""
from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import Sequence

from numpy import arange
from numpy import atleast_1d
from numpy import delete as npdelete
from numpy import ndarray
from numpy import unique

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.qual_measure.quality_measure import MeasureType
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.python_compatibility import Final


class MLErrorMeasure(MLQualityMeasure):
    """An abstract error measure for machine learning."""

    __OUTPUT_NAME_SEPARATOR: Final[str] = "#"
    """A string to join output names."""

    _GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT: Final[dict[bool, str]] = {
        True: "raw_values",
        False: "uniform_average",
    }
    """Map from the argument "multioutput" of |g| to that of sklearn."""

    def __init__(
        self,
        algo: MLSupervisedAlgo,
        fit_transformers: bool = MLQualityMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for supervised learning.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def evaluate_learn(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
        """
        self._train_algo(samples)
        return self._post_process_measure(
            self._compute_measure(
                self.algo.output_data,
                self.algo.predict(self.algo.input_data),
                multioutput,
            ),
            multioutput,
            as_dict,
        )

    def evaluate_test(
        self,
        test_data: Dataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
        """
        self._train_algo(samples)
        return self._post_process_measure(
            self._compute_measure(
                test_data.get_data_by_names(self.algo.output_names, False),
                self.algo.predict(
                    test_data.get_data_by_names(self.algo.input_names, False)
                ),
                multioutput,
            ),
            multioutput,
            as_dict,
        )

    def evaluate_loo(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
        """
        return self.evaluate_kfolds(
            samples=samples,
            n_folds=self.algo.learning_set.n_samples,
            multioutput=multioutput,
            as_dict=as_dict,
        )

    def evaluate_kfolds(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = MLQualityMeasure._RANDOMIZE,
        seed: int | None = None,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
        """
        self._train_algo(samples)
        samples = self._assure_samples(samples)
        folds, samples = self._compute_folds(samples, n_folds, randomize, seed)

        input_data = self.algo.input_data
        output_data = self.algo.output_data

        algo = deepcopy(self.algo)

        qualities = []
        for fold in folds:
            algo.learn(
                samples=npdelete(samples, fold), fit_transformers=self._fit_transformers
            )
            expected = output_data[fold]
            predicted = algo.predict(input_data[fold])
            quality = self._compute_measure(expected, predicted, multioutput)
            qualities.append(quality)

        return self._post_process_measure(
            sum(qualities) / len(qualities), multioutput, as_dict
        )

    def evaluate_bootstrap(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: None | None = None,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether to express the measure as a dictionary
                whose keys are the output names.
        """
        samples = self._assure_samples(samples)
        self._train_algo(samples)
        n_samples = samples.size
        input_data = self.algo.input_data
        output_data = self.algo.output_data

        all_indices = arange(n_samples)

        algo = deepcopy(self.algo)

        qualities = []
        generator = self._get_rng(seed)
        for _ in range(n_replicates):
            training_indices = unique(generator.choice(n_samples, n_samples))
            test_indices = npdelete(all_indices, training_indices)
            algo.learn(
                [samples[index] for index in training_indices],
                fit_transformers=self._fit_transformers,
            )
            test_samples = [samples[index] for index in test_indices]
            quality = self._compute_measure(
                output_data[test_samples],
                algo.predict(input_data[test_samples]),
                multioutput,
            )
            qualities.append(quality)

        return self._post_process_measure(
            sum(qualities) / len(qualities), multioutput, as_dict
        )

    @abstractmethod
    def _compute_measure(
        self,
        outputs: ndarray,
        predictions: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        """Compute the quality measure.

        Args:
            outputs: The reference data.
            predictions: The predicted labels.
            multioutput: Whether to return the quality measure
                for each output component. If not, average these measures.

        Returns:
            The value of the quality measure.
        """

    def _post_process_measure(
        self, measure: float | ndarray, multioutput: bool, as_dict: bool
    ) -> MeasureType:
        """Post-process a measure.

        Args:
            measure: The measure to post-process.
            multioutput: If ``True``, return the quality measure for each
                output component. Otherwise, average these measures.
            as_dict: Whether to return the measure as is or as a dictionary
                whose keys are the output names.

        Returns:
            The post-processed measure.
        """
        if not as_dict:
            return measure

        data = atleast_1d(measure)
        names = self.algo.output_names
        if not multioutput:
            return {self.__OUTPUT_NAME_SEPARATOR.join(names): data}

        return split_array_to_dict_of_arrays(data, self.algo.learning_set.sizes, names)
