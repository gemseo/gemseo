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
"""The base class to assess the quality of a regressor."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Final

from numpy import atleast_1d

from gemseo.mlearning.core.quality.base_ml_algo_quality import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType
from gemseo.mlearning.resampling.bootstrap import Bootstrap
from gemseo.mlearning.resampling.cross_validation import CrossValidation
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.supervised import BaseMLSupervisedAlgo
    from gemseo.typing import RealArray


class BaseRegressorQuality(BaseMLAlgoQuality):
    """The base class to assess the quality of a regressor."""

    __OUTPUT_NAME_SEPARATOR: Final[str] = "#"
    """A string to join output names."""

    _GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT: Final[dict[bool, str]] = {
        True: "raw_values",
        False: "uniform_average",
    }
    """Map from the argument "multioutput" of |g| to that of sklearn."""

    algo: BaseMLSupervisedAlgo

    def __init__(
        self,
        algo: BaseMLSupervisedAlgo,
        fit_transformers: bool = BaseMLAlgoQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for supervised learning.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def compute_learning_measure(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.
        """  # noqa: D205 D212
        self._pre_process(samples)
        return self._post_process_measure(
            self._compute_measure(
                self.algo.output_data,
                self.algo.predict(self.algo.input_data),
                multioutput,
            ),
            multioutput,
            as_dict,
        )

    def compute_test_measure(
        self,
        test_data: IODataset,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.
        """  # noqa: D205 D212
        self._pre_process(samples)
        return self._post_process_measure(
            self._compute_measure(
                test_data.get_view(
                    group_names=test_data.OUTPUT_GROUP,
                    variable_names=self.algo.output_names,
                ).to_numpy(),
                self.algo.predict(
                    test_data.get_view(
                        group_names=test_data.INPUT_GROUP,
                        variable_names=self.algo.input_names,
                    ).to_numpy()
                ),
                multioutput,
            ),
            multioutput,
            as_dict,
        )

    def compute_leave_one_out_measure(
        self,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.
        """  # noqa: D205 D212
        return self.compute_cross_validation_measure(
            samples=samples,
            n_folds=self.algo.learning_set.n_samples,
            multioutput=multioutput,
            as_dict=as_dict,
            store_resampling_result=store_resampling_result,
            seed=1,
        )

    def compute_cross_validation_measure(
        self,
        n_folds: int = 5,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        randomize: bool = BaseMLAlgoQuality._RANDOMIZE,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.
        """  # noqa: D205 D212
        samples, seed = self._pre_process(samples, seed, randomize)
        cross_validation = CrossValidation(samples, n_folds, randomize, seed)
        _, predictions = cross_validation.execute(
            self.algo,
            return_models=store_resampling_result,
            input_data=self.algo.input_data,
            fit_transformers=self._fit_transformers,
            store_sampling_result=store_resampling_result,
        )
        return self._post_process_measure(
            self._compute_measure(self.algo.output_data, predictions, multioutput),
            multioutput,
            as_dict,
        )

    def compute_bootstrap_measure(
        self,
        n_replicates: int = 100,
        samples: Sequence[int] | None = None,
        multioutput: bool = True,
        seed: int | None = None,
        as_dict: bool = False,
        store_resampling_result: bool = False,
    ) -> MeasureType:
        """
        Args:
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.
        """  # noqa: D205 D212
        samples, seed = self._pre_process(samples, seed, True)
        bootstrap = Bootstrap(samples, n_replicates, seed)
        _, predictions = bootstrap.execute(
            self.algo,
            return_models=store_resampling_result,
            input_data=self.algo.input_data,
            stack_predictions=False,
            fit_transformers=self._fit_transformers,
            store_sampling_result=store_resampling_result,
        )
        output_data = self.algo.output_data
        measure = 0
        for prediction, split in zip(predictions, bootstrap.splits):
            measure += self._compute_measure(
                output_data[split.test], prediction, multioutput
            )
        return self._post_process_measure(measure / n_replicates, multioutput, as_dict)

    @abstractmethod
    def _compute_measure(
        self,
        outputs: RealArray,
        predictions: RealArray,
        multioutput: bool = True,
    ) -> MeasureType:
        """Compute the quality measure.

        Args:
            outputs: The reference data.
            predictions: The predicted labels.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.

        Returns:
            The value of the quality measure.
        """

    def _post_process_measure(
        self, measure: float | RealArray, multioutput: bool, as_dict: bool
    ) -> MeasureType:
        """Post-process a measure.

        Args:
            measure: The measure to post-process.
            multioutput: Whether the quality measure is returned
                for each component of the outputs.
                Otherwise, the average quality measure.
            as_dict: Whether the full quality measure is returned
                as a mapping from ``algo.output names`` to quality measures.
                Otherwise,
                the full quality measure as an array
                stacking these quality measures
                according to the order of ``algo.output_names``.

        Returns:
            The post-processed measure.
        """
        if not as_dict:
            return measure

        data = atleast_1d(measure)
        names = self.algo.output_names
        if not multioutput:
            return {self.__OUTPUT_NAME_SEPARATOR.join(names): data}

        return split_array_to_dict_of_arrays(
            data, self.algo.learning_set.variable_names_to_n_components, names
        )
