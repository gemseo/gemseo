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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""This module contains a class to select a machine learning algorithm from a list.

Machine learning is used to find relations or underlying structures in data.
There is however no algorithm that is universally better than the others
for an arbitrary problem.
As for optimization, there is *no free lunch* for machine learning :cite:`wolpert`.

Provided a quality measure,
one can thus compare the performances of different machine learning algorithms.

This process can be easily performed using the class :class:`.MLAlgoSelection`.

A machine learning algorithm is built using a set of (hyper)parameters,
before the learning takes place.
In order to choose the best hyperparameters,
a simple grid search over different values may be sufficient.
The :class:`.MLAlgoSelection` does this.
It can also perform a more advanced form of optimization
than a simple grid search over predefined values,
using the class :class:`.MLAlgoCalibration`.

.. seealso::

   :mod:`~gemseo.mlearning.core.ml_algo`
   :mod:`~gemseo.mlearning.core.calibration`
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from gemseo.mlearning.core.calibration import MLAlgoCalibration
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasure
from gemseo.mlearning.quality_measures.quality_measure import MLQualityMeasureFactory
from gemseo.mlearning.quality_measures.quality_measure import (
    OptionType as MeasureOptionType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.scenario import ScenarioInputDataType
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.ml_algo import MLAlgo


class MLAlgoSelection:
    """Machine learning algorithm selector."""

    dataset: Dataset
    """The learning dataset."""

    measure: str
    """The name of a quality measure to measure the quality of the machine learning
    algorithms."""

    measure_options: dict[str, int | Dataset]
    """The options for the method to evaluate the quality measure."""

    factory: MLAlgoFactory
    """The factory used for the instantiation of machine learning algorithms."""

    candidates: list[tuple[MLAlgo, float]]
    """The candidate machine learning algorithms, after possible calibration, and their
    quality measures."""

    def __init__(
        self,
        dataset: Dataset,
        measure: str | MLQualityMeasure,
        measure_evaluation_method_name: MLQualityMeasure.EvaluationMethod = MLQualityMeasure.EvaluationMethod.LEARN,  # noqa: E501
        samples: Sequence[int] | None = None,
        **measure_options: MeasureOptionType,
    ) -> None:
        """
        Args:
            dataset: The learning dataset.
            measure: The name of a quality measure
                to measure the quality of the machine learning algorithms.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            samples: The indices of the learning samples to consider.
                Other indices are neither used for training nor for testing.
                If ``None``, use all the samples.
            **measure_options: The options for the method
                to evaluate the quality measure.
                The option 'multioutput' will be set to False.

        Raises:
            ValueError: If the unsupported "multioutput" option is enabled.
        """  # noqa: D205 D212
        self.dataset = dataset
        if isinstance(measure, str):
            self.measure = MLQualityMeasureFactory().get_class(measure)
        else:
            self.measure = measure

        self.__measure_evaluation_method_name = measure_evaluation_method_name
        self.measure_options = dict(samples=samples, **measure_options)
        self.factory = MLAlgoFactory()

        self.candidates = []

        if self.measure_options.get("multioutput", False):
            raise ValueError(
                "MLAlgoSelection does not support multioutput; "
                "the measure shall return one value."
            )
        self.measure_options["multioutput"] = False

    def add_candidate(
        self,
        name: str,
        calib_space: DesignSpace | None = None,
        calib_algo: ScenarioInputDataType | None = None,
        **option_lists,
    ) -> None:
        """Add a machine learning algorithm candidate.

        Args:
            name: The name of a machine learning algorithm.
            calib_space: The design space
                defining the parameters to be calibrated
                with an :class:`.MLAlgoCalibration`.
                If ``None``, do not perform calibration.
            calib_algo: The name and the parameters
                of the optimization algorithm,
                e.g. {"algo": "fullfact", "n_samples": 10}.
                If ``None``, do not perform calibration.
            **option_lists: The parameters
                for the machine learning algorithm candidate.
                Each parameter has to be enclosed within a list.
                The list may contain different values
                to try out for the given parameter,
                or only one.

        Examples:
            >>> selector.add_candidate(
            >>>     "LinearRegressor",
            >>>     penalty_level=[0, 0.1, 1, 10, 20],
            >>>     l2_penalty_ratio=[0, 0.5, 1],
            >>>     fit_intercept=[True],
            >>> )
        """
        keys, values = option_lists.keys(), option_lists.values()

        # Set initial quality to the worst possible value
        quality = float("inf") if self.measure.SMALLER_IS_BETTER else -float("inf")

        for prodvalues in product(*values):
            params = dict(zip(keys, prodvalues))
            if calib_space:
                ml_algo_calibration = MLAlgoCalibration(
                    name,
                    self.dataset,
                    calib_space.variable_names,
                    calib_space,
                    self.measure,
                    measure_evaluation_method_name=self.__measure_evaluation_method_name,
                    measure_options=self.measure_options,
                    **params,
                )
                ml_algo_calibration.execute(calib_algo)
                algo_new = ml_algo_calibration.optimal_algorithm
                quality_new = ml_algo_calibration.optimal_criterion
            else:
                algo_new = self.factory.create(name, data=self.dataset, **params)
                quality_measurer = self.measure(algo_new)
                compute_quality_measure = getattr(
                    quality_measurer,
                    quality_measurer.EvaluationFunctionName[
                        self.__measure_evaluation_method_name
                    ],
                )
                quality_new = compute_quality_measure(**self.measure_options)

            if self.measure.is_better(quality_new, quality):
                algo = algo_new
                quality = quality_new

        if not algo.is_trained:
            algo.learn(self.measure_options["samples"])

        self.candidates.append((algo, quality))

    def select(
        self,
        return_quality: bool = False,
    ) -> MLAlgo | tuple[MLAlgo, float]:
        """Select the best model.

        The model is chosen through a grid search
        over candidates and their options,
        as well as an eventual optimization over the parameters
        in the calibration space.

        Args:
            return_quality: Whether to return the quality of the best model.

        Returns:
            The best model and its quality if required.
        """
        candidate = self.candidates[0]
        for new_candidate in self.candidates[1:]:
            if self.measure.is_better(new_candidate[1], candidate[1]):
                candidate = new_candidate

        if return_quality:
            return candidate

        return candidate[0]
