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
"""This module contains a class to select a machine learning model from a list.

Machine learning is used to find relations or underlying structures in data.
There is however no model that is universally better than the others
for an arbitrary problem.

Provided a quality measure,
one can thus compare the performances of different machine learning models.

This process can be easily performed
using the class [MLModelSelection][gemseo.mlearning.core.selection.MLModelSelection].

A machine learning model is built using a set of (hyper)parameters,
before the learning takes place.
In order to choose the best hyperparameters,
a simple grid search over different values may be sufficient.
The [MLModelSelection][gemseo.mlearning.core.selection.MLModelSelection] does this.
It can also perform a more advanced form of optimization
than a simple grid search over predefined values,
using the class
[MLModelCalibration][gemseo.mlearning.core.calibration.MLModelCalibration].

See Also:
   [gemseo.mlearning.core.models.ml_model][gemseo.mlearning.core.models.ml_model]
   [gemseo.mlearning.core.calibration][gemseo.mlearning.core.calibration]
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING
from typing import Any

from gemseo.mlearning.core.calibration import MLModelCalibration
from gemseo.mlearning.core.models.factory import ML_MODEL_FACTORY
from gemseo.mlearning.core.quality.base_ml_model_quality import BaseMLModelQuality
from gemseo.mlearning.core.quality.factory import MLModelQualityFactory

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.models.factory import MLModelFactory
    from gemseo.mlearning.core.models.ml_model import BaseMLModel
    from gemseo.mlearning.core.models.ml_model_settings import BaseMLModelSettings
    from gemseo.mlearning.core.quality.base_ml_model_quality import (
        OptionType as MeasureOptionType,
    )


class MLModelSelection:
    """Machine learning model selector."""

    dataset: Dataset
    """The training dataset."""

    measure: type[BaseMLModelQuality]
    """The name of a quality measure to measure the quality of the machine learning
    models."""

    measure_options: dict[str, int | Dataset]
    """The options for the method to evaluate the quality measure."""

    factory: MLModelFactory
    """The factory used for the instantiation of machine learning models."""

    candidates: list[tuple[BaseMLModel, float]]
    """The candidate machine learning models, after possible calibration, and their
    quality measures."""

    def __init__(
        self,
        dataset: Dataset,
        measure: str | type[BaseMLModelQuality],
        measure_evaluation_method_name: BaseMLModelQuality.EvaluationMethod = BaseMLModelQuality.EvaluationMethod.LEARN,  # noqa: E501
        samples: Sequence[int] = (),
        **measure_options: MeasureOptionType,
    ) -> None:
        """
        Args:
            dataset: The training dataset.
            measure: The name of a quality measure
                to measure the quality of the machine learning models.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            samples: The indices of the learning samples to consider.
                Other indices are neither used for training nor for testing.
                If empty, use all the samples.
            **measure_options: The options for the method
                to evaluate the quality measure.
                The option 'multioutput' will be set to False.

        Raises:
            ValueError: If the unsupported "multioutput" option is enabled.
        """  # noqa: D205 D212
        self.dataset = dataset
        if isinstance(measure, str):
            self.measure = MLModelQualityFactory().get_class(measure)
        else:
            self.measure = measure

        self.__measure_evaluation_method_name = measure_evaluation_method_name
        self.measure_options = dict(samples=samples, **measure_options)
        self.factory = ML_MODEL_FACTORY

        self.candidates = []

        if self.measure_options.get("multioutput", False):
            msg = (
                "MLModelSelection does not support multioutput; "
                "the measure shall return one value."
            )
            raise ValueError(msg)
        self.measure_options["multioutput"] = False

    def add_candidate(
        self,
        settings: BaseMLModelSettings,
        calibration_space: DesignSpace | None = None,
        calibration_settings: BaseDriverSettings | None = None,
        **settings_catalogs: Iterable[Any],
    ) -> None:
        """Add a machine learning model candidate.

        Args:
            settings: The settings of the machine learning model candidate.
            calibration_space: The space defining the settings to calibrate, if any.
            calibration_settings: The settings of the driver for calibration.
            **settings_catalogs: The catalogs of settings.
                Unlike the settings to calibrate,
                these settings are optimized using a grid search over the catalogs.
        """
        keys, values = settings_catalogs.keys(), settings_catalogs.values()

        # Set initial quality to the worst possible value
        quality = float("inf") if self.measure.SMALLER_IS_BETTER else -float("inf")

        for prodvalues in product(*values):
            params = dict(zip(keys, prodvalues, strict=False))
            for class_name, value in params.items():
                setattr(settings, class_name, value)
            if calibration_space:
                ml_model_calibration = MLModelCalibration(
                    settings,
                    self.dataset,
                    calibration_space,
                    self.measure,
                    measure_evaluation_method_name=self.__measure_evaluation_method_name,
                    measure_options=self.measure_options,
                )
                ml_model_calibration.execute(calibration_settings)
                model_new = ml_model_calibration.optimal_model
                quality_new = ml_model_calibration.optimal_criterion
            else:
                model_new = self.factory.create(
                    settings._TARGET_CLASS_NAME, self.dataset, settings=settings
                )
                quality_measurer = self.measure(model_new)
                compute_quality_measure = getattr(
                    quality_measurer,
                    quality_measurer.EvaluationFunctionName[
                        self.__measure_evaluation_method_name
                    ],
                )
                quality_new = compute_quality_measure(**self.measure_options)

            if self.measure.is_better(quality_new, quality):
                model = model_new
                quality = quality_new

        self.candidates.append((model, quality))

    def select(
        self,
        return_quality: bool = False,
    ) -> BaseMLModel | tuple[BaseMLModel, float]:
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
