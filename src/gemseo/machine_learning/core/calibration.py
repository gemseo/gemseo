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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Calibration of a machine learning model.

A machine learning model depends on hyper-parameters,
e.g. the number of clusters for
a clustering model, the regularization constant for a regression model, the kernel
for a Gaussian process regression, ... Its ability to generalize the information learned
during the training stage, and thus to avoid over-fitting, which is an over-reliance on
the learning data set, depends on the values of these hyper-parameters. Thus, the hyper-
parameters minimizing the learning quality measure are rarely those minimizing the
generalization one.
Classically, the generalization one decreases before growing again as
the model becomes more complex, while the learning error keeps decreasing. This
phenomenon is called the curse of dimensionality.

In this module,
the
[MLModelCalibration][gemseo.machine_learning.core.calibration.MLModelCalibration]
class aims to calibrate the hyper-parameters
in order to minimize this measure of the generalization quality
over a calibration parameter space.
This class relies on the
[MLModelAssessor][gemseo.machine_learning.core.calibration.MLModelAssessor] class
which is a discipline ([Discipline][gemseo.core.discipline.discipline.Discipline])
built from a machine learning model
([BaseMLModel][gemseo.machine_learning.core.models.ml_model.BaseMLModel]),
a dataset ([Dataset][gemseo.datasets.dataset.Dataset]), a quality measure
([BaseMLModelQuality][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality])
and various options for the data scaling, the quality
measure and the machine learning model.
The inputs of this discipline are hyper-parameters of the machine learning model
while the output is the quality criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import argmin
from numpy import array

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.core.discipline import Discipline
from gemseo.machine_learning.core.models.factory import ML_MODEL_FACTORY
from gemseo.machine_learning.core.quality.base_ml_model_quality import (
    BaseMLModelQuality,
)
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.algos.base_driver_settings import BaseDriverSettings
    from gemseo.algos.design_space import DesignSpace
    from gemseo.datasets.dataset import Dataset
    from gemseo.machine_learning.core.models.ml_model import BaseMLModel
    from gemseo.machine_learning.core.models.ml_model_settings import (
        BaseMLModelSettings,
    )
    from gemseo.machine_learning.core.quality.base_ml_model_quality import (
        MeasureOptionsType,
    )
    from gemseo.scenarios.base_scenario import BaseScenario
    from gemseo.typing import StrKeyMapping


class MLModelAssessor(Discipline):
    """Discipline assessing the quality of a machine learning model.

    This quality depends on the values of parameters to calibrate with the
    [MLModelCalibration][gemseo.machine_learning.core.calibration.MLModelCalibration].
    """

    __measure: type[BaseMLModelQuality]
    """The measure to assess the machine learning model."""

    __measure_options: dict[str, int | Dataset]
    """The options of the quality measure."""

    __settings: BaseMLModelSettings
    """The settings of the machine learning model."""

    __training_dataset: Dataset
    """The training dataset."""

    models: list[BaseMLModel]
    """The instances of the machine learning model (one per execution of the machine
    learning model assessor)."""

    CRITERION = "criterion"
    LEARNING = "learning"
    MULTIOUTPUT = "multioutput"

    def __init__(
        self,
        settings: BaseMLModelSettings,
        dataset: Dataset,
        parameters: Iterable[str],
        measure: type[BaseMLModelQuality],
        measure_evaluation_method_name: BaseMLModelQuality.EvaluationMethod = BaseMLModelQuality.EvaluationMethod.LEARN,  # noqa: E501
        measure_options: MeasureOptionsType = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            settings: The settings of the machine learning model.
            dataset: The training dataset.
            parameters: The parameters of the machine learning model to calibrate.
            measure: The measure to assess the machine learning model.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            measure_options: The options of the quality measure.
                If "multioutput" is missing,
                it is added with False as value.
                If empty, do not use quality measure options.

        Raises:
            ValueError: If the measure option "multioutput" is True.
        """  # noqa: D205 D212
        super().__init__()
        self.io.input_grammar.update_from_names(parameters)
        self.io.output_grammar.update_from_names([self.CRITERION, self.LEARNING])
        self.model_name = settings._TARGET_CLASS_NAME
        self.__measure = measure
        self.__measure_options = dict(measure_options)
        self.__measure_evaluation_method_name = measure_evaluation_method_name
        self.__settings = settings
        self.__training_dataset = dataset
        self.models = []
        if self.__measure_options.get("multioutput", False):
            msg = "MLModelAssessor does not support multioutput."
            raise ValueError(msg)

        self.__measure_options[self.MULTIOUTPUT] = False

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        """Run method.

        This method creates a new instance of the machine learning model, from the
        hyper-parameters stored in the data attribute of the
        [MLModelAssessor][gemseo.machine_learning.core.calibration.MLModelAssessor].
        It trains it on the training dataset and measures its
        quality with the
        [BaseMLModelQuality][gemseo.machine_learning.core.quality.base_ml_model_quality.BaseMLModelQuality].
        """
        settings = self.__settings.model_copy()
        inputs = self.io.get_input_data()
        for index in inputs:
            if len(inputs[index]) == 1:
                inputs[index] = inputs[index][0]
        for name, value in inputs.items():
            setattr(settings, name, value)

        model = ML_MODEL_FACTORY.create_from_settings(settings, self.__training_dataset)
        model.learn()
        measure = self.__measure(model)
        compute_criterion = getattr(
            measure,
            measure.EvaluationFunctionName[self.__measure_evaluation_method_name],
        )
        self.models.append(model)
        return {
            "criterion": array([compute_criterion(**self.__measure_options)]),
            "learning": array([measure.compute_learning_measure(multioutput=False)]),
        }


class MLModelCalibration:
    """Calibration of a machine learning model."""

    model_assessor: MLModelAssessor
    """The assessor for the machine learning model."""

    calibration_space: DesignSpace
    """The space defining the calibration variables."""

    maximize_objective: bool
    """Whether to maximize the quality measure."""

    dataset: Dataset | None
    """The training dataset after execution."""

    optimal_parameters: dict[str, ndarray] | None
    """The optimal parameters for the machine learning model after execution."""

    optimal_criterion: float | None
    """The optimal quality measure after execution."""

    optimal_model: BaseMLModel | None
    """The optimal machine learning model after execution."""

    scenario: BaseScenario | None
    """The scenario used to calibrate the machine learning model after execution."""

    def __init__(
        self,
        settings: BaseMLModelSettings,
        dataset: Dataset,
        calibration_space: DesignSpace,
        measure: type[BaseMLModelQuality],
        measure_evaluation_method_name: str
        | BaseMLModelQuality.EvaluationMethod = BaseMLModelQuality.EvaluationMethod.LEARN,  # noqa: E501
        measure_options: MeasureOptionsType = READ_ONLY_EMPTY_DICT,
    ) -> None:
        """
        Args:
            settings: The settings of the machine learning model.
            dataset: The training dataset.
            calibration_space: The space defining the settings to calibrate.
            measure: The measure to assess the machine learning model.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            measure_options: The options of the quality measure.
                If empty, do not use the quality measure options.
        """  # noqa: D205 D212
        model_assessor = MLModelAssessor(
            settings,
            dataset,
            calibration_space.variable_names,
            measure,
            measure_evaluation_method_name=measure_evaluation_method_name,
            measure_options=measure_options,
        )
        self.model_assessor = model_assessor
        self.calibration_space = calibration_space
        self.maximize_objective = not measure.SMALLER_IS_BETTER
        self.dataset = None
        self.optimal_parameters = None
        self.optimal_criterion = None
        self.optimal_model = None
        self.scenario = None

    def execute(self, settings: BaseDriverSettings) -> None:
        """Calibrate the machine learning model from a driver.

        The driver can be either a DOE or an optimizer.

        Args:
            settings: The settings of the driver.
        """
        if DOELibraryFactory().is_available(settings._TARGET_CLASS_NAME):
            cls = DOEScenario
        else:
            cls = MDOScenario

        self.scenario = cls(
            [self.model_assessor],
            self.model_assessor.CRITERION,
            self.calibration_space,
            formulation_name="DisciplinaryOpt",
            maximize_objective=self.maximize_objective,
        )
        self.scenario.add_observable(self.model_assessor.LEARNING)
        self.scenario.execute(settings)
        self.dataset = self.scenario.to_dataset(opt_naming=False)
        self.optimal_parameters = self.scenario.optimization_result.x_opt_as_dict
        self.optimal_criterion = self.scenario.optimization_result.f_opt
        self.optimal_model = self.models[
            argmin(self.get_history(self.model_assessor.CRITERION))
        ]

    def get_history(
        self,
        name: str,
    ) -> ndarray | None:
        """Return the history of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The history of the variable if the dataset is not empty.
        """
        if self.dataset is None:
            return None

        if name == self.model_assessor.CRITERION and self.maximize_objective:
            return -self.dataset.get_view(variable_names="-" + name).to_numpy()
        return self.dataset.get_view(variable_names=name).to_numpy()

    @property
    def models(self) -> BaseMLModel:
        """The trained machine learning models."""
        return self.scenario.disciplines[0].models
