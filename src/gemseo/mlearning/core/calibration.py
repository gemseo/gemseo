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
"""Calibration of a machine learning algorithm.

A machine learning algorithm depends on hyper-parameters,
e.g. the number of clusters for
a clustering algorithm, the regularization constant for a regression model, the kernel
for a Gaussian process regression, ... Its ability to generalize the information learned
during the training stage, and thus to avoid over-fitting, which is an over-reliance on
the learning data set, depends on the values of these hyper-parameters. Thus, the hyper-
parameters minimizing the learning quality measure are rarely those minimizing the
generalization one.
Classically, the generalization one decreases before growing again as
the model becomes more complex, while the learning error keeps decreasing. This
phenomenon is called the curse of dimensionality.

In this module, the :class:`.MLAlgoCalibration` class aims to calibrate the hyper-
parameters in order to minimize this measure of the generalization quality over a
calibration parameter space. This class relies on the :class:`.MLAlgoAssessor` class
which is a discipline (:class:`.Discipline`) built from a machine learning algorithm
(:class:`.BaseMLAlgo`), a dataset (:class:`.Dataset`), a quality measure
(:class:`.BaseMLAlgoQuality`) and various options for the data scaling, the quality
measure and the machine learning algorithm. The inputs of this discipline are hyper-
parameters of the machine learning algorithm while the output is the quality criterion.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from numpy import argmin
from numpy import array
from numpy import ndarray

from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.core.discipline import Discipline
from gemseo.mlearning.core.algos.factory import MLAlgoFactory
from gemseo.mlearning.core.quality.base_ml_algo_quality import BaseMLAlgoQuality
from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureOptionsType
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace
    from gemseo.datasets.dataset import Dataset
    from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
    from gemseo.mlearning.core.algos.ml_algo import MLAlgoSettingsType
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.scenarios.base_scenario import BaseScenario
    from gemseo.typing import StrKeyMapping


class MLAlgoAssessor(Discipline):
    """Discipline assessing the quality of a machine learning algorithm.

    This quality depends on the values of parameters to calibrate with the
    :class:`.MLAlgoCalibration`.
    """

    algo: str
    """The name of a machine learning algorithm."""

    measure: type[BaseMLAlgoQuality]
    """The measure to assess the machine learning algorithm."""

    measure_options: dict[str, int | Dataset]
    """The options of the quality measure."""

    parameters: dict[str, MLAlgoSettingsType]
    """The parameters of the machine learning algorithm."""

    dataset: Dataset
    """The training dataset."""

    transformer: TransformerType
    """The transformation strategy for data groups."""

    algos: list[BaseMLAlgo]
    """The instances of the machine learning algorithm (one per execution of the machine
    learning algorithm assessor)."""

    CRITERION = "criterion"
    LEARNING = "learning"
    MULTIOUTPUT = "multioutput"

    def __init__(
        self,
        algo: str,
        dataset: Dataset,
        parameters: Iterable[str],
        measure: type[BaseMLAlgoQuality],
        measure_evaluation_method_name: BaseMLAlgoQuality.EvaluationMethod = BaseMLAlgoQuality.EvaluationMethod.LEARN,  # noqa: E501
        measure_options: MeasureOptionsType = READ_ONLY_EMPTY_DICT,
        transformer: TransformerType = READ_ONLY_EMPTY_DICT,
        **algo_settings: MLAlgoSettingsType,
    ) -> None:
        """
        Args:
            algo: The name of a machine learning algorithm.
            dataset: A training dataset.
            parameters: The parameters of the machine learning algorithm to calibrate.
            measure: A measure to assess the machine learning algorithm.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            measure_options: The options of the quality measure.
                If "multioutput" is missing,
                it is added with False as value.
                If empty, do not use quality measure options.
            transformer: The strategies
                to transform the variables.
                The values are instances of :class:`.BaseTransformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.BaseTransformer` will be applied
                to all the variables of this group.
                If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
            **algo_settings: The settings of the machine learning algorithm.

        Raises:
            ValueError: If the measure option "multioutput" is True.
        """  # noqa: D205 D212
        super().__init__()
        self.io.input_grammar.update_from_names(parameters)
        self.io.output_grammar.update_from_names([self.CRITERION, self.LEARNING])
        self.algo = algo
        self.measure = measure
        self.measure_options = dict(measure_options)
        self.__measure_evaluation_method_name = measure_evaluation_method_name
        self.parameters = algo_settings
        self.data = dataset
        self.transformer = transformer
        self.algos = []
        if self.measure_options.get("multioutput", False):
            msg = "MLAlgoAssessor does not support multioutput."
            raise ValueError(msg)

        self.measure_options[self.MULTIOUTPUT] = False

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        """Run method.

        This method creates a new instance of the machine learning algorithm, from the
        hyper-parameters stored in the data attribute of the
        :class:`.MLAlgoAssessor`. It trains it on the training dataset and measures its
        quality with the :class:`.BaseMLAlgoQuality`.
        """
        inputs = self.io.get_input_data()
        for index in inputs:
            if len(inputs[index]) == 1:
                inputs[index] = inputs[index][0]
        self.parameters.update(inputs)
        algo = MLAlgoFactory().create(
            self.algo, data=self.data, transformer=self.transformer, **self.parameters
        )
        algo.learn()
        measure = self.measure(algo)
        compute_criterion = getattr(
            measure,
            measure.EvaluationFunctionName[self.__measure_evaluation_method_name],
        )
        self.algos.append(algo)
        return {
            "criterion": array([compute_criterion(**self.measure_options)]),
            "learning": array([measure.compute_learning_measure(multioutput=False)]),
        }


class MLAlgoCalibration:
    """Calibration of a machine learning algorithm."""

    algo_assessor: MLAlgoAssessor
    """The assessor for the machine learning algorithm."""

    calibration_space: DesignSpace
    """The space defining the calibration variables."""

    maximize_objective: bool
    """Whether to maximize the quality measure."""

    dataset: Dataset | None
    """The training dataset after execution."""

    optimal_parameters: dict[str, ndarray] | None
    """The optimal parameters for the machine learning algorithm after execution."""

    optimal_criterion: float | None
    """The optimal quality measure after execution."""

    optimal_algorithm: BaseMLAlgo | None
    """The optimal machine learning algorithm after execution."""

    scenario: BaseScenario | None
    """The scenario used to calibrate the machine learning algorithm after execution."""

    def __init__(
        self,
        algo: str,
        dataset: Dataset,
        parameters: Iterable[str],
        calibration_space: DesignSpace,
        measure: type[BaseMLAlgoQuality],
        measure_evaluation_method_name: str
        | BaseMLAlgoQuality.EvaluationMethod = BaseMLAlgoQuality.EvaluationMethod.LEARN,
        # noqa: E501
        measure_options: MeasureOptionsType = READ_ONLY_EMPTY_DICT,
        transformer: TransformerType = READ_ONLY_EMPTY_DICT,
        **algo_settings: MLAlgoSettingsType,
    ) -> None:
        """
        Args:
            algo: The name of a machine learning algorithm.
            dataset: A training dataset.
            parameters: The parameters of the machine learning algorithm
                to calibrate.
            calibration_space: The space defining the calibration variables.
            measure: A measure to assess the machine learning algorithm.
            measure_evaluation_method_name: The name of the method
                to evaluate the quality measure.
            measure_options: The options of the quality measure.
                If empty, do not use the quality measure options.
            transformer: The strategies
                to transform the variables.
                The values are instances of :class:`.BaseTransformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.BaseTransformer` will be applied
                to all the variables of this group.
                If :attr:`~.BaseMLAlgo.IDENTITY`, do not transform the variables.
            **algo_settings: The settings of the machine learning algorithm.
        """  # noqa: D205 D212
        disc = MLAlgoAssessor(
            algo,
            dataset,
            parameters,
            measure,
            measure_evaluation_method_name=measure_evaluation_method_name,
            measure_options=measure_options,
            transformer=transformer,
            **algo_settings,
        )
        self.algo_assessor = disc
        self.calibration_space = calibration_space
        self.maximize_objective = not measure.SMALLER_IS_BETTER
        self.dataset = None
        self.optimal_parameters = None
        self.optimal_criterion = None
        self.optimal_algorithm = None
        self.scenario = None

    def execute(
        self,
        algo_name: str,
        **algo_settings: Any,
    ) -> None:
        """Calibrate the machine learning algorithm from a driver.

        The driver can be either a DOE or an optimizer.

        Args:
            algo_name: The name of the algorithm.
            **algo_settings: The settings of the algorithm.
        """
        if DOELibraryFactory().is_available(algo_name):
            cls = DOEScenario
        else:
            cls = MDOScenario

        self.scenario = cls(
            [self.algo_assessor],
            self.algo_assessor.CRITERION,
            self.calibration_space,
            formulation_name="DisciplinaryOpt",
            maximize_objective=self.maximize_objective,
        )
        self.scenario.add_observable(self.algo_assessor.LEARNING)
        self.scenario.execute(algo_name=algo_name, **algo_settings)
        self.dataset = self.scenario.to_dataset(opt_naming=False)
        self.optimal_parameters = self.scenario.optimization_result.x_opt_as_dict
        self.optimal_criterion = self.scenario.optimization_result.f_opt
        self.optimal_algorithm = self.algos[
            argmin(self.get_history(self.algo_assessor.CRITERION))
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

        if name == self.algo_assessor.CRITERION and self.maximize_objective:
            return -self.dataset.get_view(variable_names="-" + name).to_numpy()
        return self.dataset.get_view(variable_names=name).to_numpy()

    @property
    def algos(self) -> BaseMLAlgo:
        """The trained machine learning algorithms."""
        return self.scenario.disciplines[0].algos
