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
e.g. the number of clusters for a clustering algorithm,
the regularization constant for a regression model,
the kernel for a Gaussian process regression, ...
Its ability to generalize the information learned during the training stage,
and thus to avoid over-fitting,
which is an over-reliance on the learning data set,
depends on the values of these hyper-parameters.
Thus,
the hyper-parameters minimizing the learning quality measure are rarely
those minimizing the generalization one.
Classically,
the generalization one decreases before growing again as the model becomes more complex,
while the learning error keeps decreasing.
This phenomenon is called the curse of dimensionality.

In this module,
the :class:`.MLAlgoCalibration` class aims to calibrate the hyper-parameters
in order to minimize this measure of the generalization quality
over a calibration parameter space.
This class relies on the :class:`.MLAlgoAssessor` class
which is a discipline (:class:`.MDODiscipline`)
built from a machine learning algorithm (:class:`.MLAlgo`),
a dataset (:class:`.Dataset`),
a quality measure (:class:`.MLQualityMeasure`)
and various options for the data scaling,
the quality measure
and the machine learning algorithm.
The inputs of this discipline are hyper-parameters of the machine learning algorithm
while the output is the quality criterion.
"""
from __future__ import annotations

from typing import Dict
from typing import Iterable
from typing import Union

from numpy import argmin
from numpy import array
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.core.scenario import Scenario
from gemseo.core.scenario import ScenarioInputDataType
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure

MeasureOptionsType = Dict[str, Union[bool, int, Dataset]]


class MLAlgoAssessor(MDODiscipline):
    """Discipline assessing the quality of a machine learning algorithm.

    This quality depends on the values of parameters to calibrate with the
    :class:`.MLAlgoCalibration`.
    """

    algo: str
    """The name of a machine learning algorithm."""

    measure: MLQualityMeasure
    """The measure to assess the machine learning algorithm."""

    measure_options: dict[str, int | Dataset]
    """The options of the quality measure."""

    parameters: list[str]
    """The parameters of the machine learning algorithm."""

    dataset: Dataset
    """The learning dataset."""

    transformer: TransformerType
    """The transformation strategy for data groups."""

    algos: list[MLAlgo]
    """The instances of the machine learning algorithm
    (one per execution of the machine learning algorithm assessor)."""

    CRITERION = "criterion"
    LEARNING = "learning"
    MULTIOUTPUT = "multioutput"

    def __init__(
        self,
        algo: str,
        dataset: Dataset,
        parameters: Iterable[str],
        measure: type[MLQualityMeasure],
        measure_options: MeasureOptionsType | None = None,
        transformer: TransformerType = MLAlgo.IDENTITY,
        **algo_options: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            algo: The name of a machine learning algorithm.
            dataset: A learning dataset.
            parameters: The parameters of the machine learning algorithm to calibrate.
            measure: A measure to assess the machine learning algorithm.
            measure_options: The options of the quality measure.
                If "multioutput" is missing,
                it is added with False as value.
                If ``None``, do not use quality measure options.
            transformer: The strategies
                to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.

            **algo_options: The options of the machine learning algorithm.

        Raises:
            ValueError: If the measure option "multioutput" is True.
        """
        super().__init__()
        self.input_grammar.update(parameters)
        self.output_grammar.update([self.CRITERION, self.LEARNING])
        self.algo = algo
        self.measure = measure
        self.measure_options = measure_options or {}
        self.parameters = algo_options
        self.data = dataset
        self.transformer = transformer
        self.algos = []

        if self.measure_options.get("multioutput", False):
            raise ValueError("MLAlgoAssessor does not support multioutput.")
        self.measure_options[self.MULTIOUTPUT] = False

    def _run(self) -> None:
        """Run method.

        This method creates a new instance of the machine learning algorithm, from the
        hyper-parameters stored in the local_data attribute of the
        :class:`.MLAlgoAssessor`. It trains it on the learning dataset and measures its
        quality with the :class:`.MLQualityMeasure`.
        """
        inputs = self.get_input_data()
        for index in inputs:
            if len(inputs[index]) == 1:
                inputs[index] = inputs[index][0]
        self.parameters.update(inputs)
        factory = MLAlgoFactory()
        algo = factory.create(
            self.algo, data=self.data, transformer=self.transformer, **self.parameters
        )
        algo.learn()
        measure = self.measure(algo)
        learning = measure.evaluate(multioutput=False)
        criterion = measure.evaluate(**self.measure_options)
        self.store_local_data(criterion=array([criterion]), learning=array([learning]))
        self.algos.append(algo)


class MLAlgoCalibration:
    """Calibration of a machine learning algorithm."""

    algo_assessor: MLAlgoAssessor
    """The assessor for the machine learning algorithm."""

    calibration_space: DesignSpace
    """The space defining the calibration variables."""

    maximize_objective: bool
    """Whether to maximize the quality measure."""

    dataset: Dataset
    """The learning dataset."""

    optimal_parameters: dict[str, ndarray]
    """The optimal parameters for the machine learning algorithm."""

    optimal_criterion: float
    """The optimal quality measure."""

    optimal_algorithm: MLAlgo
    """The optimal machine learning algorithm."""

    scenario: Scenario
    """The scenario used to calibrate the machine learning algorithm."""

    def __init__(
        self,
        algo: str,
        dataset: Dataset,
        parameters: Iterable[str],
        calibration_space: DesignSpace,
        measure: MLQualityMeasure,
        measure_options: MeasureOptionsType | None = None,
        transformer: TransformerType = MLAlgo.IDENTITY,
        **algo_options: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            algo: The name of a machine learning algorithm.
            dataset: A learning dataset.
            parameters: The parameters of the machine learning algorithm
                to calibrate.
            calibration_space: The space defining the calibration variables.
            measure: A measure to assess the machine learning algorithm.
            measure_options: The options of the quality measure.
                If ``None``, do not use the quality measure options.
            transformer: The strategies
                to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. ``"inputs"`` or ``"outputs"``
                in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If :attr:`~.MLAlgo.IDENTITY`, do not transform the variables.
            **algo_options: The options of the machine learning algorithm.
        """
        disc = MLAlgoAssessor(
            algo,
            dataset,
            parameters,
            measure,
            measure_options,
            transformer,
            **algo_options,
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
        input_data: ScenarioInputDataType,
    ) -> None:
        """Calibrate the machine learning algorithm from a driver.

        The driver can be either a DOE or an optimizer.

        Args:
            input_data: The driver properties.
        """
        doe_factory = DOEFactory()

        if doe_factory.is_available(input_data["algo"]):
            self.scenario = DOEScenario(
                [self.algo_assessor],
                "DisciplinaryOpt",
                self.algo_assessor.CRITERION,
                self.calibration_space,
                maximize_objective=self.maximize_objective,
            )
        else:
            self.scenario = MDOScenario(
                [self.algo_assessor],
                "DisciplinaryOpt",
                self.algo_assessor.CRITERION,
                self.calibration_space,
                maximize_objective=self.maximize_objective,
            )
        self.scenario.add_observable(self.algo_assessor.LEARNING)
        self.scenario.execute(input_data)
        x_opt = self.scenario.design_space.get_current_value(as_dict=True)
        f_opt = self.scenario.get_optimum().f_opt
        self.dataset = self.scenario.export_to_dataset(by_group=False, opt_naming=False)
        algo_opt = self.algos[argmin(self.get_history(self.algo_assessor.CRITERION))]
        self.optimal_parameters = x_opt
        self.optimal_criterion = f_opt
        self.optimal_algorithm = algo_opt

    def get_history(
        self,
        name: str,
    ) -> ndarray:
        """Return the history of a variable.

        Args:
            name: The name of the variable.

        Returns:
            The history of the variable.
        """
        if self.dataset is not None:
            if name == self.algo_assessor.CRITERION and self.maximize_objective:
                return -self.dataset.data["-" + name]
            else:
                return self.dataset.data[name]

    @property
    def algos(self) -> MLAlgo:
        """The trained machine learning algorithms."""
        return self.scenario.disciplines[0].algos
