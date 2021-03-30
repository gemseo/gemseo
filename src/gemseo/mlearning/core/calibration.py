# -*- coding: utf-8 -*-
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
"""
Calibration of a machine learning algorithm
===========================================

A machine learning algorithm depends on hyperparameters,
e.g. number of clusters for a clustering algorithm,
regularization constant for a regression model,
kernel for a Gaussian process regression, ...
Its quality of generalization depends on the values of these hyperparameters.
Thus, the hyperparameters minimizing the learning quality measure are rarely
those minimizing the generalization one. Classically, the generalization one
decreases before growing again as the model becomes more complex,
while the learning error keeps decreasing. This phenomenon is called the
curse of dimensionality.

In this module, the :class:`.MLAlgoCalibration` class aims to calibrate the
hyperparameters in order to minimize this generalization quality measure
over a calibration parameter space. This class relies on the
:class:`.MLAlgoAssessor` class which is a discipline
(:class:`.MDODiscipline`)
built from a machine learning algorithm (:class:`.MLAlgo`),
a dataset (:class:`.Dataset`), a quality measure (:class:`.MLQualityMeasure`)
and various options for data scaling, quality measure
and machine learning algorithm. The inputs of this discipline are
hyperparameters of the machine learning algorithm while the output is
the quality criterion.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import argmin, array

from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.mlearning.core.factory import MLAlgoFactory

standard_library.install_aliases()


class MLAlgoAssessor(MDODiscipline):
    """ Discipline assessing the quality of a machine learning algorithm """

    def __init__(
        self,
        algo,
        dataset,
        parameters,
        measure,
        measure_options=None,
        transformer=None,
        **algo_options
    ):
        """Constructor

        :param str algo: machine learning algorithm name.
        :param Dataset dataset: learning dataset.
        :param list(str) parameters: parameters.
        :param MLQualityMeasure measure: quality measure.
        :param dict measure_options: options of the quality measures.
        :param dict(str) transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :param algo_options: options of the machine learning algorithm.
        """
        super(MLAlgoAssessor, self).__init__()
        self.input_grammar.initialize_from_data_names(parameters)
        self.output_grammar.initialize_from_data_names(["criterion", "learning"])
        self.algo = algo
        self.measure = measure
        self.measure_options = measure_options or {}
        self.parameters = algo_options
        self.data = dataset
        self.transformer = transformer
        self.algos = []

    def _run(self):
        """ run method. """
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
        criterion = measure.evaluate(multioutput=False, **self.measure_options)
        self.store_local_data(criterion=array([criterion]), learning=array([learning]))
        self.algos.append(algo)


class MLAlgoCalibration(object):
    """ Calibration of a machine learning algorithm """

    CRITERION = "criterion"

    def __init__(
        self,
        algo,
        dataset,
        parameters,
        calibration_space,
        measure,
        measure_options=None,
        transformer=None,
        use_doe=True,
        **algo_options
    ):
        """Constructor

        :param str algo: machine learning algorithm name.
        :param Dataset dataset: learning dataset.
        :param list(str) parameters: parameters.
        :param DesignSpace calibration_space: calibration space.
        :param MLQualityMeasure measure: quality measure.
        :param dict measure_options: options of the quality measures.
        :param dict(str) transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :param bool use_doe: if True, use a DOEScenario to calibrate
            the ML algorithm. Otherwise, use a MDOScenario. Default: True.
        :param algo_options: options of the machine learning algorithm.
        """
        disc = MLAlgoAssessor(
            algo,
            dataset,
            parameters,
            measure,
            measure_options,
            transformer,
            **algo_options
        )
        disc.set_cache_policy(disc.MEMORY_FULL_CACHE)
        if use_doe:
            self.scenario = DOEScenario(
                [disc], "DisciplinaryOpt", self.CRITERION, calibration_space
            )
        else:
            self.scenario = MDOScenario(
                [disc], "DisciplinaryOpt", self.CRITERION, calibration_space
            )
        self.dataset = None
        self.optimal_parameters = None
        self.optimal_criterion = None
        self.optimal_algorithm = None

    def execute(self, input_data):
        """Execute the calibration from optimization or DOE data.

        :param dict input_data: optimization or DOE data.
        :return: optimal hyperparameters, optimal criterion.
        :rtype: dict, ndarray
        """
        self.scenario.disciplines[0].cache.clear()
        self.scenario.execute(input_data)
        x_opt = self.scenario.design_space.get_current_x_dict()
        f_opt = self.scenario.get_optimum().f_opt
        cache = self.scenario.disciplines[0].cache
        self.dataset = cache.export_to_dataset(by_group=False)
        algo_opt = self.algos[argmin(self.get_history(self.CRITERION))]
        self.optimal_parameters = x_opt
        self.optimal_criterion = f_opt
        self.optimal_algorithm = algo_opt

    def get_history(self, name):
        """Get history of a given variable.

        :param str name: variable name.
        :return: history of the variable.
        :rtype: ndarray
        """
        if self.dataset is not None:
            data = self.dataset.data[name]
        return data

    @property
    def algos(self):
        """ List of trained algorithms. """
        return self.scenario.disciplines[0].algos
