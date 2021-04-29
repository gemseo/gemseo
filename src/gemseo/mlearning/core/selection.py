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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Machine learning algorithm selection
====================================

Machine learning is used to find relations or underlying structures in data.
There is however no algorithm that is universally better than the others for
an arbitrary problem (see "No free lunch theorem"). Provided a quality measure,
one can thus compare the performances of different machine learning algorithms.

This process can be easily performed using the class :class:`.MLAlgoSelection`.

A machine learning algorithm is built using a set of (hyper)parameters, before
learning takes place. In order to choose the best hyperparameters, a simple
grid search over different values may be sufficient. The
:class:`.MLAlgoSelection` does this. It can also perform a more advanced form
of optimization than a simple grid search over predefined values, using the
class :class:`.MLAlgoCalibration`.

.. seealso::

   :mod:`~gemseo.mlearning.core.ml_algo`
   :mod:`~gemseo.mlearning.core.calibration`

"""
from __future__ import absolute_import, division, unicode_literals

from itertools import product

from gemseo.mlearning.core.calibration import MLAlgoCalibration
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasure


class MLAlgoSelection(object):
    """Machine learning algorithm selector."""

    def __init__(
        self,
        dataset,
        measure,
        eval_method=MLQualityMeasure.LEARN,
        samples=None,
        **measure_options
    ):
        """Constructor.

        :param str measure: MLQualityMeasure.
        :param list(int) samples: Indices of samples to consider. Other indices
            are neither used for training nor for testing. If None, use all
            samples. Default: None.
        :param str eval_method: Method for MLQalityMeasure.evaluate() method.
        :param dict measure_options: options for MLQalityMeasure.evaluate()
            method. The option multioutput will be set to False.
        """
        self.dataset = dataset
        self.measure = measure
        self.measure_options = dict(
            samples=samples, method=eval_method, **measure_options
        )
        self.factory = MLAlgoFactory()

        self.candidates = []

        if "multioutput" in measure_options and measure_options["multioutput"]:
            raise ValueError(
                "MLAlgoSelection does not support multioutput. "
                "The measure shall return one value."
            )
        self.measure_options["multioutput"] = False

    def add_candidate(self, name, calib_space=None, calib_algo=None, **option_lists):
        """Add machine learning algorithm candidate.

        :param str name: Class name of MLAlgo.
        :param DesignSpace calib_space: Design space for parameters to be
            calibrated with an MLAlgoCalibration. If None, do not perform
            calibration. Default: None.
        :param dict calib_algo: Dictionary containing optimization algorithm
            and parameters (example: {"algo": "fullfact", "n_samples": 10}).
            If None, do not perform calibration. Default: None.
        :param dict option_lists: Parameters for the MLAlgo candidate. Each
            parameter has to be enclosed within a list. The list may contain
            different values to try out for the given parameter, or only one.
        """
        keys, values = option_lists.keys(), option_lists.values()

        # Set initial quality to the worst possible value
        if self.measure.SMALLER_IS_BETTER:
            quality = float("inf")
        else:
            quality = -float("inf")

        for prodvalues in product(*values):
            params = dict(zip(keys, prodvalues))
            if not calib_space:
                algo_new = self.factory.create(name, data=self.dataset, **params)
                quality_new = self.measure(algo_new).evaluate(**self.measure_options)
            else:
                calib = MLAlgoCalibration(
                    name,
                    self.dataset,
                    calib_space.variables_names,
                    calib_space,
                    self.measure,
                    self.measure_options,
                    **params
                )
                calib.execute(calib_algo)
                algo_new = calib.optimal_algorithm
                quality_new = calib.optimal_criterion

            if self.measure.is_better(quality_new, quality):
                algo = algo_new
                quality = quality_new

        # Learn on the entire dataset if a cross-validation scheme was used
        algo.learn(self.measure_options["samples"])

        self.candidates.append((algo, quality))

    def select(self, return_quality=False):
        """Select best model.

        The model is chosen through a grid search over candidates and their
        options, as well as an eventual optimization over the parameters in
        the calibration space.

        :param bool return_quality: indicator for whether to return the
            quality of the best model.
        :return: best model and its quality if indicated.
        :rtype: MLAlgo or tuple(MLAlgo, float)
        """
        candidate = self.candidates[0]
        for new_candidate in self.candidates[1:]:
            if self.measure.is_better(new_candidate[1], candidate[1]):
                candidate = new_candidate
        if not return_quality:
            candidate = candidate[0]
        return candidate
