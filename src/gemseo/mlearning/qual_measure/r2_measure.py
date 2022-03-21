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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The R2 to measure the quality of a regression algorithm.

The :mod:`~gemseo.mlearning.qual_measure.r2_measure` module
implements the concept of R2 measures for machine learning algorithms.

This concept is implemented through the :class:`.R2Measure` class
and overloads the :meth:`!MLErrorMeasure._compute_measure` method.

The R2 is defined by

.. math::

    R_2(\\hat{y}) = 1 - \\frac{\\sum_i (\\hat{y}_i - y_i)^2}
                              {\\sum_i (y_i-\\bar{y})^2},

where
:math:`\\hat{y}` are the predictions,
:math:`y` are the data points and
:math:`\\bar{y}` is the mean of :math:`y`.
"""
from __future__ import division, unicode_literals

from copy import deepcopy
from typing import List, NoReturn, Optional, Union

from numpy import delete as npdelete
from numpy import mean, ndarray, repeat
from sklearn.metrics import mean_squared_error, r2_score

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure
from gemseo.mlearning.regression.regression import MLRegressionAlgo


class R2Measure(MLErrorMeasure):
    """The R2 measure for machine learning."""

    SMALLER_IS_BETTER = False

    def __init__(
        self,
        algo,  # type: MLRegressionAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm for regression.
        """
        super(R2Measure, self).__init__(algo)

    def _compute_measure(
        self,
        outputs,  # type: ndarray
        predictions,  # type: ndarray
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        multioutput = "raw_values" if multioutput else "uniform_average"
        return r2_score(outputs, predictions, multioutput=multioutput)

    def evaluate_kfolds(
        self,
        n_folds=5,  # type: int
        samples=None,  # type: Optional[List[int]]
        multioutput=True,  # type: bool
        randomize=False,  # type:bool
        seed=None,  # type: Optional[int]
    ):  # type: (...) -> Union[float,ndarray]
        folds, samples = self._compute_folds(samples, n_folds, randomize, seed)

        input_data = self.algo.learning_set.get_data_by_names(
            self.algo.input_names, False
        )
        output_data = self.algo.learning_set.get_data_by_names(
            self.algo.output_names, False
        )

        multiout = "raw_values" if multioutput else "uniform_average"

        algo = deepcopy(self.algo)

        sse = 0
        ymean = repeat(mean(output_data, axis=0)[None, :], len(output_data), axis=0)
        var = mean_squared_error(output_data, ymean, multioutput=multiout)
        for n_fold in range(n_folds):
            fold = folds[n_fold]
            algo.learn(samples=npdelete(samples, fold))
            mse = mean_squared_error(
                output_data[fold], algo.predict(input_data[fold]), multioutput=multiout
            )
            sse += mse * len(fold)

        return 1 - sse / var / len(ymean)

    def evaluate_bootstrap(
        self,
        n_replicates=100,  # type: int
        samples=None,  # type: Optional[List[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> NoReturn
        raise NotImplementedError
