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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The root mean squared error to measure the quality of a regression algorithm.

The :mod:`~gemseo.mlearning.qual_measure.mse_measure` module
implements the concept of root mean squared error measures
for machine learning algorithms.

This concept is implemented through the
:class:`.RMSEMeasure` class and
overloads the :meth:`!MSEMeasure.evaluate_*` methods.

The root mean squared error (RMSE) is defined by

.. math::

    \\operatorname{RMSE}(\\hat{y})=\\sqrt{\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2},

where
:math:`\\hat{y}` are the predictions and
:math:`y` are the data points.
"""
from __future__ import division, unicode_literals

from typing import Optional, Sequence, Union

from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.qual_measure.mse_measure import MSEMeasure
from gemseo.mlearning.regression.regression import MLRegressionAlgo


class RMSEMeasure(MSEMeasure):
    """The root mean Squared Error measure for machine learning."""

    def __init__(
        self,
        algo,  # type: MLRegressionAlgo
    ):  # type: (...) -> None
        """
        Args:
            algo: A machine learning algorithm for regression.
        """
        super(RMSEMeasure, self).__init__(algo)

    def evaluate_learn(
        self,
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        mse = super(RMSEMeasure, self).evaluate_learn(
            samples=samples, multioutput=multioutput
        )
        return mse ** 0.5

    def evaluate_test(
        self,
        test_data,  # type:Dataset
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        mse = super(RMSEMeasure, self).evaluate_test(
            test_data, samples=samples, multioutput=multioutput
        )
        return mse ** 0.5

    def evaluate_kfolds(
        self,
        n_folds=5,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
        randomize=False,  # type:bool
    ):  # type: (...) -> Union[float,ndarray]
        mse = super(RMSEMeasure, self).evaluate_kfolds(
            n_folds=n_folds,
            samples=samples,
            multioutput=multioutput,
            randomize=randomize,
        )
        return mse ** 0.5

    def evaluate_bootstrap(
        self,
        n_replicates=100,  # type: int
        samples=None,  # type: Optional[Sequence[int]]
        multioutput=True,  # type: bool
    ):  # type: (...) -> Union[float,ndarray]
        mse = super(RMSEMeasure, self).evaluate_bootstrap(
            n_replicates=n_replicates, samples=samples, multioutput=multioutput
        )
        return mse ** 0.5
