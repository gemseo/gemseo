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
"""The mean squared error to measure the quality of a regression algorithm.

The :mod:`~gemseo.mlearning.qual_measure.mse_measure` module
implements the concept of mean squared error measures
for machine learning algorithms.

This concept is implemented through the :class:`.MSEMeasure` class
and overloads the :meth:`!MLErrorMeasure._compute_measure` method.

The mean squared error (MSE) is defined by

.. math::

    \\operatorname{MSE}(\\hat{y})=\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2,

where :math:`\\hat{y}` are the predictions and :math:`y` are the data points.
"""
from __future__ import annotations

from numpy import ndarray
from sklearn.metrics import mean_squared_error

from gemseo.mlearning.qual_measure.error_measure import MLErrorMeasure
from gemseo.mlearning.qual_measure.quality_measure import MeasureType
from gemseo.mlearning.regression.regression import MLRegressionAlgo


class MSEMeasure(MLErrorMeasure):
    """The Mean Squared Error measure for machine learning."""

    def __init__(
        self,
        algo: MLRegressionAlgo,
        fit_transformers: bool = MLErrorMeasure._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for regression.
        """
        super().__init__(algo, fit_transformers=fit_transformers)

    def _compute_measure(
        self,
        outputs: ndarray,
        predictions: ndarray,
        multioutput: bool = True,
    ) -> MeasureType:
        return mean_squared_error(
            outputs,
            predictions,
            multioutput=self._GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT[multioutput],
        )
