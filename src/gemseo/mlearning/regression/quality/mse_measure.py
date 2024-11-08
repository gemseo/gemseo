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
r"""The mean squared error to assess the quality of a regressor.

The mean squared error (MSE) is defined by

.. math::

    \\operatorname{MSE}(\\hat{y})=\\frac{1}{n}\\sum_{i=1}^n(\\hat{y}_i-y_i)^2,

where :math:`\\hat{y}` are the predictions and :math:`y` are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import mean_squared_error

from gemseo.mlearning.regression.quality.base_regressor_quality import (
    BaseRegressorQuality,
)

if TYPE_CHECKING:
    from gemseo.mlearning.core.quality.base_ml_algo_quality import MeasureType
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import RealArray


class MSEMeasure(BaseRegressorQuality):
    """The mean squared error to assess the quality of a regressor."""

    def __init__(
        self,
        algo: BaseRegressor,
        fit_transformers: bool = BaseRegressorQuality._FIT_TRANSFORMERS,
    ) -> None:
        """
        Args:
            algo: A machine learning algorithm for regression.
        """  # noqa: D205 D212
        super().__init__(algo, fit_transformers=fit_transformers)

    def _compute_measure(
        self,
        outputs: RealArray,
        predictions: RealArray,
        multioutput: bool = True,
    ) -> MeasureType:
        return mean_squared_error(
            outputs,
            predictions,
            multioutput=self._GEMSEO_MULTIOUTPUT_TO_SKLEARN_MULTIOUTPUT[multioutput],
        )
