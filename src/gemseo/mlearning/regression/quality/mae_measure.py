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
r"""The mean absolute error to assess the quality of a regressor.

The mean absolute error (MAE) is defined by

$$\operatorname{MAE}(\hat{y})=\frac{1}{n}\sum_{i=1}^n\|\hat{y}_i-y_i\|,$$

where $\hat{y}$ are the predictions and $y$ are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sklearn.metrics import mean_absolute_error

from gemseo.mlearning.regression.quality.base_regressor_quality import (
    BaseRegressorQuality,
)

if TYPE_CHECKING:
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import NumberArray


class MAEMeasure(BaseRegressorQuality):
    """The mean absolute error to assess the quality of a regressor."""

    def __init__(  # noqa: D107
        self,
        algo: BaseRegressor,
        fit_transformers: bool = False,
    ) -> None:
        super().__init__(algo, fit_transformers=fit_transformers)

    def _compute_measure(
        self,
        outputs: NumberArray,
        predictions: NumberArray,
        multioutput: bool = True,
    ) -> float | NumberArray:
        multioutput = "raw_values" if multioutput else "uniform_average"
        return mean_absolute_error(outputs, predictions, multioutput=multioutput)
