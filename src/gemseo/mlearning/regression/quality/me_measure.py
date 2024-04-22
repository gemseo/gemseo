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
r"""The maximum error to assess the quality of a regression algorithm.

The maximum error (ME) is defined by

$$\operatorname{ME}(\hat{y})=\max_{1\leq i \leq n}\|\hat{y}_i-y_i\|,$$

where $\hat{y}$ are the predictions and $y$ are the data points.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.mlearning.regression.quality.base_regressor_quality import (
    BaseRegressorQuality,
)

if TYPE_CHECKING:
    from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
    from gemseo.typing import NumberArray


class MEMeasure(BaseRegressorQuality):
    """The maximum error to assess the quality of a regressor."""

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
        if multioutput:
            return abs(outputs - predictions).max(0)
        return abs(outputs - predictions).max()
