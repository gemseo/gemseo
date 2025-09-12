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

"""Base class for scikit-learn linear model fitting algorithms."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import TypeVar

from numpy import newaxis
from sklearn.linear_model._base import LinearModel

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import (
    BaseLinearModelFitter,
)
from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import SettingsType

if TYPE_CHECKING:
    from gemseo.typing import RealArray

SKLearnLinearModelType = TypeVar("SKLearnLinearModel", bound=LinearModel)


class BaseSKLearnLinearModelFitter(
    BaseLinearModelFitter[SKLearnLinearModelType, SettingsType]
):
    """Base class for scikit-learn linear model fitting algorithms."""

    _PRIORITARY_FITTER_KWARGS = MappingProxyType({"fit_intercept": False})

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:  # noqa: D102
        input_data, output_data = self._stack_data(input_data, output_data, extra_data)
        self._fitter.fit(input_data, output_data)
        coefficients = self._fitter.coef_
        if coefficients.ndim > 1:
            return coefficients

        return coefficients[newaxis, :]
