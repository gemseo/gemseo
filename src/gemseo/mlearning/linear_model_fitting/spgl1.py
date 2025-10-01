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

"""SPGL1 (Spectral Projected Gradient for L1 minimization) algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import vstack
from spgl1 import spgl1

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import (
    BaseLinearModelFitter,
)
from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import (
    _WrappedFittingFunction,
)
from gemseo.mlearning.linear_model_fitting.spgl1_settings import SPGL1_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class _SGPL1FittingFunction(_WrappedFittingFunction):
    """Interface to the SPGL1 fitting function."""

    def fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        return vstack([spgl1(input_data, y, **self._kwargs)[0] for y in output_data.T])


class SPGL1(BaseLinearModelFitter[_SGPL1FittingFunction, SPGL1_Settings]):
    r"""SPGL1 (Spectral Projected Gradient for L1 minimization) algorithm.

    Given the linear model fitting problem
    presented in :mod:`this page <.linear_model_fitting>`,
    this algorithm solves a penalized least squares problem of the form:

    1. Basis pursuit denoise (BPDN) when ``sigma`` is a positive number:

    .. math::

       \min_w \|w\|_1 \quad \text{s.t.} \quad \|Xw-y\|_2 \leq \sigma , \qquad \sigma > 0

    2. Basis pursuit (BP) when ``tau`` and ``sigma`` are ``0``:

    .. math::

       \min_w \|w\|_1 \quad \text{s.t.} \quad Xw=y

    3. Lasso when ``tau`` is a positive number:

    .. math::

       \min_w \|Xw-y\|_2 \quad \text{s.t.} \quad \|w\|_1 \leq \tau , \qquad \tau > 0

    where :math:`\|w\|_1` is the :math:`\ell_1`-norm of the coefficients :math:`w`
    and :math:`\|Xw-y\|_2` is the :math:`\ell_2`-norm of the residual :math:`Xw-y`.
    """  # noqa: E501

    Settings = SPGL1_Settings

    _FITTER_CLASS = _SGPL1FittingFunction

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        input_data, output_data = self._stack_data(input_data, output_data, extra_data)
        return self._fitter.fit(input_data, output_data)
