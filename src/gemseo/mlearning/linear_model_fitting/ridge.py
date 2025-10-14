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

"""Scikit-learn ridge algorithm."""

from __future__ import annotations

from sklearn.linear_model import Ridge as SKLearnRidge

from gemseo.mlearning.linear_model_fitting.base_sklearn_linear_model_fitter import (
    BaseSKLearnLinearModelFitter,
)
from gemseo.mlearning.linear_model_fitting.ridge_settings import Ridge_Settings


class Ridge(BaseSKLearnLinearModelFitter[SKLearnRidge, Ridge_Settings]):
    r"""Scikit-learn ridge algorithm.

    Given the linear model fitting problem
    presented in :mod:`this page <.linear_model_fitting>`,
    this algorithm solves a penalized least squares problem of the form:

    .. math::

       \min_w \|Xw-y\|_2^2 + \alpha \|w\|_2^2, \qquad \alpha \geq 0

    where :math:`\|w\|_2` is the :math:`\ell_2`-norm of the coefficients :math:`w`
    and :math:`\|Xw-y\|_2` is the :math:`\ell_2`-norm of the residual :math:`Xw-y`.
    """

    Settings = Ridge_Settings

    _FITTER_CLASS = SKLearnRidge
