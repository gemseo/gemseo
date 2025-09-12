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

"""Scikit-learn Orthogonal Matching Pursuit (OMP) algorithm with build-in cross-validation."""  # noqa: E501

from __future__ import annotations

from sklearn.linear_model import (
    OrthogonalMatchingPursuitCV as SKLearnOrthogonalMatchingPursuitCV,
)

from gemseo.mlearning.linear_model_fitting.base_sklearn_linear_model_fitter import (
    BaseSKLearnLinearModelFitter,
)
from gemseo.mlearning.linear_model_fitting.omp_cv_settings import (
    OrthogonalMatchingPursuitCV_Settings,
)
from gemseo.mlearning.linear_model_fitting.omp_settings import (
    OrthogonalMatchingPursuit_Settings,
)


class OrthogonalMatchingPursuitCV(
    BaseSKLearnLinearModelFitter[
        SKLearnOrthogonalMatchingPursuitCV, OrthogonalMatchingPursuit_Settings
    ]
):
    r"""Scikit-learn Orthogonal Matching Pursuit (OMP) algorithm with build-in cross-validation.

    Given the linear model fitting problem
    presented in :mod:`this page <.linear_model_fitting>`,
    this algorithm solves a penalized least squares problem of the form:

    .. math::

       \min_w \|Xw-y\|_2^2 \quad \text{s.t.} \quad \|w\|_0\leq \eta

    where :math:`\eta` is a specific number of non-zero components of :math:`w`.

    Alternatively:

    .. math::

       \min_w \|w\|_0 \quad \text{s.t.} \quad \|Xw-y\|_2^2\leq \tau

    where :math:`\tau` is a specific model error.
    """  # noqa: E501

    Settings = OrthogonalMatchingPursuitCV_Settings

    _FITTER_CLASS = SKLearnOrthogonalMatchingPursuitCV
