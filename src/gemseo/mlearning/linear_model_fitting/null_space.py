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

"""Null space algorithm."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import vstack
from scipy.linalg import lstsq
from scipy.linalg import qr

from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import (
    BaseLinearModelFitter,
)
from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter import (
    _WrappedFittingFunction,
)
from gemseo.mlearning.linear_model_fitting.null_space_settings import NullSpace_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class _NullSpaceFittingFunction(_WrappedFittingFunction):
    """Null space method."""

    def fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        """
        Raises:
            ValueError: When the extra data are missing.
        """  # noqa: D205, D212
        if not extra_data:
            msg = "The null space algorithm requires extra data."
            raise ValueError(msg)

        # We use the notations of Ghisu et al (2021).
        A = input_data  # noqa: N806
        b = output_data
        C = vstack([x[0] for x in extra_data])  # noqa: N806
        d = vstack([x[1] for x in extra_data])
        t, n = A.shape

        Q, R = qr(A.T)  # noqa: N806
        Q1 = Q[:, 0:t]  # noqa: N806
        Q2 = Q[:, t:n]  # noqa: N806
        w1 = lstsq(R.T[0:t, 0:t], b)[0]
        G = C @ Q2  # noqa: N806
        h = d - C @ Q1 @ w1
        w2 = lstsq(G, h)[0]
        return (Q1 @ w1 + Q2 @ w2).T


class NullSpace(BaseLinearModelFitter[_NullSpaceFittingFunction, NullSpace_Settings]):
    r"""The null space method.

    Given the linear model fitting problem
    presented in :mod:`this page <.linear_model_fitting>`,
    this algorithm uses the null space method
    to solve a penalized least squares problem of the form:

    .. math::

       min_w \|\tilde{Y} - \tilde{X}w\|_2 \quad s.t. \quad Y=Xw

    where :math:`\tilde{X}` and :math:`\tilde{Y}` contained additional data
    such that :math:`\text{rank}\left(\begin{matrix}X\\\tilde{X}\end{matrix}\right)`
    equals the number of features :math:`d`.

    This method was applied by Ghisu *et al* (2021)
    to fit a polynomial chaos expansion from input, output and Jacobian data.

    Tiziano Ghisu, Diego I. Lopez, Pranay Seshadri and Shahrokh Shahpar,
    *Gradient-enhanced Least-square Polynomial Chaos Expansions
    for Uncertainty Quantification and Robust Optimization*,
    AIAA 2021-3073. AIAA AVIATION 2021 FORUM. August 2021.
    """

    Settings = NullSpace_Settings

    _FITTER_CLASS = _NullSpaceFittingFunction

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        return self._fitter.fit(input_data, output_data, *extra_data)
