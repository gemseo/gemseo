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
"""Settings of the thin plate spline (TPS) regressor."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from gemseo.mlearning.regression.algos.rbf_settings import RBF
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings


class TPSRegressor_Settings(RBFRegressor_Settings):  # noqa: N801
    """The settings of the thin plate spline (TPS) regressor."""

    function: Literal[RBF.THIN_PLATE] = Field(
        default=RBF.THIN_PLATE,
        description=r"The thin plate radial basis function for :math:`r^2\log(r)`.",
    )
