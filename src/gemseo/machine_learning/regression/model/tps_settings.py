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

from typing import TYPE_CHECKING

from gemseo.machine_learning.regression.model.rbf_settings import RBF
from gemseo.machine_learning.regression.model.rbf_settings import (
    BaseRBFRegressorSettings,
)

if TYPE_CHECKING:
    from pydantic import PositiveFloat


class TPSRegressor_Settings(BaseRBFRegressorSettings):  # noqa: N801
    """The settings of the thin plate spline (TPS) regressor."""

    @property
    def kernel_(self) -> RBF:  # ruff: ignore[undocumented-public-method]
        return RBF.THIN_PLATE_SPLINE

    @property
    def epsilon_(self) -> PositiveFloat | None:  # ruff: ignore[undocumented-public-method]
        return 1.0
