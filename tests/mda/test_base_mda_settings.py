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
from __future__ import annotations

from gemseo.algos.linear_solvers.base_linear_solver_settings import (
    BaseLinearSolverSettings,
)
from gemseo.mda.base_mda_settings import BaseMDASettings


def test_base_mda_settings():
    """Verify that BaseMDA_Settings can handle linear solver settings as a mapping."""
    store_residuals = not BaseLinearSolverSettings().store_residuals
    settings = BaseMDASettings(
        linear_solver_settings={"store_residuals": store_residuals}
    )
    assert settings.linear_solver_settings.store_residuals is store_residuals
