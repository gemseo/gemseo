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

from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings


def test_newton_raphson_settings():
    """Verify that MDANewtonRaphson_Settings can handle MDA settings as a mapping."""
    store_residuals = (
        not MDANewtonRaphson_Settings().newton_linear_solver_settings.store_residuals
    )
    settings = MDANewtonRaphson_Settings(
        newton_linear_solver_settings={"store_residuals": store_residuals}
    )
    assert settings.newton_linear_solver_settings.store_residuals is store_residuals
