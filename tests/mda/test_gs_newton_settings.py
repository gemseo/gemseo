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

from gemseo.mda.gauss_seidel_newton_raphson_settings import (
    MDAGaussSeidelNewtonRaphson_Settings,
)


def test_gauss_seidel_newton_raphson_settings():
    """Verify that MDAGaussSeidelNewtonRaphson_Settings can handle MDA settings as mappings."""  # noqa: E501
    mda_gauss_seidel_newton_raphson = MDAGaussSeidelNewtonRaphson_Settings()

    log_convergence = (
        not mda_gauss_seidel_newton_raphson.newton_raphson_settings.log_convergence
    )
    settings = MDAGaussSeidelNewtonRaphson_Settings(
        newton_raphson_settings={"log_convergence": log_convergence}
    )
    assert settings.newton_raphson_settings.log_convergence is log_convergence

    log_convergence = (
        not mda_gauss_seidel_newton_raphson.gauss_seidel_settings.log_convergence
    )
    settings = MDAGaussSeidelNewtonRaphson_Settings(
        gauss_seidel_settings={"log_convergence": log_convergence}
    )
    assert settings.gauss_seidel_settings.log_convergence is log_convergence
