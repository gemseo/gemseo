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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy import isclose

from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
from gemseo.mda.quasi_newton_settings import MDAQuasiNewton_Settings

from .utils import generate_parallel_doe


@pytest.mark.parametrize(
    "inner_mda_settings",
    [MDAQuasiNewton_Settings(), MDANewtonRaphson_Settings(), MDAGaussSeidel_Settings()],
)
def test_parallel_doe(inner_mda_settings) -> None:
    """Test the execution of Newton methods in parallel.

    Args:
        inner_mda_settings: The settings of the inner MDA.
    """
    obj = generate_parallel_doe(
        "MDAChain", main_mda_settings={"inner_mda_settings": inner_mda_settings}
    )
    assert isclose(array([obj]), array([-608.17]), atol=1e-3, rtol=1e-3)
