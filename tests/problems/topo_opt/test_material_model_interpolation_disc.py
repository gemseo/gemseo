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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Simone Coniglio
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.topo_opt.material_model_interpolation_disc import (
    MaterialModelInterpolation,
)
from numpy import ones

THRESHOLD = 1e-7


@pytest.fixture(scope="module")
def material_model():
    """A material model interpolation discipline."""
    return MaterialModelInterpolation(
        e0=1, penalty=3.0, n_x=4, n_y=4, empty_elements=[], full_elements=[]
    )


def test_run_e(material_model):
    """"""
    output_data = material_model.execute(
        input_data={"xPhys": ones(material_model.N_elements)}
    )
    assert all(1.0 - THRESHOLD <= item <= 1.0 + THRESHOLD for item in output_data["E"])


def test_run_rho(material_model):
    """"""
    output_data = material_model.execute(
        input_data={"xPhys": ones(material_model.N_elements)}
    )
    assert all(
        1.0 - THRESHOLD <= item <= 1.0 + THRESHOLD for item in output_data["rho"]
    )


def test_jacobian(material_model):
    """Check the analytic Jacobian by finite differences."""
    assert material_model.check_jacobian(
        material_model.get_input_data(), threshold=THRESHOLD, auto_set_step=True
    )
