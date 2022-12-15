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
from gemseo.problems.topo_opt.fea_disc import FininiteElementAnalysis
from numpy import arange
from numpy import array
from numpy import ones
from numpy import tile


@pytest.fixture(scope="module")
def finite_element_analysis():
    """A finite element analysis."""
    nx = 4
    ny = 4
    nu = 0.3
    f_node = [(nx + 1) * (ny + 1) - 1]
    f_direction = [1]
    f_amplitude = [-1]
    fixed_node = tile(arange(ny + 1), 2)
    fixed_dir = array([0] * (ny + 1) + [1] * (ny + 1))
    return FininiteElementAnalysis(
        nu=nu,
        n_x=nx,
        n_y=ny,
        f_node=f_node,
        f_direction=f_direction,
        f_amplitude=f_amplitude,
        fixed_nodes=fixed_node,
        fixed_dir=fixed_dir,
    )


@pytest.fixture(scope="module")
def default_finite_element_analysis():
    """A finite element analysis."""

    return FininiteElementAnalysis()


def test_run(finite_element_analysis):
    """"""
    output_data = finite_element_analysis.execute(
        input_data={"E": ones(finite_element_analysis.N_elements)}
    )
    assert output_data["compliance"] > 0.0


def test_run_default(default_finite_element_analysis):
    """"""
    output_data = default_finite_element_analysis.execute(
        input_data={"E": ones(default_finite_element_analysis.N_elements)}
    )
    assert output_data["compliance"] > 0.0


def test_jacobian(finite_element_analysis):
    """Check the analytic Jacobian by finite differences."""
    indata = finite_element_analysis.get_input_data()
    assert finite_element_analysis.check_jacobian(
        indata, threshold=1e-5, auto_set_step=True
    )
