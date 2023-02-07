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
from gemseo.problems.topo_opt.density_filter_disc import DensityFilter
from numpy import ones

THRESHOLD = 1e-10


@pytest.fixture(scope="module")
def density_filter() -> DensityFilter:
    """The density filter."""
    return DensityFilter(n_x=4, n_y=4)


def test_run(density_filter):
    """"""
    output_data = density_filter.execute(
        input_data={"x": ones(density_filter.n_x * density_filter.n_y)}
    )
    assert all(
        1.0 - THRESHOLD <= item <= 1.0 + THRESHOLD for item in output_data["xPhys"]
    )


def test_jacobian(density_filter):
    """Check the analytic Jacobian by finite differences."""
    input_data = density_filter.get_input_data()
    assert density_filter.check_jacobian(
        input_data, threshold=THRESHOLD, step=1e-5, auto_set_step=True
    )
