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
from gemseo.problems.topo_opt.volume_fraction_disc import VolumeFraction
from numpy import array
from numpy import ones

THRESHOLD = 1e-10


@pytest.fixture(scope="module")
def volume_fraction() -> VolumeFraction:
    """A volume fraction discipline."""
    return VolumeFraction(n_x=10, n_y=10)


def test_run(volume_fraction):
    """"""
    output_data = volume_fraction.execute(
        {"rho": ones(volume_fraction.n_x * volume_fraction.n_y)}
    )
    assert output_data["volume fraction"] == array([1])


def test_jacobian(volume_fraction):
    """Check the analytic Jacobian by finite differences."""
    assert volume_fraction.check_jacobian(
        volume_fraction.get_input_data(), threshold=THRESHOLD, step=1e-5
    )
