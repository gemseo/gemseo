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

from numpy import array
from numpy.testing import assert_almost_equal

from gemseo.algos.doe.oat_doe.oat_doe import OATDOE


def test_oat_doe():
    """Check OAT DOE algo."""
    oat = OATDOE()
    a = oat.compute_doe(3, initial_point=array([0.2, 0.8, 0.98]), unit_sampling=True)
    assert_almost_equal(
        a,
        array([
            [0.2, 0.8, 0.98],
            [0.25, 0.8, 0.98],
            [0.25, 0.85, 0.98],
            [0.25, 0.85, 0.93],
        ]),
    )


def test_oat_doe_step():
    """Check the step option of the OAT DOE algo."""
    oat = OATDOE()
    a = oat.compute_doe(
        3, initial_point=array([0.2, 0.8, 0.98]), unit_sampling=True, step=0.1
    )
    assert_almost_equal(
        a,
        array([[0.2, 0.8, 0.98], [0.3, 0.8, 0.98], [0.3, 0.9, 0.98], [0.3, 0.9, 0.88]]),
    )
