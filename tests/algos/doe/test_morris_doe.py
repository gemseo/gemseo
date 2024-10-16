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

from gemseo.algos.doe.morris_doe.morris_doe import MorrisDOE


def test_doe_morris():
    """Check Morris DOE algo."""
    morris = MorrisDOE()
    a = morris.compute_doe(3, unit_sampling=True)
    assert a.shape == (5 * (1 + 3), 3)


def test_morris_doe_algo():
    """Check Morris DOE algo with the options doe_algo_name and doe_algo_options."""
    morris = MorrisDOE()
    a = morris.compute_doe(
        3,
        unit_sampling=True,
        doe_algo_name="CustomDOE",
        doe_algo_settings={"samples": array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]])},
    )
    assert_almost_equal(
        a,
        array([
            [0.1, 0.5, 0.9],
            [0.15, 0.5, 0.9],
            [0.15, 0.55, 0.9],
            [0.15, 0.55, 0.95],
            [0.9, 0.5, 0.1],
            [0.95, 0.5, 0.1],
            [0.95, 0.55, 0.1],
            [0.95, 0.55, 0.15],
        ]),
    )


def test_morris_doe_step():
    """Check Morris DOE algo with the options step."""
    morris = MorrisDOE()
    a = morris.compute_doe(
        3,
        unit_sampling=True,
        doe_algo_name="CustomDOE",
        doe_algo_settings={"samples": array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]])},
        step=0.1,
    )
    assert_almost_equal(
        a,
        array([
            [0.1, 0.5, 0.9],
            [0.2, 0.5, 0.9],
            [0.2, 0.6, 0.9],
            [0.2, 0.6, 1.0],
            [0.9, 0.5, 0.1],
            [1.0, 0.5, 0.1],
            [1.0, 0.6, 0.1],
            [1.0, 0.6, 0.2],
        ]),
    )
