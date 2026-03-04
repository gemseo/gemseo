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

import re

import pytest
from numpy import array
from numpy.testing import assert_almost_equal

from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.morris_doe.morris_doe import MorrisDOE
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings


def test_settings_error():
    """Check error when n_samples > 0 but not in doe_algo_settings."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "When n_samples > 0, doe_algo_settings must have an n_samples field."
        ),
    ):
        MorrisDOE_Settings(
            n_samples=3,
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]])
            ),
        )


def test_doe_morris():
    """Check Morris DOE algo."""
    morris = MorrisDOE()
    a = morris.sample_unit_hypercube(3)
    assert a.shape == (1 * (1 + 3), 3)


def test_doe_morris_doe_settings():
    """Check Morris DOE algo with LHS settings."""
    morris = MorrisDOE()
    from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings

    a = morris.sample_unit_hypercube(
        3, MorrisDOE_Settings(doe_algo_settings=LHS_Settings(n_samples=5))
    )
    assert a.shape == (5 * (1 + 3), 3)


def test_morris_doe_custom_doe():
    """Check Morris DOE algo with Custom DOE settings."""
    morris = MorrisDOE()
    a = morris.sample_unit_hypercube(
        3,
        MorrisDOE_Settings(
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]])
            )
        ),
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
    a = morris.sample_unit_hypercube(
        3,
        MorrisDOE_Settings(
            doe_algo_settings=CustomDOE_Settings(
                samples=array([[0.1, 0.5, 0.9], [0.9, 0.5, 0.1]]),
            ),
            step=0.1,
        ),
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
