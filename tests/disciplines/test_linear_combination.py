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
"""Test linear combination discipline."""
from __future__ import annotations

import pytest
from gemseo.disciplines.linear_combination import LinearCombination
from numpy import array


@pytest.fixture
def linear_combination_for_tests():
    """Define a linear combination disciplines for pytests."""
    return LinearCombination(
        input_names=["alpha", "beta", "gamma"],
        output_name="delta",
        input_coefficients={"alpha": 1.0, "beta": -2.0, "gamma": 3.0},
        offset=-2.0,
    )


def test_linear_combination_execution(linear_combination_for_tests):
    """Test  linear combination discipline execution."""
    output_data = linear_combination_for_tests.execute(
        {"alpha": array([1.0]), "beta": array([1.0]), "gamma": array([1.0])}
    )
    assert all(output_data["delta"] == array([0.0]))


def test_linear_combination_execution2points(linear_combination_for_tests):
    """Test  linear combination discipline execution."""
    output_data = linear_combination_for_tests.execute(
        {
            "alpha": array([1.0, 0.0]),
            "beta": array([1.0, -1.0]),
            "gamma": array([1.0, 0.0]),
        }
    )
    assert all(output_data["delta"] == array([0.0, 0.0]))


def test_check_gradient(linear_combination_for_tests):
    """Test jacobian computation by finite differences."""
    linear_combination_for_tests.default_inputs = {
        "alpha": array([1.0]),
        "beta": array([1.0]),
        "gamma": array([1.0]),
    }
    assert linear_combination_for_tests.check_jacobian(
        linearization_mode="auto",
        threshold=1e-3,
        step=1e-4,
    )


def test_check_gradient2points(linear_combination_for_tests):
    """Test jacobian computation by finite differences."""
    linear_combination_for_tests.default_inputs = {
        "alpha": array([1.0, 0.0]),
        "beta": array([1.0, -1.0]),
        "gamma": array([1.0, 0.0]),
    }
    assert linear_combination_for_tests.check_jacobian(
        linearization_mode="auto",
        threshold=1e-3,
        step=1e-4,
    )
