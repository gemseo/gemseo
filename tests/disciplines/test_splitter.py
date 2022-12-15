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

import pytest
from gemseo.disciplines.splitter import Splitter
from numpy import array


@pytest.fixture
def splitting_discipline_for_test():
    """Define a Slicer discipline for test."""
    return Splitter("E", {"Ep": [0, 1], "Es": [2, 3], "Er": 4})


def test_splitting_discipline_execution(splitting_discipline_for_test):
    """Test the Splitter execution."""
    output_data = splitting_discipline_for_test.execute(
        {"E": array([1.0, 2.0, 3.0, 4.0, 5.0])}
    )
    assert (output_data["Ep"] == array([1.0, 2.0])).all
    assert (output_data["Es"] == array([3.0, 4.0])).all
    assert (output_data["Er"] == array([5.0])).all


def test_check_gradient(splitting_discipline_for_test):
    """Test Splitter jacobian computation by finite differences."""
    splitting_discipline_for_test.default_inputs = {
        "E": array([1.0, 2.0, 3.0, 4.0, 5.0])
    }
    assert splitting_discipline_for_test.check_jacobian(
        input_data={"E": array([1.0, 2.0, 3.0, 4.0, 5.0])}, threshold=1e-3, step=1e-4
    )
