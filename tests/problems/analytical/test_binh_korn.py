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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Test of the Binh-Korn problem."""
from __future__ import annotations

import pytest
from gemseo.problems.analytical.binh_korn import BinhKorn
from numpy import array


@pytest.fixture()
def binh_korn() -> BinhKorn:
    """Fixture for the Binh-Korn optimization problem.

    Returns:
         A BinhKorn instance.
    """
    return BinhKorn()


def test_binh_korn_compute(binh_korn):
    """Test the Binh-Korn objective function."""
    x = array([0.0, 0.0])
    obj = binh_korn.objective(x)
    assert array([0.0, 50.0]) == pytest.approx(obj)


def test_binh_korn_compute_jac(binh_korn):
    """Test the Binh-Korn objective function Jacobian."""
    x = array([1.0, 1.0])
    jac = binh_korn.objective.jac(x)
    assert array([[8.0, 8.0], [-8.0, -8.0]]) == pytest.approx(jac)


def test_binh_korn_ineq1(binh_korn):
    """Test the Binh-Korn first ineq."""
    x = array([1.0, 0.0])
    ineq1 = binh_korn.constraints[0](x)
    assert array([-9.0]) == pytest.approx(ineq1)


def test_binh_korn_ineq1_jac(binh_korn):
    """Test the Binh-Korn first ineq jacobian."""
    x = array([1.0, 1.0])
    jac = binh_korn.constraints[0].jac(x)
    assert array([[-8.0, 2.0]]) == pytest.approx(jac)


def test_binh_korn_ineq2(binh_korn):
    """Test the Binh-Korn second ineq."""
    x = array([1.0, 0.0])
    ineq1 = binh_korn.constraints[1](x)
    assert array([-44.3]) == pytest.approx(ineq1)


def test_binh_korn_ineq2_jac(binh_korn):
    """Test the Binh-Korn second ineq jacobian."""
    x = array([1.0, 1.0])
    jac = binh_korn.constraints[1].jac(x)
    assert array([[14.0, 4.0]]) == pytest.approx(jac)
