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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.sobieski.core.utils import SobieskiBase
from numpy import array
from numpy import complex128
from numpy import eye
from numpy import float64
from numpy import zeros


def test_init():
    SobieskiBase("float64")
    SobieskiBase("complex128")
    SobieskiBase(float64)
    SobieskiBase(complex128)

    assert SobieskiBase("float64").dtype == SobieskiBase(float64).dtype
    assert SobieskiBase("complex128").dtype == SobieskiBase(complex128).dtype

    with pytest.raises(ValueError, match="foo"):
        SobieskiBase("foo")


def test_compute_a():
    base = SobieskiBase("float64")
    mtx_shifted = eye(3)
    f_bound = array([[1], [2], [3.1]])
    ao_coeff = zeros([3])
    ai_coeff = zeros([3])
    aij_coeff = zeros([3, 3])
    base._SobieskiBase__compute_a(
        mtx_shifted, f_bound, ao_coeff, ai_coeff, aij_coeff, index=0
    )

    mtx_shifted = zeros((3, 3))
    base._SobieskiBase__compute_a(
        mtx_shifted, f_bound, ao_coeff, ai_coeff, aij_coeff, index=0
    )


def test_compute_fbound():
    base = SobieskiBase("float64")
    base._SobieskiBase__compute_fbound([0], 1, 1, 1, 0)


def test_get_bounds():
    base = SobieskiBase("float64")
    lb1, ub1 = base.get_bounds_by_name("x_shared")
    lb, ub = base.get_bounds_by_name(["x_shared", "x_1"])
    assert len(lb1) + 2 == len(lb)
    assert len(ub1) + 2 == len(ub)
