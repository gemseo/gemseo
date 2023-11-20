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
"""Test for the module variable."""

from __future__ import annotations

from numpy import array

from gemseo.problems.scalable.parametric.core.variable import Variable


def test_variable():
    """Check the named tuple Variable."""
    variable = Variable("foo", 123, array([-1.0]), array([3.0]), array([0.6]))
    assert variable.name == "foo"
    assert variable.size == 123
    assert variable.lower_bound == array([-1.0])
    assert variable.upper_bound == array([3.0])
    assert variable.default_value == array([0.6])
