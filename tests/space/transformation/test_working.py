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
"""Tests for the creation of the working transformation a driver asks for."""

from __future__ import annotations

from numpy import array
from numpy.testing import assert_allclose

from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.space.transformation._working import project_onto_declared_domain


def test_project_onto_declared_domain_with_nothing_to_relax() -> None:
    """Check that a float-only design space returns the value itself.

    There is nothing for a relaxation to project, so no working space is
    built and the value is handed back untouched, the same object.
    """
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)
    value = array([2.5])

    assert project_onto_declared_domain(space, value) is value


def test_project_onto_declared_domain_rounds_and_snaps() -> None:
    """Check the projection of a space with an integer and a discrete variable.

    The integer component is rounded and the discrete one is snapped to its
    nearest choice, as `SpaceRelaxation.project` does, and the input is not
    modified in place.
    """
    space = DesignSpace()
    space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0)
    space.add_integer_variable("i", lower_bound=0, upper_bound=10)
    space.add_discrete_variable("d", [1, 3, 8])
    value = array([2.5, 4.6, 2.0])

    projected = project_onto_declared_domain(space, value)

    assert_allclose(value, array([2.5, 4.6, 2.0]))
    assert_allclose(projected, array([2.5, 5.0, 1.0]))


def test_project_onto_declared_domain_of_a_non_design_space() -> None:
    """Check that a non-design space returns the value itself.

    A random space has no domain a relaxation could project onto,
    so the value is handed back as it is.
    """
    value = array([1.0, 2.0])

    assert project_onto_declared_domain(RandomSpace(), value) is value
