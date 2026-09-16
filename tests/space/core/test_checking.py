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
"""Tests for the checking module."""

from __future__ import annotations

from gemseo.space._core.checking import check_design_space
from gemseo.space.design import DesignSpace
from gemseo.space.random import RandomSpace
from gemseo.util.testing.helper import assert_exception


def test_check_design_space_valid() -> None:
    """Check that a design space is accepted."""
    assert (
        check_design_space(DesignSpace(), "An MDOScenario", "an EvaluationScenario")
        is None
    )


def test_check_design_space_wrong_space(snapshot) -> None:
    """Check the error raised when the space is not a design space."""
    with assert_exception(TypeError, snapshot):
        check_design_space(RandomSpace(), "An MDOScenario", "an EvaluationScenario")
