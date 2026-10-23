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
"""Tests for the module default_settings."""

from __future__ import annotations

from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    default_d_0,
)
from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    default_d_i,
)
from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    default_n_disciplines,
)
from gemseo.problem.mdo.scalable.parametric.standalone.default_settings import (
    default_p_i,
)


def test_default_n_disciplines() -> None:
    """Check default_n_disciplines."""
    assert default_n_disciplines == 2


def test_default_d_0() -> None:
    """Check default_d_0."""
    assert default_d_0 == 1


def test_default_d_i() -> None:
    """Check default_d_i."""
    assert default_d_i == 1


def test_default_p_i() -> None:
    """Check default_p_i."""
    assert default_p_i == 1
