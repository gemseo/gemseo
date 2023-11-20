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

from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_0
from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_I
from gemseo.problems.scalable.parametric.core.default_settings import (
    DEFAULT_N_DISCIPLINES,
)
from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_P_I


def test_default_n_disciplines():
    """Check DEFAULT_N_DISCIPLINES."""
    assert DEFAULT_N_DISCIPLINES == 2


def test_default_d_0():
    """Check DEFAULT_D_0."""
    assert DEFAULT_D_0 == 1


def test_default_d_i():
    """Check DEFAULT_D_I."""
    assert DEFAULT_D_I == 1


def test_default_p_i():
    """Check DEFAULT_P_I."""
    assert DEFAULT_P_I == 1
