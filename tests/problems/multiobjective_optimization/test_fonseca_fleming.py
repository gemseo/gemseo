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
"""Tests for the Fonseca-Fleming analytical problem."""

from __future__ import annotations

import pytest

from gemseo.problems.multiobjective_optimization.fonseca_fleming import FonsecaFleming


@pytest.mark.parametrize("dim", [1, 3])
def test_fon(dim):
    """Test the Jacobian of the Fonseca-Fleming objective."""
    problem = FonsecaFleming(dim)
    problem.objective.check_grad(problem.design_space.get_current_value(), step=1e-8)
