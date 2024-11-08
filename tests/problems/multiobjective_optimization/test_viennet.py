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
"""Tests for the Viennet analytical problem."""

from __future__ import annotations

from numpy import array

from gemseo.problems.multiobjective_optimization.viennet import Viennet


def test_obj_jacobian():
    """Test the Jacobian of the Viennet objective function."""
    viennet = Viennet(initial_guess=array([1.0, 1.0]))
    viennet.objective.check_grad(array([0.0, 0.0]), step=1e-9, error_max=1e-6)
