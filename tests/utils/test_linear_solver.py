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

import unittest

from gemseo.utils.linear_solver import LinearSolver
from numpy import diag
from numpy import ones
from scipy.sparse import csr_matrix


class TestLinearSolver(unittest.TestCase):
    def test_init(self):
        LinearSolver()

    def test_solve(self):
        LinearSolver().solve(diag(list(range(1, 4))), ones(3))

    def test_fail_and_branches(self):
        LinearSolver().solve(diag(list(range(2))), ones(2), maxiter=1)

        self.assertRaises(
            ValueError, LinearSolver().solve, diag(list(range(2))), ones((3, 2))
        )

        a = csr_matrix([[1, 2, 0], [0, 0, 3], [4, 0, 5]])
        b = csr_matrix([1, 2, 0]).T
        LinearSolver().solve(a, b)

        LinearSolver().solve(diag(list(range(2))), ones(2), maxiter=-1)

        self.assertRaises(
            AttributeError,
            LinearSolver().solve,
            diag(list(range(2))),
            ones((3, 2)),
            linear_solver="toto",
        )
