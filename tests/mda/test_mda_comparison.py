# -*- coding: utf-8 -*-
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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest

import numpy as np

from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton import MDANewtonRaphson
from gemseo.mda.sequential_mda import GSNewtonMDA
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.chains import SobieskiMDAGaussSeidel, SobieskiMDAJacobi

DIRNAME = os.path.dirname(__file__)


class TestMDAComparison(unittest.TestCase):
    """Comparisons between MDA."""

    def test_compare_mda_jacobi_gs(self):
        """Compare the results of Jacobi and Gauss-Seidel."""
        mda = SobieskiMDAJacobi()
        out1 = mda.execute()

        mda2 = SobieskiMDAGaussSeidel()
        out2 = mda2.execute()

        for key, value1 in out1.items():
            assert (
                np.linalg.norm(np.array(out2[key] - value1)) / np.linalg.norm(value1)
                < 1e-2
            )

    def test_mda_jacobi_newton_hybrid(self):
        """Compare Newton and Gauss-Seidel MDA."""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]

        mda_j = MDAJacobi(disciplines)
        out1 = mda_j.execute()

        mda_newton = MDANewtonRaphson(disciplines)
        out2 = mda_newton.execute()

        mda_hybrid = GSNewtonMDA(disciplines)
        out3 = mda_hybrid.execute()

        for key, value1 in out1.items():
            nv1 = np.linalg.norm(value1)
            if nv1 > 1e-14:
                assert (
                    np.linalg.norm(np.array(out2[key] - value1))
                    / np.linalg.norm(value1)
                    < 1e-2
                )
                assert (
                    np.linalg.norm(np.array(out3[key] - value1))
                    / np.linalg.norm(value1)
                    < 1e-2
                )
            else:
                assert np.linalg.norm(np.array(out2[key] - value1)) < 1e-2
                assert np.linalg.norm(np.array(out3[key] - value1)) < 1e-2
