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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from os.path import exists

import numpy as np
import pytest

from gemseo.mda.jacobi import MDAJacobi
from gemseo.mda.newton import MDANewtonRaphson
from gemseo.mda.sequential_mda import GSNewtonMDA, MDASequential
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem

DIRNAME = os.path.dirname(__file__)


@pytest.mark.usefixtures("tmp_wd")
class TestSequential(unittest.TestCase):
    """Test the sequential MDA."""

    @staticmethod
    def test_sequential_mda_sellar():
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]

        mda1 = MDAJacobi(disciplines, max_mda_iter=1)
        mda2 = MDANewtonRaphson(disciplines)
        mda_sequence = [mda1, mda2]

        mda = MDASequential(disciplines, mda_sequence, max_mda_iter=20)
        mda.reset_history_each_run = True
        mda.execute()

        y_ref = np.array([0.80004953, 1.79981434])
        y_opt = np.array([mda.local_data["y_0"][0].real, mda.local_data["y_1"][0].real])
        assert np.linalg.norm(y_ref - y_opt) / np.linalg.norm(y_ref) < 1e-4

        mda3 = GSNewtonMDA(disciplines, max_mda_iter=4)
        mda3.execute()
        filename = "GS_sellar.pdf"
        mda3.plot_residual_history(show=False, save=True, filename=filename)

        assert exists(filename)
        y_opt = np.array(
            [mda3.local_data["y_0"][0].real, mda3.local_data["y_1"][0].real]
        )
        assert np.linalg.norm(y_ref - y_opt) / np.linalg.norm(y_ref) < 1e-4
