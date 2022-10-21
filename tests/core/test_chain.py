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
from __future__ import annotations

import os
import unittest
from itertools import permutations

import pytest
from gemseo.core.chain import MDOAdditiveChain
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.execution_sequence import ParallelExecSequence
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import allclose
from numpy import ones

DIRNAME = os.path.dirname(__file__)


class Testmdochain(unittest.TestCase):
    """"""

    def get_disciplines_list(self, perm, dtype="complex128"):

        disciplines = [
            SobieskiStructure(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiPropulsion(dtype),
            SobieskiMission(dtype),
        ]

        diff_order_disc = []
        for p in perm:
            diff_order_disc.append(disciplines[p])
        return diff_order_disc

    def test_linearize_sobieski_chain_combinatorial(self):
        """"""
        for perm in permutations(range(4)):
            disciplines = self.get_disciplines_list(perm)
            chain = MDOChain(disciplines)
            ok = chain.check_jacobian(
                derr_approx="complex_step", step=1e-30, threshold=1e-6
            )
            assert ok

    def test_add_differentiated_inputs(self):
        disciplines = [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        chain = MDOChain(disciplines)
        chain.add_differentiated_inputs()
        chain.add_differentiated_outputs()

    def test_parallel_chain_combinatorial_thread(self):
        for perm in permutations(range(4)):
            disciplines = self.get_disciplines_list(perm)
            chain = MDOParallelChain(disciplines, use_threading=True)
            chain.linearize(force_all=True)
            ok = chain.check_jacobian(
                chain.default_inputs,
                derr_approx="complex_step",
                step=1e-30,
                threshold=1e-6,
            )
            assert ok

    @pytest.mark.skip_under_windows
    def test_parallel_chain_combinatorial_mprocess(self):
        # Keep the two first only as MP is slow there
        perms = list(permutations(range(4)))[:2]
        for perm in perms:
            disciplines = self.get_disciplines_list(perm)
            chain = MDOParallelChain(disciplines, use_threading=False)
            ok = chain.check_jacobian(
                chain.default_inputs,
                derr_approx="complex_step",
                step=1e-30,
                threshold=1e-6,
            )
            assert ok

    def test_workflow_dataflow(self):
        disciplines = self.get_disciplines_list(range(4))
        chain = MDOParallelChain(disciplines)
        assert isinstance(chain.get_expected_workflow(), ParallelExecSequence)
        assert chain.get_expected_dataflow() == []

    def test_common_in_out(self):
        # Check that the linearization works with a discipline
        # that has inputs and outputs of the same name
        a = AnalyticDiscipline({"x": "x"}, name="a")
        o = AnalyticDiscipline({"o": "x+y"}, name="o")
        chain = MDOChain([a, o])
        assert chain.check_jacobian(
            {"x": ones(1), "y": ones(1)},
            step=1e-6,
            threshold=1e-5,
            derr_approx="finite_differences",
        )

    def test_double_mission_chain(self):

        # Create a chain that adds two missions
        disciplines = [SobieskiMission(), SobieskiMission()]
        outputs_to_sum = ["y_4"]
        chain = MDOAdditiveChain(disciplines, outputs_to_sum)

        # Check the output value
        chain.execute()
        mission = SobieskiMission()
        mission.execute()
        assert allclose(chain.local_data["y_4"], mission.local_data["y_4"] * 2.0)

        # Check the output Jacobian
        chain.check_jacobian(threshold=1e-5)
