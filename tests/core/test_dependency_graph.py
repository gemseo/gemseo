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

import unittest

import pytest
from numpy import ones

from gemseo.algos.design_space import DesignSpace
from gemseo.core.coupling_structure import DependencyGraph, MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)
from gemseo.utils.xdsmizer import XDSMizer

DESC_LIST_5_DISC = [
    ("A", ["b"], ["a", "c"]),
    ("B", ["a"], ["b"]),
    ("C", ["c", "e"], ["d"]),
    ("D", ["d"], ["e", "f"]),
    ("E", ["f"], []),
]

DESC_LIST_16_DISC = [
    ("A", ["a"], ["b"]),
    ("B", ["c"], ["a", "n"]),
    ("C", ["b", "d"], ["c", "e"]),
    ("D", ["f"], ["d", "g"]),
    ("E", ["e"], ["f", "h", "o"]),
    ("F", ["g", "j"], ["i"]),
    ("G", ["i", "h"], ["k", "l"]),
    ("H", ["k", "m"], ["j"]),
    ("I", ["l"], ["m", "w"]),
    ("J", ["n", "o"], ["p", "q"]),
    ("K", ["y"], ["x"]),
    ("L", ["w", "x"], ["y", "z"]),
    ("M", ["p", "s"], ["r"]),
    ("N", ["r"], ["t", "u"]),
    ("O", ["q", "t"], ["s", "v"]),
    ("P", ["u", "v", "z"], ["obj"]),
]


DESC_LIST_3_DISC_WEAK = [
    ("A", ["x"], ["a"]),
    ("B", ["x", "a"], ["b"]),
    ("C", ["x", "a"], ["c"]),
]

DESC_LIST_4_DISC_WEAK = [
    ("A", ["x"], ["a"]),
    ("B", ["x", "a"], ["b"]),
    ("C", ["x", "a"], ["c"]),
    ("D", ["b", "c"], ["d"]),
]


def generate_disciplines_from_desc(description_list):
    """
    :param description_list:

    """
    disciplines = []
    data = ones(1)
    for desc in description_list:
        name = desc[0]
        input_d = {k: data for k in desc[1]}
        output_d = {k: data for k in desc[2]}
        disc = MDODiscipline(name)
        disc.input_grammar.initialize_from_base_dict(input_d)
        disc.output_grammar.initialize_from_base_dict(output_d)
        disciplines.append(disc)
    return disciplines


@pytest.mark.usefixtures("tmp_wd")
class TestDependencyGraph(unittest.TestCase):
    """"""

    def test_couplings_sellar(self):
        """"""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        coupling_structure = MDOCouplingStructure(disciplines)

        assert coupling_structure.graph.execution_sequence == [[(1, 0)], [(2,)]]
        coupling_structure.graph.export_initial_graph(file_path="_initial_graph.pdf")
        coupling_structure.graph.export_reduced_graph(file_path="_reduced_graph.pdf")

    def test_couplings_sobieski(self):
        """"""
        disciplines = [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        coupling_structure = MDOCouplingStructure(disciplines)

        assert coupling_structure.graph.execution_sequence == [[(2, 1, 0)], [(3,)]]
        coupling_structure.graph.export_initial_graph(file_path="_initial_graph.pdf")
        coupling_structure.graph.export_reduced_graph(file_path="_reduced_graph.pdf")

    def test_graph_disciplines_couplings(self):
        disciplines = generate_disciplines_from_desc(DESC_LIST_5_DISC)
        graph = DependencyGraph(disciplines)
        couplings = graph.get_disciplines_couplings()
        expected = [
            (disciplines[0], disciplines[1], ["a"]),
            (disciplines[1], disciplines[0], ["b"]),
            (disciplines[0], disciplines[2], ["c"]),
            (disciplines[2], disciplines[3], ["d"]),
            (disciplines[3], disciplines[2], ["e"]),
            (disciplines[3], disciplines[4], ["f"]),
        ]
        assert len(expected) == len(couplings)
        for ref in expected:
            assert ref in couplings
        # self.assertEqual(sorted(expected), sorted(couplings))

    def export_graphs(self, spec):
        disciplines = generate_disciplines_from_desc(spec)
        coupling_structure = MDOCouplingStructure(disciplines)
        coupling_structure.graph.export_initial_graph(file_path="_initial_graph.pdf")
        coupling_structure.graph.export_reduced_graph(file_path="_reduced_graph.pdf")

    def test_graph_4disciplines_weak(self):
        self.export_graphs(DESC_LIST_4_DISC_WEAK)

    def test_graph_3disciplines_weak(self):
        self.export_graphs(DESC_LIST_3_DISC_WEAK)

    def test_5_disc(self):
        self.export_graphs(DESC_LIST_5_DISC)

    def test_5_disc_parallel(self):
        self.export_graphs(DESC_LIST_5_DISC)

    def test_16_disc_parallel(self):
        disciplines = generate_disciplines_from_desc(DESC_LIST_16_DISC)

        design_space = DesignSpace()
        scenario = MDOScenario(
            disciplines, "MDF", objective_name="obj", design_space=design_space
        )
        mda = scenario.formulation.mda

        mda.coupling_structure.graph.export_initial_graph(
            file_path="_initial_graph.pdf"
        )
        mda.coupling_structure.graph.export_reduced_graph(
            file_path="_reduced_graph.pdf"
        )
        XDSMizer(scenario).run()
