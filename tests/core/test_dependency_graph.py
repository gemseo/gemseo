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
from builtins import str

from future import standard_library
from numpy import ones

from gemseo import SOFTWARE_NAME
from gemseo.algos.design_space import DesignSpace
from gemseo.api import configure_logger
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
from gemseo.utils.py23_compat import TemporaryDirectory
from gemseo.utils.xdsmizer import XDSMizer

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)

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


class Test_DependencyGraph(unittest.TestCase):
    """ """

    def print_coupling_structure(self, coupling_structure):
        """

        :param coupling_structure:

        """
        graph = coupling_structure.graph
        LOGGER.info("Initial initial_graph: " + str(graph.initial_graph))
        LOGGER.info("Strongly connected components: " + str(graph.components))
        LOGGER.info("Reduced graph: " + str(graph.reduced_graph))
        LOGGER.info("Reduced edges: " + str(graph.reduced_edges))
        LOGGER.info("Execution sequence: " + str(graph.execution_sequence))
        # strong/weak disciplines
        strong_disciplines = coupling_structure.strongly_coupled_disciplines()
        weak_disciplines = coupling_structure.weakly_coupled_disciplines()
        LOGGER.info(
            "Strongly coupled disciplines: " + str([d.name for d in strong_disciplines])
        )
        LOGGER.info(
            "Weakly coupled disciplines: " + str([d.name for d in weak_disciplines])
        )
        # strong/weak couplines
        LOGGER.info("Strong couplings: " + str(coupling_structure.strong_couplings()))
        LOGGER.info("Weak couplings: " + str(coupling_structure.weak_couplings()))

    def test_couplings_sellar(self):
        """ """
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        coupling_structure = MDOCouplingStructure(disciplines)

        LOGGER.info("*** Sellar problem")
        with TemporaryDirectory() as base_path:
            assert coupling_structure.graph.execution_sequence == [[(1, 0)], [(2,)]]
            self.print_coupling_structure(coupling_structure)
            coupling_structure.graph.export_initial_graph(
                file_path=base_path + "_initial_graph.pdf"
            )
            coupling_structure.graph.export_reduced_graph(
                file_path=base_path + "_reduced_graph.pdf"
            )

    def test_couplings_sobieski(self):
        """ """
        disciplines = [
            SobieskiAerodynamics(),
            SobieskiStructure(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        coupling_structure = MDOCouplingStructure(disciplines)

        LOGGER.info("*** Sobieski problem")
        with TemporaryDirectory() as base_path:
            assert coupling_structure.graph.execution_sequence == [[(2, 1, 0)], [(3,)]]
            self.print_coupling_structure(coupling_structure)
            coupling_structure.graph.export_initial_graph(
                file_path=base_path + "_initial_graph.pdf"
            )
            coupling_structure.graph.export_reduced_graph(
                file_path=base_path + "_reduced_graph.pdf"
            )

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

    def export_graphs(self, spec, base_path):
        disciplines = generate_disciplines_from_desc(spec)
        coupling_structure = MDOCouplingStructure(disciplines)
        self.print_coupling_structure(coupling_structure)
        coupling_structure.graph.export_initial_graph(
            file_path=base_path + "_initial_graph.pdf"
        )
        coupling_structure.graph.export_reduced_graph(
            file_path=base_path + "_reduced_graph.pdf"
        )

    def test_graph_4disciplines_weak(self):
        with TemporaryDirectory() as base_path:
            self.export_graphs(DESC_LIST_4_DISC_WEAK, base_path)

    def test_graph_3disciplines_weak(self):
        with TemporaryDirectory() as base_path:
            self.export_graphs(DESC_LIST_3_DISC_WEAK, base_path)

    def test_5_disc(self):
        with TemporaryDirectory() as base_path:
            self.export_graphs(DESC_LIST_5_DISC, base_path)

    def test_5_disc_parallel(self):
        with TemporaryDirectory() as base_path:
            self.export_graphs(DESC_LIST_5_DISC, base_path)

    def test_16_disc_parallel(self):
        disciplines = generate_disciplines_from_desc(DESC_LIST_16_DISC)

        design_space = DesignSpace()
        scenario = MDOScenario(
            disciplines, "MDF", objective_name="obj", design_space=design_space
        )
        mda = scenario.formulation.mda
        with TemporaryDirectory() as base_path:
            self.print_coupling_structure(mda.coupling_structure)
            mda.coupling_structure.graph.export_initial_graph(
                file_path=base_path + "_initial_graph.pdf"
            )
            mda.coupling_structure.graph.export_reduced_graph(
                file_path=base_path + "_reduced_graph.pdf"
            )
            XDSMizer(scenario).run(outdir=base_path)
