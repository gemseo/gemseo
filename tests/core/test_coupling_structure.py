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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import unittest
from builtins import range, str
from copy import deepcopy
from os import remove
from os.path import exists

from future import standard_library
from numpy import array

from gemseo import LOGGER, SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

from .test_dependency_graph import DESC_LIST_16_DISC, generate_disciplines_from_desc

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class TestCouplingStructure(unittest.TestCase):
    """Test the methods of the coupling structure class"""

    def test_couplings_sellar(self):
        """Verify the strong/weak/total couplings of Sellar pb"""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        coupling_structure = MDOCouplingStructure(disciplines)

        strong_couplings = coupling_structure.strong_couplings()
        weak_couplings = coupling_structure.weak_couplings()
        assert strong_couplings == sorted(["y_1", "y_0"])
        assert weak_couplings == sorted(["c_1", "c_2", "obj"])

        input_coupl = coupling_structure.input_couplings(disciplines[1])
        assert input_coupl == ["y_0"]
        input_coupl = coupling_structure.input_couplings(disciplines[2])
        assert input_coupl == sorted(["y_0", "y_1"])
        self.assertRaises(TypeError, coupling_structure.find_discipline, self)

        self.assertRaises(ValueError, coupling_structure.find_discipline, "self")

    def test_strong_weak_coupling(self):
        disciplines = [SobieskiStructure(), SobieskiMission()]
        coupling_structure = MDOCouplingStructure(disciplines)
        s1_o_strong = coupling_structure.output_couplings(disciplines[0], strong=True)
        assert len(s1_o_strong) == 0
        s1_o_weak = coupling_structure.output_couplings(disciplines[0], strong=False)
        assert s1_o_weak == ["y_14"]

    def test_n2(self):
        """Verify the strong/weak/total couplings of Sellar pb"""
        disciplines = [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        coupling_structure = MDOCouplingStructure(disciplines)

        coupling_structure.plot_n2_chart("n2.png", False, show=False, save=True)
        assert exists("n2.png")
        remove("n2.png")
        coupling_structure.plot_n2_chart("n2.png", True, show=False, save=True)
        assert exists("n2.png")
        remove("n2.png")

        from random import shuffle

        disc_shuff = deepcopy(DESC_LIST_16_DISC)
        shuffle(disc_shuff)
        disciplines = generate_disciplines_from_desc(disc_shuff)
        coupling_structure = MDOCouplingStructure(disciplines)

        fname = "n2_16d.png"
        coupling_structure.plot_n2_chart(fname, False, show=False, save=True)
        assert exists(fname)
        remove(fname)

    def test_n2_many_io(self):
        a = MDODiscipline("a")
        b = MDODiscipline("b")
        a.input_grammar.initialize_from_data_names(["i" + str(i) for i in range(30)])
        a.output_grammar.initialize_from_data_names(["o" + str(i) for i in range(30)])
        b.output_grammar.initialize_from_data_names(["i" + str(i) for i in range(30)])
        b.input_grammar.initialize_from_data_names(["o" + str(i) for i in range(30)])

        fpath = "n2.pdf"
        if exists(fpath):
            remove(fpath)
        cpl = MDOCouplingStructure([a, b])
        cpl.plot_n2_chart(save=True, show=False)
        remove(fpath)

    def test_self_coupled(self):
        sc_disc = SelfCoupledDisc()
        sc_disc.execute()

        coupl = MDOCouplingStructure([sc_disc])
        assert coupl.get_all_couplings() == ["y"]
        assert coupl.strongly_coupled_disciplines() == [sc_disc]
        assert coupl.weakly_coupled_disciplines() == []
        assert coupl.weak_couplings() == []
        assert coupl.strong_couplings() == ["y"]


class SelfCoupledDisc(MDODiscipline):
    def __init__(self):
        MDODiscipline.__init__(self)
        self.input_grammar.initialize_from_data_names(["y"])
        self.output_grammar.initialize_from_data_names(["y"])
        self.default_inputs["y"] = array([0.2])

    def _run(self):
        self.local_data["y"] = 1.0 - self.local_data["y"]
