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

from __future__ import division, unicode_literals

import unittest
from os.path import exists
from random import shuffle

import pytest
from numpy import array

from gemseo.core.analytic_discipline import AnalyticDiscipline
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.sellar.sellar import (
    C_1,
    C_2,
    OBJ,
    Y_1,
    Y_2,
    Sellar1,
    Sellar2,
    SellarSystem,
)
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

from .test_dependency_graph import DISC_DESCRIPTIONS, create_disciplines_from_desc


@pytest.mark.usefixtures("tmp_wd")
class TestCouplingStructure(unittest.TestCase):
    """Test the methods of the coupling structure class."""

    def test_couplings_sellar(self):
        """Verify the strong/weak/total couplings of Sellar pb."""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        coupling_structure = MDOCouplingStructure(disciplines)

        strong_couplings = coupling_structure.strong_couplings()
        weak_couplings = coupling_structure.weak_couplings()
        assert strong_couplings == [Y_1, Y_2]
        assert weak_couplings == [C_1, C_2, OBJ]

        input_coupl = coupling_structure.input_couplings(disciplines[1])
        assert input_coupl == [Y_1]
        input_coupl = coupling_structure.input_couplings(disciplines[2])
        assert input_coupl == [Y_1, Y_2]
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
        """Verify the strong/weak/total couplings of Sellar pb."""
        disciplines = [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        coupling_structure = MDOCouplingStructure(disciplines)

        coupling_structure.plot_n2_chart("n2_1.png", False, show=False, save=True)
        assert exists("n2_1.png")
        coupling_structure.plot_n2_chart("n2_2.png", True, show=False, save=True)
        assert exists("n2_2.png")

        disc_shuff = list(DISC_DESCRIPTIONS["16"].items())
        shuffle(disc_shuff)
        disc_shuff = dict(disc_shuff)
        disciplines = create_disciplines_from_desc(disc_shuff)
        coupling_structure = MDOCouplingStructure(disciplines)

        fname = "n2_16d.png"
        coupling_structure.plot_n2_chart(fname, False, show=False, save=True)
        assert exists(fname)

        coupling_structure = MDOCouplingStructure([disciplines[0]])
        with pytest.raises(ValueError):
            coupling_structure.plot_n2_chart("n2_3.png", False, show=False, save=True)

    def test_n2_many_io(self):
        a = MDODiscipline("a")
        b = MDODiscipline("b")
        a.input_grammar.initialize_from_data_names(["i" + str(i) for i in range(30)])
        a.output_grammar.initialize_from_data_names(["o" + str(i) for i in range(30)])
        b.output_grammar.initialize_from_data_names(["i" + str(i) for i in range(30)])
        b.input_grammar.initialize_from_data_names(["o" + str(i) for i in range(30)])

        cpl = MDOCouplingStructure([a, b])
        cpl.plot_n2_chart(save=True, show=False)

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


def get_strong_couplings(analytic_expressions):
    """Computes the strong coupling associated.

    Args:
        analytic_expressions: the list of formulas dict
        to build the AnalyticDisciplines.

    Returns:
        The strong couplings.
    """
    disciplines = [
        AnalyticDiscipline(expressions_dict=desc) for desc in analytic_expressions
    ]
    return MDOCouplingStructure(disciplines).strong_couplings()


def test_strong_couplings_basic():
    """Tests a particular coupling structure."""

    coupl = get_strong_couplings(
        (
            {"c1": "x+0.2*c2", "out1": "x"},
            {"c2": "x+0.2*c1", "out2": "x"},
            {"obj": "x+c1+c2+out1+out2+cs"},
        )
    )

    assert coupl == ["c1", "c2"]


def test_strong_couplings_self_coupled():
    """Tests a particular coupling structure with self couplings."""

    coupl = get_strong_couplings(
        (
            {"cs": "x+0.2*cs"},
            {"c1": "x+0.2*c2", "out1": "x"},
            {"c2": "x+0.2*c1", "out2": "x"},
            {"obj": "x+c1+c2+out1+out2+cs"},
        )
    )

    assert coupl == ["c1", "c2", "cs"]
