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
from __future__ import annotations

import re
import unittest
from os.path import exists
from random import shuffle
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from numpy import array

from gemseo import create_discipline
from gemseo.core import coupling_structure
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.discipline import Discipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.variables import C_1
from gemseo.problems.mdo.sellar.variables import C_2
from gemseo.problems.mdo.sellar.variables import OBJ
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.testing.disciplines_creator import create_disciplines_from_desc
from gemseo.utils.testing.helpers import image_comparison

from .test_dependency_graph import DISC_DESCRIPTIONS

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


@pytest.mark.usefixtures("tmp_wd")
class TestCouplingStructure(unittest.TestCase):
    """Test the methods of the coupling structure class."""

    def test_couplings_sellar(self) -> None:
        """Verify the strong/weak/total couplings of Sellar pb."""
        disciplines = [Sellar1(), Sellar2(), SellarSystem()]
        coupling_structure = CouplingStructure(disciplines)

        strong_couplings = coupling_structure.strong_couplings
        weak_couplings = coupling_structure.weak_couplings
        assert strong_couplings == [Y_1, Y_2]
        assert weak_couplings == [C_1, C_2, OBJ]

        input_coupl = coupling_structure.get_input_couplings(disciplines[1])
        assert input_coupl == [Y_1]
        input_coupl = coupling_structure.get_input_couplings(disciplines[2])
        assert input_coupl == [Y_1, Y_2]

        self.assertRaises(ValueError, coupling_structure.find_discipline, "self")

    def test_strong_weak_coupling(self) -> None:
        disciplines = [SobieskiStructure(), SobieskiMission()]
        coupling_structure = CouplingStructure(disciplines)
        s1_o_strong = coupling_structure.get_output_couplings(disciplines[0])
        assert len(s1_o_strong) == 0
        s1_o_weak = coupling_structure.get_output_couplings(
            disciplines[0], strong=False
        )
        assert s1_o_weak == ["y_14"]

    def test_n2(self) -> None:
        """Verify the strong/weak/total couplings of Sellar pb."""
        disciplines = [
            SobieskiStructure(),
            SobieskiAerodynamics(),
            SobieskiPropulsion(),
            SobieskiMission(),
        ]
        coupling_structure = CouplingStructure(disciplines)

        coupling_structure.plot_n2_chart("n2_1.png", False)
        assert exists("n2_1.png")
        coupling_structure.plot_n2_chart("n2_2.png")
        assert exists("n2_2.png")

        disc_shuff = list(DISC_DESCRIPTIONS["16"].items())
        shuffle(disc_shuff)
        disc_shuff = dict(disc_shuff)
        disciplines = create_disciplines_from_desc(disc_shuff)
        coupling_structure = CouplingStructure(disciplines)

        fname = "n2_16d.png"
        coupling_structure.plot_n2_chart(fname, False)
        assert exists(fname)

        coupling_structure = CouplingStructure([disciplines[0]])
        with pytest.raises(
            ValueError, match=re.escape("N2 diagrams need at least two disciplines.")
        ):
            coupling_structure.plot_n2_chart("n2_3.png", False)

    def test_n2_many_io(self) -> None:
        a = DummyDiscipline("a")
        b = DummyDiscipline("b")
        a.io.input_grammar.update_from_names(["i" + str(i) for i in range(30)])
        a.io.output_grammar.update_from_names(["o" + str(i) for i in range(30)])
        b.io.output_grammar.update_from_names(["i" + str(i) for i in range(30)])
        b.io.input_grammar.update_from_names(["o" + str(i) for i in range(30)])

        cpl = CouplingStructure([a, b])
        cpl.plot_n2_chart()

    def test_self_coupled(self) -> None:
        sc_disc = SelfCoupledDisc()
        sc_disc.execute()

        coupl = CouplingStructure([sc_disc])
        assert coupl.all_couplings == ["y"]
        assert coupl.strongly_coupled_disciplines == [sc_disc]
        assert coupl.weakly_coupled_disciplines == []
        assert coupl.weak_couplings == []
        assert coupl.strong_couplings == ["y"]


class SelfCoupledDisc(Discipline):
    def __init__(self) -> None:
        Discipline.__init__(self)
        self.io.input_grammar.update_from_names(["y"])
        self.io.output_grammar.update_from_names(["y"])
        self.io.input_grammar.defaults["y"] = array([0.2])

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self.io.data["y"] = 1.0 - self.io.data["y"]


def get_strong_couplings(analytic_expressions):
    """Computes the strong coupling associated.

    Args:
        analytic_expressions: the list of formulas dict
        to build the AnalyticDisciplines.

    Returns:
        The strong couplings.
    """
    disciplines = [AnalyticDiscipline(desc) for desc in analytic_expressions]
    return CouplingStructure(disciplines).strong_couplings


def test_strong_couplings_basic() -> None:
    """Tests a particular coupling structure."""
    coupl = get_strong_couplings((
        {"c1": "x+0.2*c2", "out1": "x"},
        {"c2": "x+0.2*c1", "out2": "x"},
        {"obj": "x+c1+c2+out1+out2+cs"},
    ))

    assert coupl == ["c1", "c2"]


def test_strong_couplings_self_coupled() -> None:
    """Tests a particular coupling structure with self couplings."""
    coupl = get_strong_couplings((
        {"cs": "x+0.2*cs"},
        {"c1": "x+0.2*c2", "out1": "x"},
        {"c2": "x+0.2*c1", "out2": "x"},
        {"obj": "x+c1+c2+out1+out2+cs"},
    ))

    assert coupl == ["c1", "c2", "cs"]


@pytest.mark.parametrize(
    ("show_data_names", "descriptions", "baseline_images"),
    [
        (
            False,
            ({"y1": "x1"}, {"y2": "x2"}, {"y3": "x3"}),
            ["n_2_no_coupling"],
        ),
        (True, ({"y1": "x1"}, {"y2": "x2"}, {"y3": "x3"}), ["n_2_no_coupling"]),
        (
            False,
            ({"y1": "x1+y2"}, {"y2": "x2+y1"}, {"y3": "x3+y1+y2"}),
            ["n_2_coupling_no_names"],
        ),
        (
            True,
            ({"y1": "x1+y2"}, {"y2": "x2+y1"}, {"y3": "x3+y1+y2"}),
            ["n_2_coupling_names"],
        ),
        (
            False,
            ({"y1": "y2"}, {"y2": "y1"}, {"y3": "y1+y3"}, {"y4": "y1+y2+y3"}),
            ["n_2_self_coupled_no_names"],
        ),
        (
            True,
            ({"y1": "y2"}, {"y2": "y1"}, {"y3": "y1+y3"}, {"y4": "y1+y2+y3"}),
            ["n_2_self_coupled"],
        ),
    ],
)
@image_comparison(None)
def test_n2_no_coupling(tmp_wd, baseline_images, show_data_names, descriptions) -> None:
    """Test that an N2 plot is generated correctly when there are no couplings.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
        baseline_images: The reference images to be compared.
        show_data_names: If ``True``, show the names of the coupling data ;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
        descriptions: The inputs and outputs to create analytic disciplines.
    """
    disciplines = [
        AnalyticDiscipline(desc, name=f"discipline_{next(iter(desc))}")
        for desc in descriptions
    ]

    CouplingStructure(disciplines).plot_n2_chart(
        f"{baseline_images[0]}.png", show_data_names
    )


def test_coupl_properties() -> None:
    """Test the weak_couplings and get_input_couplings on the Sellar problem."""
    disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
    coupl = CouplingStructure(disciplines)
    scd = coupl.strongly_coupled_disciplines
    assert id(scd) == id(coupl.strongly_coupled_disciplines)

    wcp = coupl.weak_couplings
    assert id(wcp) == id(coupl.weak_couplings)

    assert coupl.get_input_couplings(disciplines[0]) == ["y_2"]

    assert coupl.get_input_couplings(disciplines[1]) == ["y_1"]

    assert coupl.get_input_couplings(disciplines[2]) == ["y_1", "y_2"]

    assert sorted(coupl.get_input_couplings(disciplines[2], strong=False)) == [
        "y_1",
        "y_2",
    ]

    disciplines = create_discipline(["Sellar1", "SellarSystem"])
    coupl = CouplingStructure(disciplines)
    assert coupl.get_input_couplings(disciplines[1], strong=False) == ["y_1"]
    assert coupl.get_input_couplings(disciplines[1]) == []


def test_check_disciplines_consistency() -> None:
    """Test that CouplingStructure checks the consistency of the disciplines."""
    disciplines = create_discipline(["Sellar1", "Sellar2"])
    with mock.patch.object(coupling_structure, "check_disciplines_consistency") as func:
        CouplingStructure(disciplines)

    assert func.call_args.args == (disciplines, True, False)
