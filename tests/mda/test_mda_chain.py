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

from os.path import exists

import numpy as np
import pytest
from numpy import ones

from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammar import SimpleGrammar
from gemseo.core.json_grammar import JSONGrammar
from gemseo.mda.mda_chain import MDAChain
from gemseo.problems.sellar.sellar import Sellar1, Sellar2, SellarSystem
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

DISC_DESCR_16D = [
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
    ("P", ["u", "v", "z"], []),
]


@pytest.mark.usefixtures("tmp_wd")
def test_sellar():
    """"""
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda_chain = MDAChain(
        disciplines, tolerance=1e-12, max_mda_iter=20, chain_linearize=False
    )
    input_data = {
        "x_local": np.array([0.7]),
        "x_shared": np.array([1.97763897, 0.2]),
        "y_0": np.array([1.0]),
        "y_1": np.array([1.0]),
    }
    inputs = ["x_local", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    assert mda_chain.check_jacobian(
        input_data,
        derr_approx=MDODiscipline.COMPLEX_STEP,
        inputs=inputs,
        outputs=outputs,
        threshold=1e-5,
    )
    mda_chain.plot_residual_history(filename="mda_chain_residuals")
    res_file = "MDAJacobi_mda_chain_residuals.png"
    assert exists(res_file)


def test_sellar_chain_linearize():
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    inputs = ["x_local", "x_shared"]
    outputs = ["obj", "c_1", "c_2"]
    mda_chain = MDAChain(
        disciplines,
        tolerance=1e-13,
        max_mda_iter=30,
        chain_linearize=True,
        warm_start=True,
    )

    ok = mda_chain.check_jacobian(
        derr_approx=MDODiscipline.FINITE_DIFFERENCES,
        inputs=inputs,
        outputs=outputs,
        step=1e-6,
        threshold=1e-5,
    )
    assert ok


def test_sobieski():
    disciplines = [
        SobieskiAerodynamics(),
        SobieskiStructure(),
        SobieskiPropulsion(),
        SobieskiMission(),
    ]
    MDAChain(disciplines)


def generate_disciplines_from_desc(
    description_list, grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE
):
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


def test_16_disc_parallel():
    disciplines = generate_disciplines_from_desc(DISC_DESCR_16D)
    MDAChain(disciplines, sub_mda_class="MDAJacobi")


@pytest.mark.parametrize(
    "in_gtype", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
def test_simple_grammar_type(in_gtype):
    disciplines = generate_disciplines_from_desc(DISC_DESCR_16D, in_gtype)
    mda = MDAChain(
        disciplines,
        sub_mda_class="MDAJacobi",
        grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
    )

    assert type(mda.input_grammar) == SimpleGrammar
    assert type(mda.mdo_chain.input_grammar) == SimpleGrammar
    for smda in mda.sub_mda_list:
        assert type(smda.input_grammar) == SimpleGrammar


def test_mix_sim_jsongrammar():
    disciplines = [Sellar1(), Sellar2(), SellarSystem()]
    mda_chain_s = MDAChain(
        disciplines,
        grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
    )
    assert type(mda_chain_s.input_grammar) == SimpleGrammar

    out_1 = mda_chain_s.execute()

    mda_chain = MDAChain(
        disciplines,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
    )
    assert type(mda_chain.input_grammar) == JSONGrammar

    out_2 = mda_chain.execute()

    assert out_1["obj"] == out_2["obj"]
