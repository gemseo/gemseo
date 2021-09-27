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

from __future__ import division, unicode_literals

from os.path import dirname, join

import numpy as np
import pytest
from numpy import array

from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar


def get_indict():
    """"""
    return {
        "Mach": 1.0,
        "Cl": 2.0,
        "Turbulence_model": "SA",
        "Navier-Stokes": True,
        "bounds": np.array([1.0, 2.0]),
    }


def get_base_grammar_from_inherit():
    """"""

    class MyGrammar(SimpleGrammar):
        """"""

        def __init__(self, name):
            super(MyGrammar, self).__init__(name)
            self.data_names = [
                "Mach",
                "Cl",
                "Turbulence_model",
                "Navier-Stokes",
                "bounds",
            ]
            self.data_types = [float, float, type("str"), type(True), np.array]

    return MyGrammar("CFD_inputs")


def get_base_grammar_from_instanciation():
    """"""
    my_grammar = SimpleGrammar("CFD_inputs")
    my_grammar.data_names = [
        "Mach",
        "Cl",
        "Turbulence_model",
        "Navier-Stokes",
        "bounds",
    ]
    my_grammar.data_types = [float, float, type("str"), type(True), np.array]

    return my_grammar


def get_base_grammar_from_base_dict():
    """"""
    my_grammar = SimpleGrammar("CFD_inputs")
    my_grammar.initialize_from_base_dict(get_indict())
    return my_grammar


def check_g1_in_g2(g1, g2):
    """

    :param g1: param g2:
    :param g2:

    """
    for g1_name, g1_type in zip(g1.data_names, g2.data_types):
        assert g1_name in g2.data_names
        assert g1_name in g2.get_data_names()
        indx = g1.data_names.index(g1_name)
        assert g1_type == g2.data_types[indx]
        assert g2.is_data_name_existing(g1_name)


def check_g1_eq_g2(g1, g2):
    """

    :param g1: param g2:
    :param g2:

    """
    check_g1_in_g2(g1, g2)
    check_g1_in_g2(g2, g1)


def test_inherit_vs_instanciation():
    """"""
    g1 = get_base_grammar_from_instanciation()
    g2 = get_base_grammar_from_inherit()
    check_g1_eq_g2(g1, g2)


def test_dict_init_vs_instanciation():
    """"""
    g1 = get_base_grammar_from_instanciation()
    g2 = get_base_grammar_from_base_dict()
    check_g1_eq_g2(g1, g2)


def test_update_from():
    """"""
    g = get_base_grammar_from_instanciation()
    with pytest.raises(Exception):
        g.update_from({})
    ge = SimpleGrammar(name="empty")
    ge.update_from(g)
    n = len(ge.data_names)
    ge.update_from_if_not_in(ge, g)
    # Update again
    ge.update_from(g)
    # Check no updates are added again
    assert n == len(ge.data_names)
    g.data_types[-1] = "unknowntype"
    with pytest.raises(Exception):
        ge.update_from_if_not_in(*(ge, g))

    my_grammar = SimpleGrammar("toto")
    my_grammar.initialize_from_base_dict({"X": 2})
    ge.update_from_if_not_in(my_grammar, g)

    g_json = JSONGrammar("titi")
    my_grammar.update_from_if_not_in(g_json, g_json)
    my_grammar.clear()


def test_invalid_data():
    gram = SimpleGrammar("dummy")
    gram.data_names = ["X"]
    gram.data_types = [float]

    gram.load_data({"X": 1.1})
    for data in [
        {},
        {"Mach": 2},
        {"X": "1"},
        {"X": "/opt"},
        {"X": array([1.0])},
        1,
        "X",
    ]:
        with pytest.raises(InvalidDataException):
            gram.load_data(data)

    gram = SimpleGrammar("dummy")
    gram.data_names = ["X"]
    gram.data_types = ["x"]
    with pytest.raises(TypeError):
        gram.load_data({"X": 1.1})


def test_is_alldata_exist():
    """"""
    g = get_base_grammar_from_instanciation()
    assert not g.is_all_data_names_existing(["bidon"])
    assert g.is_all_data_names_existing(["Mach"])


def test_get_type_of_data_error():
    """"""
    g = get_base_grammar_from_instanciation()
    with pytest.raises(Exception):
        g.get_type_of_data_named(["bidon"])


@pytest.mark.parametrize(
    "infile", ["grammar_test1.json", "grammar_test2.json", "grammar_test3.json"]
)
def test_update_from_simple_json(infile):
    sgrammar = SimpleGrammar("simple")
    jgrammar = JSONGrammar(
        "jgrammar", schema_file=join(dirname(__file__), "data", infile)
    )
    sgrammar.update_from(jgrammar)
    assert sorted(sgrammar.get_data_names()) == sorted(jgrammar.get_data_names())


@pytest.mark.parametrize(
    "infile", ["grammar_test1.json", "grammar_test2.json", "grammar_test3.json"]
)
def test_update_from_ifnotin_simple_json(infile):
    sgrammar = SimpleGrammar("simple")
    jgrammar = JSONGrammar(
        "jgrammar", schema_file=join(dirname(__file__), "data", infile)
    )
    exclude_grammar = JSONGrammar(
        "jgrammar", schema_file=join(dirname(__file__), "data", "grammar_test1.json")
    )
    sgrammar.update_from_if_not_in(jgrammar, exclude_grammar)
    assert sorted(sgrammar.get_data_names()) == sorted(
        list(set(jgrammar.get_data_names()) - set(exclude_grammar.get_data_names()))
    )