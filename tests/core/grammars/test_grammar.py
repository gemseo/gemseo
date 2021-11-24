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


import numpy as np
import pytest
from numpy import array, ndarray

from gemseo.core.grammars.errors import InvalidDataException
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.utils.py23_compat import Path

TEST_PATH = Path(__file__).parent / "data"


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
    return SimpleGrammar(
        "CFD_inputs",
        names_to_types={
            "Mach": float,
            "Cl": float,
            "Turbulence_model": str,
            "Navier-Stokes": type(True),
            "bounds": ndarray,
        },
    )


def get_base_grammar_from_instanciation():
    my_grammar = SimpleGrammar("CFD_inputs")
    my_grammar.update_elements(Mach=float)
    my_grammar.update_elements(Cl=float)
    my_grammar.update_elements(Turbulence_model=str)
    my_grammar.update_elements(**{"Navier-Stokes": bool})
    my_grammar.update_elements(bounds=np.ndarray)
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
        indx = list(g1.data_names).index(g1_name)
        assert g1_type == list(g2.data_types)[indx]
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
    with pytest.raises(
        TypeError, match="is not a type and cannot be used as a type specification"
    ):
        g.update_elements(**{g.data_names[0]: "unknowntype"})

    my_grammar = SimpleGrammar("toto")
    my_grammar.initialize_from_base_dict({"X": 2})
    ge.update_from_if_not_in(my_grammar, g)

    g_json = JSONGrammar("titi")
    my_grammar.update_from_if_not_in(g_json, g_json)
    my_grammar.clear()


def test_invalid_data():
    gram = SimpleGrammar("dummy", {"X": float})
    gram.load_data({"X": 1.1})

    with pytest.raises(TypeError):
        SimpleGrammar("dummy", {"X": "x"})


@pytest.mark.parametrize(
    "data", [{}, {"Mach": 2}, {"X": "1"}, {"X": "/opt"}, {"X": array([1.0])}, 1, "X"]
)
def test_invalid_data2(data):
    gram = SimpleGrammar("dummy", {"X": float})
    with pytest.raises(InvalidDataException):
        gram.load_data(data)


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
    jgrammar = JSONGrammar("jgrammar", schema_file=TEST_PATH / infile)
    sgrammar.update_from(jgrammar)
    assert sorted(sgrammar.get_data_names()) == sorted(jgrammar.get_data_names())


@pytest.mark.parametrize(
    "infile", ["grammar_test1.json", "grammar_test2.json", "grammar_test3.json"]
)
def test_update_from_ifnotin_simple_json(infile):
    sgrammar = SimpleGrammar("simple")
    jgrammar = JSONGrammar("jgrammar", schema_file=TEST_PATH / infile)
    exclude_grammar = JSONGrammar(
        "jgrammar", schema_file=TEST_PATH / "grammar_test1.json"
    )
    sgrammar.update_from_if_not_in(jgrammar, exclude_grammar)
    assert sorted(sgrammar.get_data_names()) == sorted(
        list(set(jgrammar.get_data_names()) - set(exclude_grammar.get_data_names()))
    )


def test_add_remove_item():
    grammar = SimpleGrammar("")
    assert "a" not in grammar
    assert not grammar.is_data_name_existing("a")
    grammar.update_elements(a=int)
    assert "a" in grammar.get_data_names()
    assert grammar.is_data_name_existing("a")
    assert "a" in grammar

    grammar.remove_item("a")
    assert not grammar.data_types
    assert not grammar.data_names
    assert not grammar.is_data_name_existing("a")


@pytest.mark.parametrize("grammar_type", [JSONGrammar, SimpleGrammar])
def test_is_array(grammar_type):
    grammar = grammar_type("Grammar")
    grammar.initialize_from_base_dict({"a": array([1]), "b": 1, "c": [1]})
    assert "a" in grammar
    assert grammar.is_type_array("a")
    assert not grammar.is_type_array("b")
    assert grammar.is_type_array("c") == (grammar_type == JSONGrammar)


def test_array_type():
    grammar = SimpleGrammar("g")
    grammar.initialize_from_base_dict({"a": array([1.0])})
    with pytest.raises(InvalidDataException):
        grammar.load_data({"x": 2.0})


def test_add_elements():
    grammar = SimpleGrammar("g")
    grammar.update_elements(x=int)
    assert "x" in grammar

    with pytest.raises(TypeError, match="is not a type"):
        grammar.update_elements(x=1)


def test_update_required_not_in_grammar():
    """Test an error is raised when setting a non-existent element as required."""
    grammar = SimpleGrammar("g")

    with pytest.raises(KeyError, match="Data named x is not in the grammar."):
        grammar.update_required_elements(x=True)


def test_update_required_not_bool():
    """Test that an error is raised when no boolean value is given to
    update_required_elements."""
    grammar = SimpleGrammar("g")
    grammar.update_elements(x=int)

    with pytest.raises(TypeError, match="Boolean is required for element x."):
        grammar.update_required_elements(x=int)


def test_init_with_required():
    """Test that the grammar is established correctly when given required elements."""
    grammar = SimpleGrammar(
        "g",
        names_to_types={"toto": str, "foo": int, "toto2": int},
        required_names={"toto": False, "foo": True},
    )
    assert not grammar.is_required("toto")
    assert grammar.is_required("foo")
    assert grammar.is_required("toto2")


def test_is_required_error():
    """Check that an error is raised for elements that are not in the grammar."""
    grammar = SimpleGrammar("g", names_to_types={"toto": str})
    with pytest.raises(ValueError, match="Element foo is not in the grammar."):
        grammar.is_required("foo")
