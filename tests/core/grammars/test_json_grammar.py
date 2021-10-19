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

import logging
from numbers import Number

import pytest
from numpy import array, ndarray

from gemseo.core.grammar import InvalidDataException, SimpleGrammar
from gemseo.core.json_grammar import JSONGrammar
from gemseo.utils.py23_compat import PY2, Path

TEST_PATH = Path(__file__).parent / "data"


def get_indict():
    return {
        "Mach": 1.0,
        "Cl": 2.0,
        "Turbulence_model": "SA",
        "Navier-Stokes": True,
        "bounds": [1.0, 2.0],
    }


def get_indict_grammar():
    g = JSONGrammar(name="basic")
    g.initialize_from_base_dict(typical_data_dict=get_indict())
    return g


def test_basic_grammar_init_from_dict():
    g = get_indict_grammar()
    g.load_data(get_indict())
    g_str = repr(g)
    assert "properties" in g_str
    assert "required" in g_str
    for k in get_indict():
        assert k in g_str

    array_dct = get_indict()
    array_dct["bounds"] = array(array_dct["bounds"])
    g2 = JSONGrammar(name="basic2")
    g2.initialize_from_base_dict(typical_data_dict=array_dct)

    for k in g.get_data_names():
        assert k in g2.get_data_names()

    g2_str = repr(g2)
    assert "properties" in g2_str
    assert "required" in g2_str
    for k in get_indict():
        assert k in g2_str


def test_nan():
    grammar = JSONGrammar("test_gram")
    typical_data_dict = {"no_nan": [1]}
    grammar.initialize_from_base_dict(typical_data_dict)
    grammar.load_data({"no_nan": array([float(1)])})
    with pytest.raises(InvalidDataException):
        grammar.load_data({"no_nan": array([float("nan")])})
    with pytest.raises(InvalidDataException):
        grammar.load_data({"no_nan": array([float("nan"), 1.0])})


#     def test_bench():
#         from numpy import ones
#         from timeit import default_timer as timer
#         n = 1000000
#         p = 1
#         nn = 1
#         typical_data_dict = {str(k): ones(n) for k in range(p)}
#         grammar = JSONGrammar("test_gram")
#         grammar.initialize_from_base_dict(typical_data_dict)
#         t0 = timer()
#         for i in range(nn):
#             grammar.load_data(typical_data_dict)
#         t00 = timer()
#         print("DT ", (t00 - t0) / nn)
#         from numpy import issubdtype, number, isnan
#
#         def check(data):
#             for k, v in data.items():
#                 assert isinstance(v, ndarray)
#                 assert issubdtype(v.dtype, number)
#                 min = v.min()
#                 max = v.max()
#                 assert not isnan(min)
#
#         t1 = timer()
#         for i in range(nn):
#             check(typical_data_dict)
#         t2 = timer()
#         print("DT 2", ((t2 - t1) / (t00 - t0))**-1)


def test_init_from_datanames():
    grammar = JSONGrammar("t")
    names = ["a", "b"]
    grammar.initialize_from_data_names(names)
    for name in names:
        assert name in grammar.get_data_names()


@pytest.mark.usefixtures("tmp_wd")
def test_init_from_base_dict():
    grammar = JSONGrammar("test_gram")
    grammar.initialize_from_base_dict({"a": [1], "b": "b"})


def test_update_from():
    g = get_indict_grammar()
    with pytest.raises(TypeError):
        g.update_from({})
    ge = JSONGrammar(name="empty")
    ge.update_from(g)
    assert sorted(ge.get_data_names()) == sorted(g.get_data_names())

    gs = SimpleGrammar("b")
    with pytest.raises(TypeError):
        ge.update_from_if_not_in(gs, gs)


def test_update_from_if_not_in():

    dct_1 = {
        "Mach": 1.0,
        "Cl": 2.0,
        "Turbulence_model": "SA",
        "Navier-Stokes": True,
        "bounds": [1.0, 2.0],
    }
    description_dict_1 = {
        "Mach": "Mach number",
        "Navier-Stokes": "Equations to be solved",
    }

    dct_2 = {"Mach": 1.0, "Cl": 2.0, "Turbulence_model": "SA"}
    g1 = JSONGrammar(name="basic")
    g1.initialize_from_base_dict(
        typical_data_dict=dct_1,
        description_dict=description_dict_1,
    )

    g2 = JSONGrammar(name="basic")
    g2.initialize_from_base_dict(typical_data_dict=dct_2)

    ge = JSONGrammar(name="empty")
    ge.update_from_if_not_in(g1, g2)

    assert sorted(ge.get_data_names()) == sorted(["bounds", "Navier-Stokes"])

    assert (
        ge.schema.to_schema()["properties"]["Navier-Stokes"]["description"] is not None
    )


@pytest.mark.usefixtures("tmp_wd")
def test_update_from_dict():
    g1 = JSONGrammar("g1")
    typical_data_dict = {"max_iter": 1}
    g1.initialize_from_base_dict(typical_data_dict)

    g2 = JSONGrammar(name="basic_str", schema=g1.schema)
    assert "max_iter" in g2.get_data_names()


def test_invalid_data():
    fpath = TEST_PATH / "grammar_test1.json"
    assert Path(fpath).exists()
    gram = JSONGrammar(name="toto", schema_file=fpath)
    gram.load_data({"X": 1})
    gram.load_data({"X": 1.1})
    for data in [{}, {"Y": 2}, {"X": "/opt"}, {"X": array([1.0])}, 1, "X"]:
        with pytest.raises(InvalidDataException):
            gram.load_data(data)


def test_init_from_unexisting_schema():
    fpath = TEST_PATH / "IDONTEXIST.json"
    assert not Path(fpath).exists()
    with pytest.raises(Exception):
        JSONGrammar("toto", fpath)


@pytest.mark.usefixtures("tmp_wd")
def test_write_schema():
    g = JSONGrammar(name="toto")
    fpath = "out_test.json"
    g.initialize_from_base_dict(typical_data_dict={"X": 1})
    g.write_schema(fpath)
    assert Path(fpath).exists()


def test_set_item():
    g = get_indict_grammar()
    g.set_item_value("Mach", {"type": "string"})
    with pytest.raises(InvalidDataException):
        g.load_data(get_indict())
    data = get_indict()
    data["Mach"] = "1"
    g.load_data(data)

    with pytest.raises(ValueError):
        g.set_item_value("unknown", {"type": "string"})


@pytest.mark.parametrize(
    "infile", ["grammar_test1.json", "grammar_test2.json", "grammar_test3.json"]
)
def test_to_simple_grammar_names(infile):
    grammar = JSONGrammar(infile, schema_file=TEST_PATH / infile)
    simp = grammar.to_simple_grammar()
    assert sorted(simp.get_data_names()) == sorted(grammar.get_data_names())
    assert grammar.name == simp.name


def test_to_simple_grammar_number():
    grammar = JSONGrammar("number", schema_file=TEST_PATH / "grammar_test1.json")
    simp = grammar.to_simple_grammar()
    assert simp.data_types == [Number]

    simp.load_data({"X": 1.0})
    simp.load_data({"X": 1})
    simp.load_data({"X": 1j})

    with pytest.raises(InvalidDataException):
        simp.load_data({})

    with pytest.raises(InvalidDataException):
        simp.load_data({"X": "X"})


def test_to_simple_grammar_array_number():
    grammar = JSONGrammar("number", schema_file=TEST_PATH / "grammar_test3.json")
    simp = grammar.to_simple_grammar()

    if PY2:
        # workaround for genson that uses unordered dict
        assert set(simp.data_types) == {Number, ndarray, None}
    else:
        assert simp.data_types == [Number, ndarray, None]

    simp.load_data({"X": 1, "Y": array([1.0]), "Z": 10})
    simp.load_data({"X": 1.0, "Y": array([1]), "Z": array([10])})
    simp.load_data({"X": 1j, "Y": array([1.0j]), "Z": 1j})

    with pytest.raises(InvalidDataException):
        simp.load_data({"X": 1j, "Y": 1.0})


def test_to_simple_grammar_string_array(caplog):
    """Check that a warning message is logged when a JSONGrammar has a string array."""
    grammar = JSONGrammar(
        "grammar_with_string_array",
        schema_file=TEST_PATH / "grammar_test4.json",
    )
    caplog.set_level(logging.WARNING)
    grammar.to_simple_grammar()
    expected = (
        "Unsupported type 'string' in JSONGrammar 'grammar_with_string_array' "
        "for property 'X' in conversion to simple grammar."
    )
    assert expected in caplog.text


def test_is_type_array_errors():
    fpath = TEST_PATH / "grammar_test1.json"
    gram = JSONGrammar(name="toto", schema_file=fpath)
    with pytest.raises(ValueError, match="is not in the grammar"):
        gram.is_type_array("IDONTEXIST")


def test_properties_dict():
    gram = JSONGrammar("")
    with pytest.raises(ValueError, match="Schema has no properties"):
        gram.properties_dict


def test_description_with_anyof_types_parameter():
    """Test that the description of a parameter is correctly taken into account when
    ``anyOf`` tag is used to define several types."""

    description = {
        "X": "description of X",
        "Y": "description of Y",
        "Z": "description of Z",
    }

    json_file = Path(__file__).parent / "data" / "grammar_test3.json"
    json_grammar = JSONGrammar("my_json", json_file, descriptions=description)
    json_dict = json_grammar.schema.to_dict()
    properties = json_dict["properties"]

    assert properties["X"]["description"] == description["X"]
    assert properties["Y"]["description"] == description["Y"]
    assert properties["Z"]["anyOf"][0]["description"] == description["Z"]
    assert properties["Z"]["anyOf"][1]["description"] == description["Z"]


@pytest.mark.parametrize(
    "file_name, element, expected",
    [("grammar_test1.json", "X", True), ("grammar_test3.json", "Z", False)],
)
def test_required(file_name, element, expected):
    """Test that the required data names are handled correctly.

    Args:
        file_name: The json grammar test file.
        element: The element of the grammar to test.
        expected: The expected boolean to get.
    """
    fpath = TEST_PATH / file_name
    gram = JSONGrammar(name="toto", schema_file=fpath)
    assert expected == gram.is_required(element)

    simple_g = gram.to_simple_grammar()
    assert expected == simple_g.is_required(element)
