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

from __future__ import absolute_import, division, print_function, unicode_literals

import os
import unittest
from builtins import isinstance, str
from os.path import dirname, exists, join

from future import standard_library
from numpy import array, ndarray

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.grammar import InvalidDataException, SimpleGrammar
from gemseo.core.json_grammar import JSONGrammar
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()
configure_logger(SOFTWARE_NAME)


class Test_JSONGrammar(unittest.TestCase):
    def get_indict(self):
        return {
            "Mach": 1.0,
            "Cl": 2.0,
            "Turbulence_model": "SA",
            "Navier-Stokes": True,
            "bounds": [1.0, 2.0],
        }

    def get_indict_grammar(self):
        g = JSONGrammar(name="basic")
        g.initialize_from_base_dict(
            typical_data_dict=self.get_indict(), schema_file=None
        )
        return g

    @link_to("Req-WF-7")
    def test_instanciation(self):

        JSONGrammar(name="empty")

    def test_basic_grammar_init_from_dict(self):

        g = self.get_indict_grammar()
        g.load_data(self.get_indict())
        g_str = str(g)
        assert "properties" in g_str
        assert "required" in g_str
        for k in self.get_indict():
            assert k in g_str

        array_dct = self.get_indict()
        array_dct["bounds"] = array(array_dct["bounds"])
        g2 = JSONGrammar(name="basic2")
        g2.initialize_from_base_dict(typical_data_dict=array_dct, schema_file=None)

        for k in g.get_data_names():
            assert k in g2.get_data_names()

        g2_str = str(g2)
        assert "properties" in g2_str
        assert "required" in g2_str
        for k in self.get_indict():
            assert k in g2_str

    def test_nan(self):
        grammar = JSONGrammar("test_gram")
        typical_data_dict = {"no_nan": [1]}
        grammar.initialize_from_base_dict(typical_data_dict)
        grammar.load_data({"no_nan": array([float(1)])})
        self.assertRaises(
            InvalidDataException, grammar.load_data, {"no_nan": array([float("nan")])}
        )
        self.assertRaises(
            InvalidDataException,
            grammar.load_data,
            {"no_nan": array([float("nan"), 1.0])},
        )

    #     def test_bench(self):
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

    def test_init_from_datanames(self):
        grammar = JSONGrammar("t")
        names = ["a", "b"]
        grammar.initialize_from_data_names(names)
        for name in names:
            assert name in grammar.get_data_names()

    def test_init_from_base_dict(self):
        grammar = JSONGrammar("test_gram")
        typical_data_dict = {"a": [1], "b": "b"}
        grammar.initialize_from_base_dict(
            typical_data_dict, schema_file=None, write_schema=True
        )

    def test_update_from(self):
        g = self.get_indict_grammar()
        self.assertRaises(Exception, g.update_from, {})
        ge = JSONGrammar(name="empty")
        ge.update_from(g)
        assert sorted(ge.get_data_names()) == sorted(g.get_data_names())

        gs = SimpleGrammar("b")
        self.assertRaises(TypeError, ge.update_from_if_not_in, gs, gs)

    def test_update_from_if_not_in(self):

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
            schema_file=None,
            description_dict=description_dict_1,
        )

        g2 = JSONGrammar(name="basic")
        g2.initialize_from_base_dict(typical_data_dict=dct_2, schema_file=None)

        ge = JSONGrammar(name="empty")
        ge.update_from_if_not_in(g1, g2)

        assert sorted(ge.get_data_names()) == sorted(["bounds", "Navier-Stokes"])

        assert (
            ge.schema.to_dict()["properties"]["Navier-Stokes"]["description"]
            is not None
        )

    def test_update_from_dict(self):
        g1 = JSONGrammar("g1")
        typical_data_dict = {"max_iter": 1}
        g1.initialize_from_base_dict(
            typical_data_dict, schema_file="test.json", write_schema=True
        )

        g2 = JSONGrammar(name="basic_str", schema=g1.schema)
        assert "max_iter" in g2.properties
        os.remove("test.json")

    def test_update_descr(self):
        g1 = JSONGrammar("g1")
        key = "max_iter"
        g1.initialize_from_base_dict({key: 1})
        descr = "max number of iterations"
        g1.add_description({key: descr})
        assert g1.schema.to_dict()["properties"][key]["description"] == descr

    @link_to("Req-WF-7")
    def test_invalid_data(self):
        fpath = join(dirname(__file__), "grammar_test1.json")
        assert exists(fpath)
        gram = JSONGrammar(name="toto", schema_file=fpath)
        gram.load_data({"X": 1})
        gram.load_data({"X": 1.1})
        self.assertRaises(InvalidDataException, gram.load_data, {})
        self.assertRaises(InvalidDataException, gram.load_data, {"Y": 2})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": "1"})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": "/opt"})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": array([1.0])})
        self.assertRaises(InvalidDataException, gram.load_data, 1)
        self.assertRaises(InvalidDataException, gram.load_data, "X")

    def test_init_from_unexisting_schema(self):
        fpath = join(dirname(__file__), "IDONTEXIST.json")
        assert not exists(fpath)
        self.assertRaises(Exception, JSONGrammar, "toto", fpath)

    def test_write_schema(self):
        g = JSONGrammar(name="toto")
        fpath = join(dirname(__file__), "out_test.json")
        g.initialize_from_base_dict(
            typical_data_dict={"X": 1}, schema_file=fpath, write_schema=True
        )
        assert exists(fpath)
        os.remove(fpath)

    def test_set_item(self):
        g = self.get_indict_grammar()
        g.set_item_value("Mach", {"type": "string"})
        self.assertRaises(InvalidDataException, g.load_data, self.get_indict())
        data = self.get_indict()
        data["Mach"] = "1"
        g.load_data(data)

        self.assertRaises(ValueError, g.set_item_value, "unknown", {"type": "string"})
