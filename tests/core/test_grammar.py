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
from builtins import super, zip

import numpy as np
from future import standard_library
from numpy import array

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.grammar import InvalidDataException, SimpleGrammar
from gemseo.core.json_grammar import JSONGrammar
from gemseo.third_party.junitxmlreq import link_to

standard_library.install_aliases()


configure_logger(SOFTWARE_NAME)


class Test_Grammar(unittest.TestCase):
    """ """

    def get_indict(self):
        """ """
        return {
            "Mach": 1.0,
            "Cl": 2.0,
            "Turbulence_model": "SA",
            "Navier-Stokes": True,
            "bounds": np.array([1.0, 2.0]),
        }

    @link_to("Req-WF-7", "Req-WF-5")
    def test_instanciation(self):
        """ """
        SimpleGrammar(name="empty")

    def get_base_grammar_from_inherit(self):
        """ """

        class myGrammar(SimpleGrammar):
            """ """

            def __init__(self, name):
                super(myGrammar, self).__init__(name)
                self.data_names = [
                    "Mach",
                    "Cl",
                    "Turbulence_model",
                    "Navier-Stokes",
                    "bounds",
                ]
                self.data_types = [float, float, type("str"), type(True), np.array]

        return myGrammar("CFD_inputs")

    def get_base_grammar_from_instanciation(self):
        """ """
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

    def get_base_grammar_from_base_dict(self):
        """ """
        my_grammar = SimpleGrammar("CFD_inputs")
        my_grammar.initialize_from_base_dict(self.get_indict())
        return my_grammar

    def check_g1_in_g2(self, g1, g2):
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

    def check_g1_eq_g2(self, g1, g2):
        """

        :param g1: param g2:
        :param g2:

        """
        self.check_g1_in_g2(g1, g2)
        self.check_g1_in_g2(g2, g1)

    def test_inherit_vs_instanciation(self):
        """ """
        g1 = self.get_base_grammar_from_instanciation()
        g2 = self.get_base_grammar_from_inherit()
        self.check_g1_eq_g2(g1, g2)

    def test_dict_init_vs_instanciation(self):
        """ """
        g1 = self.get_base_grammar_from_instanciation()
        g2 = self.get_base_grammar_from_base_dict()
        self.check_g1_eq_g2(g1, g2)

    def test_update_from(self):
        """ """
        g = self.get_base_grammar_from_instanciation()
        self.assertRaises(Exception, g.update_from, {})
        ge = SimpleGrammar(name="empty")
        ge.update_from(g)
        n = len(ge.data_names)
        ge.update_from_if_not_in(ge, g)
        # Update again
        ge.update_from(g)
        # Check no updates are added again
        assert n == len(ge.data_names)
        g.data_types[-1] = "unknowntype"
        self.assertRaises(Exception, ge.update_from_if_not_in, *(ge, g))

        my_grammar = SimpleGrammar("toto")
        my_grammar.initialize_from_base_dict({"X": 2})
        ge.update_from_if_not_in(my_grammar, g)

        g_json = JSONGrammar("titi")
        self.assertRaises(TypeError, my_grammar.update_from_if_not_in, g_json, g_json)
        my_grammar.clear()

    @link_to("Req-WF-7", "Req-WF-5")
    def test_invalid_data(self):
        """ """
        gram = SimpleGrammar("dummy")
        gram.data_names = ["X"]
        gram.data_types = [float]

        gram.load_data({"X": 1.1})
        self.assertRaises(InvalidDataException, gram.load_data, {})
        self.assertRaises(InvalidDataException, gram.load_data, {"Mach": 2})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": "1"})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": "/opt"})
        self.assertRaises(InvalidDataException, gram.load_data, {"X": array([1.0])})
        self.assertRaises(InvalidDataException, gram.load_data, 1)
        self.assertRaises(InvalidDataException, gram.load_data, "X")

        gram = SimpleGrammar("dummy")
        gram.data_names = ["X"]
        gram.data_types = ["x"]
        self.assertRaises(TypeError, gram.load_data, {"X": 1.1})

    def test_default_data(self):
        """ """
        fpath = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "grammar_test2.json"
        )
        assert os.path.exists(fpath)
        gram = SimpleGrammar(name="toto")
        gram.defaults["X"] = 1
        d = gram.load_data({})
        assert "X" in d
        assert d["X"] == 1

    def test_is_alldata_exist(self):
        """ """
        g = self.get_base_grammar_from_instanciation()
        self.assertFalse(
            g.is_all_data_names_existing(
                [
                    "bidon",
                ]
            )
        )
        self.assertTrue(
            g.is_all_data_names_existing(
                [
                    "Mach",
                ]
            )
        )

    def test_get_type_of_data_error(self):
        """ """
        g = self.get_base_grammar_from_instanciation()
        self.assertRaises(
            Exception,
            lambda: g.get_type_of_data_named(
                [
                    "bidon",
                ]
            ),
        )
