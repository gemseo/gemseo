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

from __future__ import absolute_import, division, unicode_literals

import os
import re

import pytest

from gemseo.core.factory import Factory
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.py23_compat import Path
from gemseo.utils.singleton import SingleInstancePerAttributeEq

# test data
DATA = Path(__file__).parent / "data"


@pytest.fixture
def reset_factory():
    """Reset the factory singletons."""
    yield
    SingleInstancePerAttributeEq.instances.clear()


def test_unknown_internal_modules_paths(reset_factory):
    Factory(MDOFormulation)


def test_print_configuration(tmp_path, reset_factory):
    factory = Factory(MDOFormulation, ("gemseo.formulations",))

    # check table header
    header_patterns = [
        r"^\+-+\+$",
        r"^\|\s+MDOFormulation\s+\|$",
        r"^\+-+\+-+\+-+\+$",
        r"^\|\s+Module\s+\|\s+Is available \?\s+\|\s+Purpose or error "
        r"message\s+\|$",
    ]

    for pattern, line in zip(header_patterns, str(factory).split("\n")):
        assert re.findall(pattern, line)

    # check table body
    formulations = ["BiLevel", "DisciplinaryOpt", "IDF", "MDF"]

    for formulation in formulations:
        pattern = "\\|\\s+{}\\s+\\|\\s+Yes\\s+\\|.+\\|".format(formulation)
        assert re.findall(pattern, str(factory))


def test_unknown_class(reset_factory):
    msg = "Class to search must be a class!"
    with pytest.raises(TypeError, match=msg):
        Factory("UnknownClass")


def test_create_error(reset_factory):
    msg = (
        "Class dummy is not available!\n"
        "Available ones are: BiLevel, DisciplinaryOpt, IDF, MDF"
    )
    factory = Factory(MDOFormulation)
    with pytest.raises(ImportError, match=msg):
        factory.create("dummy")


def test_bad_option(reset_factory):
    factory = Factory(MDOFormulation, ("gemseo.formulations",))
    with pytest.raises(TypeError):
        factory.create("MDF", bad_option="bad_value")


def test_parse_docstrings(reset_factory):
    factory = Factory(MDOFormulation, ("gemseo.formulations",))
    formulations = factory.classes

    assert len(formulations) > 3

    for form in formulations:
        doc = factory.get_options_doc(form)
        assert "disciplines" in doc
        assert "maximize_objective" in doc

        opt_vals = factory.get_default_options_values(form)
        assert len(opt_vals) >= 1

        grammar = factory.get_options_grammar(form)
        grammar.load_data(opt_vals, raise_exception=True)

        opt_doc = factory.get_options_doc(form)
        data_names = grammar.get_data_names()
        assert "name" not in data_names
        assert "design_space" not in data_names
        assert "objective_name" not in data_names
        for item in data_names:
            if item not in opt_doc:
                raise ValueError(
                    "Undocumented option "
                    + str(item)
                    + " in formulation : "
                    + str(form)
                )


def test_ext_plugin(monkeypatch, reset_factory):
    monkeypatch.syspath_prepend(DATA)
    factory = Factory(MDOFormulation)
    factory.create("DummyBiLevel")


def test_ext_gems_path(reset_factory):
    os.environ["GEMSEO_PATH"] = str(DATA)
    factory = Factory(MDOFormulation)
    factory.create("DummyBiLevel")
    del os.environ["GEMSEO_PATH"]


def test_ext_gemseo_path(reset_factory):
    os.environ["GEMSEO_PATH"] = str(DATA)
    factory = Factory(MDOFormulation)
    factory.create("DummyBiLevel")
    del os.environ["GEMSEO_PATH"]
