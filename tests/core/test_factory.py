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

import re
import shutil
import subprocess
import sys

import pytest

from gemseo.core.factory import Factory
from gemseo.core.formulation import MDOFormulation
from gemseo.utils.py23_compat import PY2, Path, importlib_metadata

# test data
DATA = Path(__file__).parent / "data/factory"


def test_print_configuration(tmp_path, reset_factory):
    """Verify the string representation of a factory."""
    factory = Factory(MDOFormulation, ("gemseo.formulations",))

    # check table header
    header_patterns = [
        r"^\+-+\+$",
        r"^\|\s+MDOFormulation\s+\|$",
        r"^\+-+\+-+\+-+\+$",
        r"^\|\s+Module\s+\|\s+Is available \?\s+\|\s+Purpose or error "
        r"message\s+\|$",
    ]

    for pattern, line in zip(header_patterns, repr(factory).split("\n")):
        assert re.findall(pattern, line)

    # check table body
    formulations = ["BiLevel", "DisciplinaryOpt", "IDF", "MDF"]

    for formulation in formulations:
        pattern = "\\|\\s+{}\\s+\\|\\s+Yes\\s+\\|.+\\|".format(formulation)
        assert re.findall(pattern, repr(factory))


def test_unknown_class(reset_factory):
    """Verify that Factory catches bad classes."""
    msg = "Class to search must be a class!"
    with pytest.raises(TypeError, match=msg):
        Factory("UnknownClass")


def test_create_error(reset_factory):
    """Verify that Factory.create catches bad sub-classes."""
    factory = Factory(MDOFormulation)
    msg = "Class dummy is not available!\nAvailable ones are: "
    with pytest.raises(ImportError, match=msg):
        factory.create("dummy")


def test_create_bad_option(reset_factory):
    """Verify that a Factory.create catches bad options."""
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
            assert item in opt_doc


def test_ext_plugin_syspath(monkeypatch, reset_factory):
    """Verify that plugins are discovered from the python path."""
    monkeypatch.syspath_prepend(DATA)
    # Add a new dummy item in sys.path because the first item will be removed,
    # and monkeypatch can only prepend.
    monkeypatch.syspath_prepend("")
    # There could be more classes available with the plugins
    assert "DummyBiLevel" in Factory(MDOFormulation).classes


def test_ext_plugin_syspath_is_first(reset_factory, tmp_path):
    """Verify that plugins are not discovered from the first path in sys.path."""
    # This test requires to use subprocess such that python can
    # be called from a temporary directory that will be automatically
    # inserted first in sys.path.
    if sys.version_info < (3, 8):
        # dirs_exist_ok appeared in python 3.8
        tmp_path.rmdir()
        shutil.copytree(str(DATA), str(tmp_path))
    else:
        shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)

    # Create a module that shall fail to load the plugin.
    code = """
from gemseo.core.factory import Factory
from gemseo.core.formulation import MDOFormulation
assert 'DummyBiLevel' in Factory(MDOFormulation).classes
"""
    module_path = tmp_path / "module.py"
    module_path.write_text(code)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.check_output(
            "{} {}".format(sys.executable, module_path),
            shell=True,
            stderr=subprocess.STDOUT,
        )

    assert "AssertionError" in str(exc_info.value.output)


def test_ext_plugin_gems_path(monkeypatch, reset_factory):
    """Verify that plugins are discovered from the GEMS_PATH env variable."""
    monkeypatch.setenv("GEMS_PATH", DATA)
    # There could be more classes available with the plugins
    assert "DummyBiLevel" in Factory(MDOFormulation).classes


def test_ext_plugin_gemseo_path(monkeypatch, reset_factory):
    """Verify that plugins are discovered from the GEMSEO_PATH env variable."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    # There could be more classes available with the plugins
    assert "DummyBiLevel" in Factory(MDOFormulation).classes


def test_wanted_classes(monkeypatch, reset_factory):
    """Verify that the classes found are the expected ones."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    # There could be more classes available with the plugins
    assert "DummyBiLevel" in Factory(MDOFormulation).classes


@pytest.mark.skipif(PY2, reason="plugin entry points are not supported for Python 2")
def test_wanted_classes_with_entry_points(monkeypatch, reset_factory):
    """Verify that the classes found are the expected ones."""

    class DummyEntryPoint:
        name = "dummy-name"
        value = "dummy_formulations"

    def entry_points():
        return {Factory.PLUGIN_ENTRY_POINT: [DummyEntryPoint]}

    monkeypatch.setattr(importlib_metadata, "entry_points", entry_points)
    monkeypatch.syspath_prepend(DATA / "gemseo_dummy_plugins")

    # There could be more classes available with the plugins
    assert "DummyBiLevel" in Factory(MDOFormulation).classes
