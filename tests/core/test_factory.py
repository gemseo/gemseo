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
from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest

from gemseo.caches.cache_factory import CacheFactory
from gemseo.core.base_factory import BaseFactory
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.utils.base_multiton import BaseABCMultiton

# test data
DATA = Path(__file__).parent / "data/factory"


class MultitonFactory(metaclass=BaseABCMultiton):
    _CLASS = int
    _MODULE_NAMES = ()


def test_multiton():
    """Verify the multiton behavior."""

    class MultitonFactory2(metaclass=BaseABCMultiton):
        _CLASS = str
        _MODULE_NAMES = ()

    a = MultitonFactory()
    assert a is MultitonFactory()
    assert a is not MultitonFactory2()


def test_multiton_cache_clear():
    """Verify the clearing of the cache of the multiton."""
    # The cache is not empty because of the Multiton* classes declared in the module.
    MultitonFactory()
    assert BaseABCMultiton._BaseMultiton__keys_to_class_instances
    BaseFactory.clear_cache()
    assert not BaseABCMultiton._BaseMultiton__keys_to_class_instances


def test_print_configuration(reset_factory):
    """Verify the string representation of a factory."""
    factory = MDOFormulationsFactory()

    # check table header
    header_patterns = [
        r"^\+-+\+$",
        r"^\|\s+MDOFormulation\s+\|$",
        r"^\+-+\+-+\+-+\+$",
        r"^\|\s+Module\s+\|\s+Is available\?\s+\|\s+Purpose or error " r"message\s+\|$",
    ]

    for pattern, line in zip(header_patterns, repr(factory).split("\n")):
        assert re.findall(pattern, line)

    # check table body
    formulations = ["BiLevel", "DisciplinaryOpt", "IDF", "MDF"]

    for formulation in formulations:
        pattern = f"\\|\\s+{formulation}\\s+\\|\\s+Yes\\s+\\|.+\\|"
        assert re.findall(pattern, repr(factory))


def test_create_error(reset_factory):
    """Verify that Factory.create catches bad sub-classes."""
    factory = MDOFormulationsFactory()
    msg = "The class dummy is not available; the available ones are: "
    with pytest.raises(ImportError, match=msg):
        factory.create("dummy", "dummy", "dummy", "dummy")


def test_create_bad_option(reset_factory):
    """Verify that a Factory.create catches bad options."""
    factory = MDOFormulationsFactory()
    with pytest.raises(TypeError):
        factory.create("MDF", bad_option="bad_value")


@pytest.mark.parametrize(
    "formulation_name", ["BiLevel", "DisciplinaryOpt", "IDF", "MDF"]
)
def test_parse_docstrings(reset_factory, tmp_wd, formulation_name):
    factory = MDOFormulationsFactory()
    formulations = factory.class_names

    assert len(formulations) > 3

    doc = factory.get_options_doc(formulation_name)
    assert "disciplines" in doc
    assert "maximize_objective" in doc

    opt_vals = factory.get_default_option_values(formulation_name)
    assert len(opt_vals) >= 1

    grammar = factory.get_options_grammar(formulation_name, write_schema=True)
    file_name = f"{grammar.name}.json"
    assert (
        Path(DATA / file_name).read_text().split()
        == Path(file_name).read_text().split()
    )

    grammar.validate(opt_vals)

    opt_doc = factory.get_options_doc(formulation_name)
    data_names = grammar.keys()
    assert "name" not in data_names
    assert "design_space" not in data_names
    assert "objective_name" not in data_names
    for item in data_names:
        assert item in opt_doc


def test_ext_plugin_syspath_is_first(reset_factory, tmp_path):
    """Verify that plugins are not discovered from the first path in sys.path."""
    # This test requires to use subprocess such that python can
    # be called from a temporary directory that will be automatically
    # inserted first in sys.path.
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)

    # Create a module that shall fail to load the plugin.
    code = """
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
assert 'DummyBiLevel' in MDOFormulationsFactory().class_names
"""
    module_path = tmp_path / "module.py"
    module_path.write_text(code)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        subprocess.check_output(
            f"{sys.executable} {module_path}",
            shell=True,
            stderr=subprocess.STDOUT,
        )

    assert "AssertionError" in str(exc_info.value.output)


def test_ext_plugin_gemseo_path(monkeypatch, reset_factory):
    """Verify that plugins are discovered from the GEMSEO_PATH env variable."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    # There could be more classes available with the plugins
    assert "DummyBiLevel" in MDOFormulationsFactory().class_names


def test_ext_plugin_gemseo_path_bad_package(monkeypatch, reset_factory):
    """Verify that plugins are discovered from the GEMSEO_PATH env variable."""
    monkeypatch.setenv("GEMSEO_PATH", DATA / "gemseo_dummy_plugins")
    assert MDOFormulationsFactory().failed_imports == {"bad": "division by zero"}


def test_wanted_classes_with_entry_points(monkeypatch, reset_factory):
    """Verify that the classes found are the expected ones."""

    class DummyEntryPoint:
        name = "dummy-name"
        value = "dummy_formulations"

    def entry_points():
        return {BaseFactory.PLUGIN_ENTRY_POINT: [DummyEntryPoint]}

    monkeypatch.setattr(metadata, "entry_points", entry_points)
    monkeypatch.syspath_prepend(DATA / "gemseo_dummy_plugins")

    # There could be more classes available with the plugins
    assert "DummyBiLevel" in MDOFormulationsFactory().class_names


def test_get_library_name(reset_factory):
    """Verify that the library names found are the expected ones."""
    factory = MDOFormulationsFactory()
    assert factory.get_library_name("MDF") == "gemseo"


def test_concrete_classes():
    """Check that the factory considers only the concrete classes."""
    factory = MDOFormulationsFactory()
    assert "BiLevel" in factory.class_names
    assert factory._CLASS not in factory.class_names


def test_str():
    """Verify str() on a factory."""
    factory = MDOFormulationsFactory()
    assert str(factory) == "Factory of MDOFormulation objects"


def test_positional_arguments():
    """Check that BaseFactory supports the positional arguments."""
    cache = CacheFactory().create("SimpleCache", 0.1)
    assert cache.tolerance == 0.1


def test_creation_error(caplog):
    """Check that BaseFactory logs a message in the case of a creation error."""
    with pytest.raises(TypeError):
        CacheFactory().create("SimpleCache", 1, 2, 3, a=2)

    record_tuple = caplog.record_tuples[0]
    assert record_tuple[1] == logging.ERROR
    assert record_tuple[2] == (
        "Failed to create class SimpleCache with positional arguments (1, 2, 3) "
        "and keyword arguments {'a': 2}."
    )
