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

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from gemseo.caches.factory import CacheFactory
from gemseo.core import base_factory
from gemseo.core.base_factory import BaseFactory
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.utils.base_multiton import BaseABCMultiton

# test data
DATA = Path(__file__).parent / "data/factory"


class MultitonFactory(metaclass=BaseABCMultiton):
    _CLASS = int
    _PACKAGE_NAMES = ()


def test_multiton() -> None:
    """Verify the multiton behavior."""

    class MultitonFactory2(metaclass=BaseABCMultiton):
        _CLASS = str
        _PACKAGE_NAMES = ()

    a = MultitonFactory()
    assert a is MultitonFactory()
    assert a is not MultitonFactory2()


def test_multiton_cache_clear() -> None:
    """Verify the clearing of the cache of the multiton."""
    # The cache is not empty because of the Multiton* classes declared in the module.
    MultitonFactory()
    assert BaseABCMultiton._BaseMultiton__keys_to_class_instances
    BaseFactory.clear_cache()
    assert not BaseABCMultiton._BaseMultiton__keys_to_class_instances


def test_print_configuration(reset_factory) -> None:
    """Verify the string representation of a factory."""
    factory = MDOFormulationFactory()

    # check table header
    header_patterns = [
        r"^\+-+\+$",
        r"^\|\s+BaseMDOFormulation\s+\|$",
        r"^\+-+\+-+\+-+\+$",
        r"^\|\s+Module\s+\|\s+Is available\?\s+\|\s+Purpose or error " r"message\s+\|$",
    ]

    for pattern, line in zip(header_patterns, repr(factory).split("\n")):
        assert re.findall(pattern, line)

    # check table body
    formulations = ["BiLevel", "BiLevelBCD", "DisciplinaryOpt", "IDF", "MDF"]

    for formulation in formulations:
        pattern = f"\\|\\s+{formulation}\\s+\\|\\s+Yes\\s+\\|.+\\|"
        assert re.findall(pattern, repr(factory))


def test_create_error(reset_factory) -> None:
    """Verify that Factory.create catches bad sub-classes."""
    factory = MDOFormulationFactory()
    msg = "The class dummy is not available; the available ones are: "
    with pytest.raises(ImportError, match=msg):
        factory.create("dummy", "dummy", "dummy", "dummy")


def test_create_bad_option(reset_factory) -> None:
    """Verify that a Factory.create catches bad options."""
    factory = MDOFormulationFactory()
    with pytest.raises(TypeError):
        factory.create("MDF", bad_option="bad_value")


@pytest.mark.parametrize(
    "formulation_name", ["BiLevel", "DisciplinaryOpt", "IDF", "MDF"]
)
def test_parse_docstrings(reset_factory, tmp_wd, formulation_name) -> None:
    factory = MDOFormulationFactory()
    formulations = factory.class_names

    assert len(formulations) > 3

    doc = factory.get_options_doc(formulation_name)
    assert "disciplines" in doc
    assert "maximize_objective" not in doc

    opt_vals = factory.get_default_option_values(formulation_name)
    assert len(opt_vals) >= 1

    grammar = factory.get_options_grammar(formulation_name, write_schema=True)
    file_name = f"{grammar.name}.json"
    assert (
        Path(file_name).read_text().split()
        == Path(DATA / file_name).read_text().split()
    )

    grammar.validate(opt_vals)

    opt_doc = factory.get_options_doc(formulation_name)
    data_names = grammar.keys()
    assert "name" not in data_names
    assert "design_space" not in data_names
    assert "objective_name" not in data_names
    for item in data_names:
        assert item in opt_doc


def test_ext_plugin_syspath_is_first(reset_factory, tmp_path) -> None:
    """Verify that plugins are not discovered from the first path in sys.path."""
    # This test requires to use subprocess such that python can
    # be called from a temporary directory that will be automatically
    # inserted first in sys.path.
    shutil.copytree(DATA, tmp_path, dirs_exist_ok=True)

    # Create a module that shall fail to load the plugin.
    code = """
from gemseo.formulations.factory import MDOFormulationFactory
assert 'DummyBiLevel' in MDOFormulationFactory().class_names
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


def test_ext_plugin_gemseo_path(monkeypatch, reset_factory) -> None:
    """Verify that plugins are discovered from the GEMSEO_PATH env variable."""
    monkeypatch.setenv("GEMSEO_PATH", DATA)
    # There could be more classes available with the plugins
    assert MDOFormulationFactory().is_available("DummyBiLevel")


def test_ext_plugin_gemseo_path_bad_package(monkeypatch, reset_factory) -> None:
    """Verify that plugins are discovered from the GEMSEO_PATH env variable."""
    monkeypatch.setenv("GEMSEO_PATH", DATA / "gemseo_dummy_plugins")
    assert MDOFormulationFactory().failed_imports["bad"] == "division by zero"


def test_wanted_classes_with_entry_points(monkeypatch, reset_factory) -> None:
    """Verify that the classes found are the expected ones."""

    class DummyEntryPoint:
        name = "dummy-name"
        value = "dummy_formulations"

    def entry_points(group):
        return [DummyEntryPoint]

    monkeypatch.setattr(base_factory, "entry_points", entry_points)
    monkeypatch.syspath_prepend(DATA / "gemseo_dummy_plugins")

    # There could be more classes available with the plugins
    assert MDOFormulationFactory().is_available("DummyBiLevel")


def test_get_library_name(reset_factory) -> None:
    """Verify that the library names found are the expected ones."""
    factory = MDOFormulationFactory()
    assert factory.get_library_name("MDF") == "gemseo"


def test_concrete_classes() -> None:
    """Check that the factory considers only the concrete classes."""
    factory = MDOFormulationFactory()
    assert factory.is_available("BiLevel")
    assert not factory.is_available(factory._CLASS)


def test_str() -> None:
    """Verify str() on a factory."""
    factory = MDOFormulationFactory()
    assert str(factory) == "Factory of BaseMDOFormulation objects"


def test_positional_arguments() -> None:
    """Check that BaseFactory supports the positional arguments."""
    cache = CacheFactory().create("SimpleCache", tolerance=0.1)
    assert cache.tolerance == 0.1
