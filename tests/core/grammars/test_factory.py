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
from __future__ import annotations

import sys
import types

from gemseo.core.discipline import Discipline
from gemseo.core.grammars.factory import GRAMMAR_FACTORY
from gemseo.core.grammars.json import JSONGrammar
from gemseo.core.grammars.simple import SimpleGrammar
from gemseo.utils.testing.helpers import assert_exception


def test_is_available(reset_factory) -> None:
    assert GRAMMAR_FACTORY.is_available("JSONGrammar")


def test_is_available_unknown(reset_factory) -> None:
    assert not GRAMMAR_FACTORY.is_available("UnknownGrammar")


def test_class_names(reset_factory) -> None:
    assert GRAMMAR_FACTORY.class_names == [
        "JSONGrammar",
        "PydanticGrammar",
        "SimpleGrammar",
        "SimplerGrammar",
    ]


def test_create(reset_factory) -> None:
    grammar_name = "my_grammar"
    grammar = GRAMMAR_FACTORY.create("SimpleGrammar", name=grammar_name)
    assert isinstance(grammar, SimpleGrammar)
    assert grammar.name == grammar_name


def test_create_unknown(reset_factory, snapshot) -> None:
    with assert_exception(ImportError, snapshot):
        GRAMMAR_FACTORY.create("UnknownGrammar", name="g")


def test_create_search_file_non_json(reset_factory, snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        GRAMMAR_FACTORY.create("SimpleGrammar", name="g", search_file=True)


def test_create_search_file_missing_discipline_class(reset_factory, snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        GRAMMAR_FACTORY.create("JSONGrammar", name="g", search_file=True)


def test_create_search_file(reset_factory, tmp_path) -> None:
    """Verify that a JSON grammar file is located and loaded via `search_file`."""

    class DummyDiscipline(Discipline):
        """A stand-in discipline used only for its class name."""

    source = JSONGrammar("g")
    source.update_from_types({"x": float})
    source.to_file(tmp_path / f"{DummyDiscipline.__name__}_input.json")
    grammar = GRAMMAR_FACTORY.create(
        "JSONGrammar",
        name="g",
        search_file=True,
        discipline_class=DummyDiscipline,
        directory_path=tmp_path,
        file_name_suffix="input",
    )
    assert isinstance(grammar, JSONGrammar)
    assert "x" in grammar


def test_create_search_file_default_directory(
    reset_factory, tmp_path, monkeypatch
) -> None:
    """Verify that the module directory is used when no directory is given."""

    class DummyDiscipline(Discipline):
        """A stand-in discipline used only for its class name."""

    fake_module = types.ModuleType("fake_grammar_mod")
    fake_module.__file__ = str(tmp_path / "mod.py")
    monkeypatch.setitem(sys.modules, "fake_grammar_mod", fake_module)
    DummyDiscipline.__module__ = "fake_grammar_mod"

    source = JSONGrammar("g")
    source.update_from_types({"x": float})
    source.to_file(tmp_path / f"{DummyDiscipline.__name__}_input.json")

    grammar = GRAMMAR_FACTORY.create(
        "JSONGrammar",
        name="g",
        search_file=True,
        discipline_class=DummyDiscipline,
        file_name_suffix="input",
    )
    assert isinstance(grammar, JSONGrammar)
    assert "x" in grammar


def test_create_search_file_parent_class(reset_factory, tmp_path) -> None:
    """Verify that the grammar file of a parent class is found."""

    class Parent(Discipline):
        """A stand-in discipline used only for its class name."""

    class Child(Parent):
        """A discipline without its own grammar file."""

    source = JSONGrammar("g")
    source.update_from_types({"x": float})
    source.to_file(tmp_path / f"{Parent.__name__}_input.json")

    grammar = GRAMMAR_FACTORY.create(
        "JSONGrammar",
        name="g",
        search_file=True,
        discipline_class=Child,
        directory_path=tmp_path,
        file_name_suffix="input",
    )
    assert isinstance(grammar, JSONGrammar)
    assert "x" in grammar


def test_create_search_file_not_found(reset_factory, tmp_path, snapshot) -> None:
    """Verify the error when no grammar file is found."""

    class DummyDiscipline(Discipline):
        """A stand-in discipline used only for its class name."""

    with assert_exception(FileNotFoundError, snapshot):
        GRAMMAR_FACTORY.create(
            "JSONGrammar",
            name="g",
            search_file=True,
            discipline_class=DummyDiscipline,
            directory_path=tmp_path,
            file_name_suffix="input",
        )
