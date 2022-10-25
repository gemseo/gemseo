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

import pytest
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.grammars.simple_grammar import SimpleGrammar


@pytest.fixture
def factory():
    return GrammarFactory()


def test_is_available(factory):
    assert factory.is_available("JSONGrammar")


def test_create(factory):
    grammar_name = "my_grammar"
    grammar = factory.create("SimpleGrammar", name=grammar_name)
    assert isinstance(grammar, SimpleGrammar)
    assert grammar.name == grammar_name


def test_grammars(factory):
    assert "SimpleGrammar" in factory.grammars
