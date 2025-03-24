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

from typing import TYPE_CHECKING

import pytest

from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.grammars.required_names import RequiredNames

if TYPE_CHECKING:
    from gemseo.core.grammars.base_grammar import BaseGrammar

FACTORY = GrammarFactory()


@pytest.fixture(params=tuple(FACTORY.class_names))
def grammar(request) -> BaseGrammar:
    """Return a grammar with an element named ``name``."""
    grammar = FACTORY.create(request.param, name="g")
    grammar.update_from_names(["name"])
    return grammar


def test_init(grammar):
    """Verify init."""
    rn = RequiredNames(grammar)
    assert not rn

    rn = RequiredNames(grammar, names=["name"])
    assert "name" in rn

    match = r"The name 'bad' is not in the grammar\."
    with pytest.raises(KeyError, match=match):
        RequiredNames(grammar, names=["bad"])


def test_add(grammar):
    """Verify add."""
    rn = RequiredNames(grammar)
    rn.add("name")

    match = r"The name 'bad' is not in the grammar\."
    with pytest.raises(KeyError, match=match):
        rn.add("bad")


def test_discard(grammar):
    """Verify discard."""
    rn = RequiredNames(grammar, names=["name"])
    rn.discard("name")
    assert "name" not in rn
    rn.discard("bad")


def test_contains(grammar):
    """Verify __contains__."""
    rn = RequiredNames(grammar, names=["name"])
    assert "name" in rn
    assert "bad" not in rn


def test_iter(grammar):
    """Verify __iter__."""
    rn = RequiredNames(grammar, names=["name"])
    assert set(iter(rn)) == set(rn)


def test_str(grammar):
    """Verify __str__."""
    rn = RequiredNames(grammar, names=["name"])
    assert str(rn) == "{'name'}"


def test_len(grammar):
    """Verify __len__."""
    rn = RequiredNames(grammar, names=["name"])
    assert len(rn) == 1


def test_from_iterable(grammar):
    """Verify _from_iterable."""
    rn = RequiredNames(grammar, names=["name"])
    rn | {"name"}

    match = r"The name 'bad' is not in the grammar\."
    with pytest.raises(KeyError, match=match):
        rn | {"bad"}


def test_get_names_difference(grammar):
    """Verify get_names_difference."""
    rn = RequiredNames(grammar, names=["name"])
    assert rn.get_names_difference(()) == {"name"}
    assert rn.get_names_difference(["dummy"]) == {"name"}
    assert rn.get_names_difference(["name"]) == set()
