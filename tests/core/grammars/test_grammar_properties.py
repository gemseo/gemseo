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

from gemseo.core.grammars.properties import GrammarProperties
from gemseo.core.grammars.simple import SimpleGrammar
from gemseo.utils.testing.helpers import assert_exception

exclude_names = pytest.mark.parametrize(
    "excluded_names",
    [
        (),
        ["dummy"],
        ["name"],
    ],
)


@pytest.fixture
def properties() -> GrammarProperties:
    """Return a GrammarProperties object."""
    return GrammarProperties(
        SimpleGrammar("g", name_to_type={"name": None, "other_name": None}), {}
    )


def test_init() -> None:
    """Verify the initialization from an existing dictionary."""
    data = {"name": 0}
    properties = GrammarProperties(
        SimpleGrammar("g", name_to_type={"name": None}),
        data,
    )
    assert properties == data


def test_init_error(snapshot) -> None:
    """Verify the error when initializing from an existing dictionary."""
    with assert_exception(KeyError, snapshot):
        GrammarProperties(SimpleGrammar("g"), {"bad-name": 0})


def test_len(properties: GrammarProperties) -> None:
    """Verify len."""
    assert len(properties) == 0
    properties["name"] = 0
    assert len(properties) == 1


def test_iter(properties: GrammarProperties) -> None:
    """Verify iter."""
    assert list(iter(properties)) == []
    properties["name"] = 0
    assert list(iter(properties)) == ["name"]


def test_delitem(properties: GrammarProperties, snapshot) -> None:
    """Verify delete."""
    # Non existing name.
    with assert_exception(KeyError, snapshot):
        del properties["dummy"]

    # Existing name.
    properties["name"] = 0
    del properties["name"]
    assert "name" not in properties


def test_getitem(properties: GrammarProperties, snapshot) -> None:
    """Verify setitem."""
    # Non existing name.
    with assert_exception(KeyError, snapshot):
        properties["dummy"]

    # Existing name.
    properties["name"] = 0
    assert properties["name"] == 0


def test_setitem(properties: GrammarProperties, snapshot) -> None:
    """Verify setitem."""
    # Set without error.
    properties["name"] = 0
    assert properties["name"] == 0

    # Non existing name.
    with assert_exception(KeyError, snapshot):
        properties["dummy"] = 0
