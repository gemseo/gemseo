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

from gemseo.core.grammars.defaults import Defaults
from gemseo.core.grammars.simple_grammar import SimpleGrammar

exclude_names = pytest.mark.parametrize(
    "excluded_names",
    [
        (),
        ["dummy"],
        ["name"],
    ],
)


@pytest.fixture
def defaults() -> Defaults:
    """Return a Defaults object."""
    return Defaults(
        SimpleGrammar("g", names_to_types={"name": None, "other_name": None}), {}
    )


def test_init() -> None:
    """Verify the initialization from an existing dictionary."""
    data = {"name": 0}
    defaults = Defaults(
        SimpleGrammar("g", names_to_types={"name": None}),
        data,
    )
    assert defaults == data


def test_init_error() -> None:
    """Verify the error when initializing from an existing dictionary."""
    msg = "The name bad-name is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        Defaults(SimpleGrammar("g"), {"bad-name": 0})


def test_len(defaults: Defaults) -> None:
    """Verify len."""
    assert len(defaults) == 0
    defaults["name"] = 0
    assert len(defaults) == 1


def test_iter(defaults: Defaults) -> None:
    """Verify iter."""
    assert list(iter(defaults)) == []
    defaults["name"] = 0
    assert list(iter(defaults)) == ["name"]


def test_delitem(defaults: Defaults) -> None:
    """Verify delete."""
    # Non existing name.
    with pytest.raises(KeyError, match="dummy"):
        del defaults["dummy"]

    # Existing name.
    defaults["name"] = 0
    del defaults["name"]
    assert "name" not in defaults


def test_getitem(defaults: Defaults) -> None:
    """Verify setitem."""
    # Non existing name.
    with pytest.raises(KeyError, match="dummy"):
        defaults["dummy"]

    # Existing name.
    defaults["name"] = 0
    assert defaults["name"] == 0


def test_setitem(defaults: Defaults) -> None:
    """Verify setitem."""
    # Set without error.
    defaults["name"] = 0
    assert defaults["name"] == 0

    # Non existing name.
    msg = r"The name dummy is not in the grammar\."
    with pytest.raises(KeyError, match=msg):
        defaults["dummy"] = 0
