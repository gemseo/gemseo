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
from pandas import DataFrame

from gemseo.core.grammars.defaults import Defaults
from gemseo.core.grammars.simple_grammar import SimpleGrammar

from ..test_discipline_data import to_df_key

exclude_names = pytest.mark.parametrize(
    "excluded_names",
    [
        (),
        ["dummy"],
        ["name"],
    ],
)


@pytest.fixture()
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


def test_contains(defaults: Defaults) -> None:
    """Verify contains."""
    assert "name" not in defaults
    defaults["name"] = 0
    assert "name" in defaults


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


@exclude_names
def test_update(defaults, excluded_names) -> None:
    """Verify update."""
    defaults_before = dict(defaults)
    other_defaults = {"name": 1}
    defaults.update(other_defaults, exclude=excluded_names)
    for name, value in defaults.items():
        if name in excluded_names:
            assert value == defaults_before[name]
        else:
            assert value == other_defaults[name]


def test_rename(defaults: Defaults) -> None:
    """Verify the renaming."""
    defaults["name"] = 0

    # With an existing name.
    defaults.rename("name", "other_name")
    assert "name" not in defaults
    assert defaults["other_name"] == 0

    # With a non-existing name.
    assert "dummy" not in defaults
    defaults.rename("dummy", "foo")


def test_restrict(defaults: Defaults) -> None:
    """Verify the restriction."""
    defaults["name"] = 0
    defaults["other_name"] = 0
    defaults.restrict("name", "non-existing-name")
    assert defaults.keys() == {"name"}


def test_setitem(defaults: Defaults) -> None:
    """Verify setitem."""
    # Set without error.
    defaults["name"] = 0
    assert defaults["name"] == 0

    # Non existing name.
    msg = r"The name dummy is not in the grammar\."
    with pytest.raises(KeyError, match=msg):
        defaults["dummy"] = 0


def test_setitem_dataframe(defaults: Defaults) -> None:
    """Verify setitem with a DataFrame."""
    defaults = Defaults(
        SimpleGrammar("g", names_to_types={"name~column": DataFrame}), {}
    )
    # Set without error.
    defaults[to_df_key("name", "column")] = [1.0]
    assert defaults["name"].equals(DataFrame(data={"column": [1.0]}))
    defaults["name"]["column"] = 2.0
    assert defaults["name"].equals(DataFrame(data={"column": [2.0]}))
    defaults["name"] = DataFrame(data={"column": [3.0]})
    assert defaults["name"].equals(DataFrame(data={"column": [3.0]}))

    # Non-existing column.
    msg = rf"The name name{defaults.SEPARATOR}dummy is not in the grammar\."
    with pytest.raises(KeyError, match=msg):
        defaults[to_df_key("name", "dummy")] = 0

    # Non-existing columns.
    msg = (
        rf"The names name{defaults.SEPARATOR}dummy1, "
        rf"name{defaults.SEPARATOR}dummy2 are not in the grammar\."
    )
    with pytest.raises(KeyError, match=msg):
        defaults["name"] = DataFrame(data={"dummy1": [0.0], "dummy2": [0.0]})


def test_copy(defaults: Defaults) -> None:
    """Verify copy."""
    defaults["name"] = 0
    copy = defaults.copy()
    assert copy.keys() == defaults.keys()
    assert copy["name"] is defaults["name"]
