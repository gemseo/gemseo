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
"""Tests for the read-only mapping."""

from __future__ import annotations

import pickle

import pytest

from gemseo.util.read_only_mapping import ReadOnlyMapping


class MyReadOnlyMapping(ReadOnlyMapping): ...


def test_repr() -> None:
    """Check the string representation of a ReadOnlyMapping."""
    mapping = MyReadOnlyMapping({"a": 1, "b": 2})
    assert repr(mapping) == "MyReadOnlyMapping({'a': 1, 'b': 2})"


def test_read_access() -> None:
    """Check the read access to the wrapped dictionary."""
    mapping = ReadOnlyMapping({"a": 1, "b": 2})
    assert mapping["a"] == 1
    assert len(mapping) == 2
    assert list(mapping) == ["a", "b"]
    assert dict(mapping) == {"a": 1, "b": 2}


def test_is_a_live_view() -> None:
    """Check that the mapping reflects the mutations of the backing dictionary."""
    data = {"a": 1}
    mapping = ReadOnlyMapping(data)
    data["b"] = 2
    assert mapping["b"] == 2


def test_write_is_forbidden() -> None:
    """Check that insertion, update and deletion are forbidden."""
    mapping = ReadOnlyMapping({"a": 1})
    with pytest.raises(TypeError):
        mapping["b"] = 2
    with pytest.raises(TypeError):
        del mapping["a"]


def test_pickle_round_trip() -> None:
    """Check that the mapping survives a pickle round trip."""
    mapping = ReadOnlyMapping({"a": 1, "b": 2})
    restored = pickle.loads(pickle.dumps(mapping))
    assert dict(restored) == {"a": 1, "b": 2}
