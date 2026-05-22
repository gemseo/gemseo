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
# Antoine DECHAUME
from __future__ import annotations

import pickle
from pathlib import Path
from pathlib import PurePath
from pathlib import PurePosixPath
from pathlib import PureWindowsPath

import pytest

from gemseo.core.discipline.discipline_data import DisciplineData


def test_is_a_dict() -> None:
    """Verify DisciplineData behaves as a dict subclass."""
    data = DisciplineData({"a": 1, "b": 2})
    assert isinstance(data, dict)
    data["c"] = 3
    assert data == {"a": 1, "b": 2, "c": 3}


def test_getstate_converts_path_to_portable() -> None:
    """Verify __getstate__ casts ``Path`` values to OS-specific PurePath."""
    path = Path("dummy")
    data = DisciplineData({"path": path, "number": 1, "string": "x"})
    state = data.__getstate__()
    assert isinstance(state["path"], PurePath)
    assert state["number"] == 1
    assert state["string"] == "x"


def test_getstate_does_not_mutate_data() -> None:
    """Verify __getstate__ leaves the original mapping unchanged."""
    path = Path("dummy")
    data = DisciplineData({"path": path})
    data.__getstate__()
    assert data["path"] is path


@pytest.mark.parametrize("path", [PurePosixPath("dummy"), PureWindowsPath("dummy")])
def test_setstate_converts_pure_path_to_path(path) -> None:
    """Verify __setstate__ converts any ``PurePath`` value back to ``Path``."""
    data = DisciplineData()
    data.__setstate__({"path": path, "number": 1})
    assert data["path"] == Path("dummy")
    assert isinstance(data["path"], Path)
    assert data["number"] == 1


@pytest.mark.parametrize("path", [PurePosixPath("dummy"), PureWindowsPath("dummy")])
def test_pickle_roundtrip(path) -> None:
    """Verify a full pickle round-trip restores a ``Path`` value."""
    data = DisciplineData({"path": path, "number": 1})
    restored = pickle.loads(pickle.dumps(data))
    assert restored["path"] == Path("dummy")
    assert isinstance(restored["path"], Path)
    assert restored["number"] == 1
