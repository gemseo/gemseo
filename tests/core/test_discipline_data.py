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
from pathlib import PurePosixPath
from pathlib import PureWindowsPath

import pytest

from gemseo.core.discipline.discipline_data import DisciplineData


@pytest.mark.parametrize("path", [PurePosixPath("dummy"), PureWindowsPath("dummy")])
def test_serialization(path) -> None:
    """Verify serialization of path data."""
    data = {"path": path}
    assert pickle.loads(pickle.dumps(DisciplineData(data)))["path"] == Path("dummy")
