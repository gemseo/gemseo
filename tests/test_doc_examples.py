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

import runpy
from pathlib import Path
from shutil import copytree

import pytest

from gemseo.utils.global_configuration import _configuration

_DIRECTORY_MANAGER_EXAMPLE = "plot_howto_directory_manager.py"

EXAMPLE_PATHS = sorted(
    (
        path
        for path in Path(__file__, "..", "..", "docs", "examples")
        .resolve()
        .rglob("*.py")
        if path.name.startswith("plot_")
    ),
    key=lambda path: (path.name == _DIRECTORY_MANAGER_EXAMPLE, path.name),
)
"""The directory manager example is sorted last because it leaks global state."""


@pytest.fixture
def reset_global_configuration() -> None:
    """Reset the global configuration to defaults before each example."""
    _configuration.__init__()


@pytest.mark.doc_examples
@pytest.mark.parametrize(
    "example_path", EXAMPLE_PATHS, ids=(path.name for path in EXAMPLE_PATHS)
)
def test_script_execution(
    example_path: Path,
    tmp_wd: Path,
    monkeypatch,
    reset_global_configuration,
) -> None:
    dir_path = example_path.parent.name
    copytree(example_path.parent, dir_path)
    monkeypatch.chdir(dir_path)
    runpy.run_path(example_path.name)
