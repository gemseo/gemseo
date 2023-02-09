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
from re import findall
from shutil import copytree

import pytest

EXAMPLE_PATHS = [
    path
    for path in Path(__file__, "..", "..", "doc_src", "_examples")
    .resolve()
    .rglob("*.py")
    if (path.parent / "README.rst").is_file()
    and not findall(r"(run|post_process_|save_from_)\w*\.py$", path.name)
]


@pytest.mark.doc_examples
@pytest.mark.parametrize(
    "example_path", EXAMPLE_PATHS, ids=(path.name for path in EXAMPLE_PATHS)
)
def test_script_execution(example_path, tmp_wd, monkeypatch):
    dir_path = example_path.parent.name
    Path(copytree(example_path.parent, dir_path))
    monkeypatch.chdir(dir_path)
    runpy.run_path(example_path.name)
