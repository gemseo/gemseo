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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Sebastien Bocquet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pickle
from pathlib import Path
from re import match

import pytest
from gemseo.utils.run_folder_manager import FoldersIter
from gemseo.utils.run_folder_manager import RunFolderManager

BASE_DIR = Path("resource_dir")


@pytest.fixture
def directories(tmp_wd):
    """Generate three directories and a file to test the NUMBERED folder_iters.

    1. toto1: a mix of str and integer. Should not be considered.
    2. 3: an integer. Should be considered.
    3. dir_no_number: a string without any number. Should not be considered.
    4. a file is added in the base directory.
    """
    (BASE_DIR / "toto1").mkdir(parents=True)
    (BASE_DIR / "3").mkdir()
    (BASE_DIR / "dir_no_number").mkdir()
    filepath = BASE_DIR / "toto.txt"
    with filepath.open("w") as f:
        f.write("foo")


@pytest.fixture
def empty_directory(tmp_wd):
    """Generate an empty directory."""
    Path("empty_resource_dir").mkdir()


def test_get_unique_run_folder_path(directories):
    """Test the method: ``get_unique_run_folder_path``."""
    folder_manager = RunFolderManager("resource_dir")
    assert (
        folder_manager.get_unique_run_folder_path() == Path("resource_dir/4").absolute()
    )
    assert (
        folder_manager.get_unique_run_folder_path() == Path("resource_dir/5").absolute()
    )


def test_get_unique_run_folder_path_empty(empty_directory):
    """Test the method: ``get_unique_run_folder_path`` on empty directory."""
    folder_manager = RunFolderManager("empty_resource_dir")
    assert (
        folder_manager.get_unique_run_folder_path()
        == Path("empty_resource_dir/1").absolute()
    )


def test_uuid_folder(tmp_wd, directories):
    """Test that unique folder based on ``UUID`` can be written in a non empty
    directory."""

    folder_manager = RunFolderManager(
        output_folder_basepath=BASE_DIR, folders_iter=FoldersIter.UUID
    )
    for _ in range(2):
        folder_name = folder_manager.get_unique_run_folder_path()
        assert match("[0-9a-fA-F]{12}$", str(folder_name.name)) is not None
        Path.mkdir(folder_name)


def test_run_folder_manager_serialization(tmp_wd):
    """Test that a :class:`~.RunFolderManager` can be serialised and deserialised in
    ``UUID`` mode."""

    folder_manager = RunFolderManager(
        output_folder_basepath=BASE_DIR, folders_iter=FoldersIter.UUID
    )
    with open("run_folder.pkl", "wb") as file:
        pickle.dump(folder_manager, file)

    with open("run_folder.pkl", "rb") as file:
        loaded_folder_manager = pickle.load(file)
        assert loaded_folder_manager.__dict__ == folder_manager.__dict__
