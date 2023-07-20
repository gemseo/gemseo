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

from pathlib import Path

import pytest
from gemseo.utils.run_folder_manager import RunFolderManager


@pytest.fixture
def directories(tmp_wd):
    """Generate three directories to test the NUMBERED folder_iters.

    1. toto1: a mix of str and integer. Should not be consider.
    2. 3: an integer. Should be consider.
    3. dir_no_number: a string without any number. Should not be consider.
    """
    Path("resource_dir/toto1").mkdir(parents=True)
    Path("resource_dir/3").mkdir()
    Path("resource_dir/dir_no_number").mkdir()


@pytest.fixture
def empty_directory(tmp_wd):
    """Generate empty directory."""
    Path("empty_resource_dir").mkdir()


def test_get_unique_run_folder_path(directories):
    """Test the method: ``get_unique_run_folder_path``."""
    folder_manager = RunFolderManager("resource_dir")
    assert folder_manager.get_unique_run_folder_path() == Path("resource_dir/4")
    assert folder_manager.get_unique_run_folder_path() == Path("resource_dir/5")


def test_get_unique_run_folder_path_empty(empty_directory):
    """Test the method: ``get_unique_run_folder_path`` on empty directory."""
    folder_manager = RunFolderManager("empty_resource_dir")
    assert folder_manager.get_unique_run_folder_path() == Path("empty_resource_dir/1")
