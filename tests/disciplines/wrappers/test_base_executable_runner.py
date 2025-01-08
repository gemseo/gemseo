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

import logging
from pathlib import Path

import pytest

from gemseo.disciplines.wrappers._base_executable_runner import _BaseExecutableRunner
from gemseo.utils.directory_creator import DirectoryNamingMethod


@pytest.mark.parametrize("root_directory", [".", Path()])
@pytest.mark.parametrize(
    "identifiers", [DirectoryNamingMethod.UUID, DirectoryNamingMethod.NUMBERED]
)
def test_create_directory(tmp_wd, root_directory, identifiers) -> None:
    base_exec_runner = _BaseExecutableRunner(
        root_directory=root_directory,
        command_line="echo hello world",
        identifiers=identifiers,
    )
    path = base_exec_runner.directory_creator.create()
    assert path.exists()
    assert path == base_exec_runner.directory_creator.last_directory


@pytest.mark.parametrize("command", ["echo hello world", "ls"])
def test_executable_command(tmp_wd, command) -> None:
    base_exec_runner = _BaseExecutableRunner(
        root_directory=".",
        command_line=command,
    )
    assert command == base_exec_runner.command_line


def test_change_working_directory(tmp_wd) -> None:
    """Test to use a different execution directory."""
    Path("toto").mkdir()
    _BaseExecutableRunner(
        "python -c open('toto.txt','w')", ".", working_directory="toto"
    ).execute()
    assert not Path("toto.txt").exists()
    assert Path("toto/toto.txt").exists()


def test_directories_and_files(tmp_wd, caplog) -> None:
    """Test to copy files and directories."""
    # Create empty files and directory.
    Path("directory_tata").mkdir()
    with (
        Path("toto.txt").open("w") as f,
        Path("directory_tata/tata.txt").open("w") as g,
    ):
        f.write("toto")
        g.write("tata")

    exec_runner = _BaseExecutableRunner(
        "python --version",
        ".",
        data_paths=[
            "toto.txt",
            Path("directory_tata"),
            "not_existing_file.txt",
        ],
    )

    # Conversion into Path() objects is done in the __init__.
    assert exec_runner._data_paths == [
        Path("toto.txt"),
        Path("directory_tata"),
        Path("not_existing_file.txt"),
    ]

    wd = exec_runner.directory_creator.create()
    exec_runner.execute()

    # Test the warning message if the file/directory does not exist.
    msg = (
        f"Can't copy not_existing_file.txt into {wd} "
        "since it is neither a file nor a directory."
    )
    assert (
        "gemseo.disciplines.wrappers._base_executable_runner",
        logging.WARNING,
        msg,
    ) in caplog.record_tuples

    # Test the directory and files are copied.
    assert Path("toto.txt").exists()
    assert Path("directory_tata/tata.txt").exists()
    assert (wd / "toto.txt").exists()
    assert (wd / "directory_tata/tata.txt").exists()


def test_run_options(tmp_wd) -> None:
    log_file = Path("myfile_stdout.txt")
    with log_file.open("w") as outfile:
        exec_runner = _BaseExecutableRunner(
            "python --version",
            ".",
            stdout=outfile,
        )
        exec_runner.execute()

    assert "Python 3." in log_file.read_text()


@pytest.mark.parametrize(
    "options",
    [
        {"shell": True},
        {"cwd": "."},
        {"args": "python -h"},
        {"shell": False, "cwd": ".."},
    ],
)
def test_run_options_error(tmp_wd, options) -> None:
    msg = (
        f"{set(options.keys())} must not be defined a second time in "
        "subprocess_run_options."
    )
    with pytest.raises(
        KeyError,
        match=msg,
    ):
        _BaseExecutableRunner(
            "python --version",
            ".",
            **options,
        )
