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
"""Base class to make an executable runner by running a command line."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from shutil import copy2
from typing import TYPE_CHECKING
from typing import Any

from gemseo.utils.directory_creator import DirectoryCreator
from gemseo.utils.directory_creator import DirectoryNamingMethod
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

LOGGER = logging.getLogger(__name__)


class _BaseExecutableRunner(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Handle executing a command line in a subprocess.

    This class also handles the creation of the directory where the command line is
    executed, as well as the copy of the files required for the execution.
    """

    command_line: str
    """The command line to run the executable."""

    _files: Iterable[str | Path] = ()
    """The files to copy into the execution directory."""

    _working_directory: str | Path
    """The directory within to execute the command line."""

    __directory_creator: DirectoryCreator
    """The object generating directories with unique names."""

    __subprocess_run_options: Mapping[str, Any]
    """The options of the ``subprocess.run`` method."""

    def __init__(
        self,
        root_directory: str | Path,
        command_line: str,
        directory_naming_method: DirectoryNamingMethod = DirectoryNamingMethod.UUID,
        files: Iterable[str | Path] = (),
        working_directory: str | Path = "",
        **subprocess_run_options: Any,
    ) -> None:
        """
        Args:
            root_directory: The path to the root directory,
                wherein unique directories will be created at each execution.
                If empty, use the current working directory.
            command_line: The command line to run the executable.
                E.g. ``python my_script.py -i input.txt -o output.txt``
            directory_naming_method: The method to create the execution directories.
            files: The files to copy into the execution directory.
            working_directory: The directory within to execute the command line.
                If empty, execute the command line into the unique generated directory.
            **subprocess_run_options: The options of the ``subprocess.run`` method.
        """  # noqa:D205 D212 D415
        self.__directory_creator = DirectoryCreator(
            root_directory=root_directory,
            directory_naming_method=directory_naming_method,
        )
        self.command_line = command_line
        self._files = files
        self._working_directory = working_directory
        self.__set_subprocess_run_options(subprocess_run_options)

    def __set_subprocess_run_options(
        self,
        subprocess_run_options: Mapping[str, Any],
    ) -> None:
        """Set the ``subprocess.run`` options.

        By default, the ``stderr`` option is set to ``subprocess.STDOUT``.

        Args:
            subprocess_run_options: The options for the ``subprocess.run`` method.

        Raises:
            KeyError: When the options ``cwd``, ``args`` or ``shell`` are given.
        """
        self.__subprocess_run_options = {"stderr": subprocess.STDOUT}

        intersection = {"cwd", "args", "shell"}.intersection(subprocess_run_options)
        if intersection:
            msg = (
                f"{intersection} must not be defined a second time "
                "in subprocess_run_options."
            )
            raise KeyError(msg)
        self.__subprocess_run_options.update(subprocess_run_options)

    def create_directory(self) -> Path:
        """Create a unique execution directory."""
        return self.__directory_creator.create()

    def __copy_files(self):
        """Copy the files into the execution directory."""
        if self.__directory_creator.last_directory:
            for file_ in self._files:
                copy2(
                    file_,
                    self.__directory_creator.last_directory / Path(file_).name,
                )

    def execute(self) -> None:
        """Execute the command line."""
        if self._working_directory:
            working_directory = self._working_directory
        else:
            working_directory = self.__directory_creator.last_directory

        self.__copy_files()

        self._pre_processing()

        completed = subprocess.run(
            self.command_line.split(),
            cwd=working_directory,
            **self.__subprocess_run_options,
        )

        if completed.returncode != 0:
            LOGGER.error(
                "Failed to execute the command %s,"
                "from %s, "
                "in the working directory %s.",
                self.command_line,
                working_directory,
                self.__directory_creator.last_directory,
            )

        completed.check_returncode()

        self._post_processing()

    def _pre_processing(self) -> None:
        """Execute the pre-processing steps.

        These steps are executed before the command line.
        """

    def _post_processing(self) -> None:
        """Execute the post-processing steps.

        These steps are executed after the command line.
        """

    @property
    def last_execution_directory(self) -> Path | None:
        """The last directory wherein the executable was executed, or ``None``."""
        return self.__directory_creator.last_directory
