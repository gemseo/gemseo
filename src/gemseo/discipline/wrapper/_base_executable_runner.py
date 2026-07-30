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
from shutil import copytree
from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.serializable import Serializable
from gemseo.util._directory_manager.manager import _get_cwd
from gemseo.util.directory_creator import DirectoryCreator
from gemseo.util.directory_creator import Naming
from gemseo.util.global_configuration import _configuration

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.util.typing import StrKeyMapping
    from gemseo.util.typing import StrPath

LOGGER = logging.getLogger(__name__)


class _BaseExecutableRunner(Serializable):
    """Handle executing a command line in a subprocess.

    The creation of directories where the data of the command line are stored can
    be automatically handled.
    If the :attr:`gemseo.configuration._directory_manager.enable` is True,
    then no particular setting should be provided.
    (See :ref:`_directory_manager` for more info)
    Otherwise,
    the method used to automatically create the data directory is defined
    with the ``directory_naming_method`` argument.
    This class also handles the copy of the files required for the execution.

    The working directory is the directory where the command line is executed,
    it can be different from the directory where the data are stored.
    The working directory can be specified with the ``working_directory``
    argument.
    Otherwise,
    the working directory is the same directory as the directory where the
    data are stored.
    """

    command_line: str
    """The command line to run the executable."""

    _data_paths: Iterable[Path]
    """The directories and files to copy into the execution directory."""

    __execution_directory: Path | None
    """The directory from which the command line is executed."""

    __data_directory: Path | None
    """The data directory, None until created."""

    __directory_creator: DirectoryCreator
    """The object generating directories with unique names.

    It is not used when the directory manager is enabled; the choice is made
    at execution time since the enable setting can change after construction
    (e.g. it is disabled in the worker processes until the pickled state of
    the parent is applied)."""

    __subprocess_run_options: StrKeyMapping
    """The options of the `subprocess.run` method."""

    def __init__(
        self,
        command_line: str,
        root_data_directory: StrPath = "",
        naming: Naming = Naming.UUID,
        data_paths: Iterable[StrPath] = (),
        execution_directory: StrPath = "",
        **subprocess_run_options: Any,
    ) -> None:
        """
        Args:
            command_line: The command line to run the executable.
                E.g. `python my_script.py -i input.txt -o output.txt`
            root_data_directory: The path to the root directory for storing the data,
                wherein unique directories will be created at each execution.
                If empty, use the current working directory.
                When the directory manager is enabled, this argument is ignored.
            naming: The naming convention to create the execution directories.
                When the directory manager is enabled, this argument is ignored.
            data_paths: The directories and files to copy into the execution
                directory.
            execution_directory: The directory within to execute the command line.
                If empty, execute the command line in the same directory as the
                one used where the data are stored.
            **subprocess_run_options: The options of the `subprocess.run` method.
        """  # noqa:D205 D212 D415
        self.command_line = command_line
        self._data_paths = list(map(Path, data_paths))
        self.__execution_directory = (
            Path(execution_directory) if execution_directory else None
        )
        self.__data_directory = None
        self.__set_subprocess_run_options(subprocess_run_options)
        self.__directory_creator = DirectoryCreator(
            naming,
            root_directory=root_data_directory,
        )

    def __set_subprocess_run_options(
        self,
        subprocess_run_options: StrKeyMapping,
    ) -> None:
        """Set the `subprocess.run` options.

        By default, the `stderr` option is set to `subprocess.STDOUT`.

        Args:
            subprocess_run_options: The options for the `subprocess.run` method.

        Raises:
            KeyError: When the options `cwd`, `args` or `shell` are given.
        """
        self.__subprocess_run_options = {"stderr": subprocess.STDOUT}

        intersection = {"cwd", "args", "shell"}.intersection(subprocess_run_options)
        if intersection:
            msg = (
                f"{sorted(intersection)} must not be defined a second time "
                "in subprocess_run_options."
            )
            raise KeyError(msg)
        self.__subprocess_run_options.update(subprocess_run_options)

    def __copy_data_paths(self) -> None:
        """Copy the directories and files into the directory of the command line."""
        destination_directory = self.execution_directory
        if destination_directory is None:
            return
        for path in self._data_paths:
            dst = destination_directory / path.name
            if path.is_file():
                copy2(path, dst)
            elif path.is_dir():
                copytree(path, dst)

            else:
                msg = (
                    f"Can't copy {path} into {destination_directory} "
                    "since it is neither a file nor a directory."
                )
                LOGGER.warning(msg)

    @property
    def execution_directory(self) -> Path | None:
        """The directory where the command line is executed.

        None when no directory was given and the data directory is not yet
        created: the command line is then executed in the current directory.
        """
        if self.__execution_directory is None:
            return self.__data_directory
        return self.__execution_directory

    @property
    def data_directory(self) -> Path | None:
        """The data directory, None until created."""
        return self.__data_directory

    def create_data_directory(self) -> Path:
        """Create the data directory and set its related attribute.

        Returns:
            The data directory.
        """
        if _configuration.directory_manager.enable:
            data_directory = _get_cwd()
        else:
            data_directory = self.__directory_creator.create()
        self.__data_directory = data_directory
        return data_directory

    def execute(self) -> None:
        """Execute the command line."""
        self.__copy_data_paths()
        self._pre_processing()
        execution_directory = self.execution_directory

        completed = subprocess.run(
            self.command_line.split(),
            cwd=execution_directory,
            **self.__subprocess_run_options,
        )

        if completed.returncode != 0:
            LOGGER.error(
                "Failed to execute the command %s, "
                "from the execution directory %s, "
                "with the data directory %s.",
                self.command_line,
                execution_directory,
                self.__data_directory,
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
