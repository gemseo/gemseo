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
"""Tools for the creation of directories."""

from __future__ import annotations

from ast import literal_eval
from multiprocessing import Lock
from multiprocessing import Value
from pathlib import Path
from uuid import uuid4

from strenum import StrEnum

from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta


class DirectoryNamingMethod(StrEnum):
    """The method to generate directory names."""

    NUMBERED = "NUMBERED"
    """The generated directories are named by an integer i+1, i+2, i+3 etc, where ``i``
    is the maximum value of the already existing directories."""

    UUID = "UUID"
    """A unique number based on the UUID function is generated.

    This last option shall be used if multiple MDO processes are run in the same working
    directory. This is multi-process safe.
    """


class DirectoryCreator(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A class to create directories."""

    __counter: Value
    """The number of created directories."""

    __root_directory: Path
    """The absolute path of the root directory, wherein unique directories will be
    created."""

    __lock: Lock
    """The non-recursive lock object."""

    __directory_naming_method: DirectoryNamingMethod
    """The method to create the directory names."""

    __last_directory: Path | None
    """The last created directory or ``None`` if none has been created."""

    def __init__(
        self,
        root_directory: str | Path = "",
        directory_naming_method: DirectoryNamingMethod = DirectoryNamingMethod.NUMBERED,
    ) -> None:
        """
        Args:
            root_directory: The path to the root directory,
                wherein unique directories will be created.
                If empty, use the current working directory.
            directory_naming_method: The method to create the directory names.
        """  # noqa:D205 D212 D415
        self.__root_directory = (
            Path.cwd()
            if root_directory == ""
            else Path(root_directory).absolute().resolve()
        )
        self.__directory_naming_method = directory_naming_method
        if directory_naming_method == DirectoryNamingMethod.NUMBERED:
            self.__lock = Lock()
            self.__counter = Value("i", self.__get_initial_counter())
        else:
            self.__counter = 1
        self.__last_directory = None

    @property
    def last_directory(self) -> Path | None:
        """The last created directory or ``None`` if none has been created."""
        return self.__last_directory

    # TODO: API: Make this method either protected or removed in new major version.
    def get_unique_run_folder_path(self) -> Path:
        """Generate a directory path.

        Returns:
            The directory path.
        """
        self.__last_directory = self.__root_directory / self.__generate_uid()
        return self.__last_directory

    def create(self) -> Path:
        """Create a directory.

        Returns:
            The directory path.
        """
        self.get_unique_run_folder_path()
        self.__last_directory.mkdir(parents=True, exist_ok=True)
        return self.__last_directory

    def __generate_uid(self) -> str:
        """Generate a unique identifier.

        Generate a unique identifier for the current execution.
        If the ``directory_naming_method`` strategy is
        :attr:`~.DirectoryNamingMethod.NUMBERED`,
        the successive uid are named by an integer 1, 2, 3 etc.
        Otherwise, a unique number based on the UUID function is generated.
        This last option shall be used if multiple MDO processes are run
        in the same working directory, since it is multi-process safe.

        Returns:
            A unique identifier.
        """
        if self.__directory_naming_method == DirectoryNamingMethod.NUMBERED:
            with self.__lock:
                self.__counter.value += 1
                return str(self.__counter.value)
        elif self.__directory_naming_method == DirectoryNamingMethod.UUID:
            return str(uuid4()).split("-")[-1]
        return None

    def __get_initial_counter(self) -> int:
        """Return the initial value of the counter for creating directories.

        This accounts for the already existing directories in :attr:`.root_directory`.

        Returns:
             The initial value of the counter.
        """
        # Only keep directories which are a number.
        out_dirs = [
            path.name
            for path in self.__root_directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]

        if not out_dirs:
            return 0
        return max(literal_eval(n) for n in out_dirs)
