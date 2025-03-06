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
from pathlib import Path

from gemseo.utils.name_generator import NameGenerator

# TODO: API Rename to Naming and update all references.
DirectoryNamingMethod = NameGenerator.Naming


class DirectoryCreator(NameGenerator):
    """A class to create directories."""

    __root_directory: Path
    """The absolute path of the root directory, wherein unique directories will be
    created."""

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
        self.__root_directory = Path(root_directory) if root_directory else Path.cwd()
        self.__last_directory = None
        # TODO: API Rename to naming_method
        super().__init__(naming_method=directory_naming_method)

    @property
    def last_directory(self) -> Path | None:
        """The last created directory or ``None`` if none has been created."""
        return self.__last_directory

    def create(self) -> Path:
        """Create a directory.

        Returns:
            The directory path.
        """
        self.__last_directory = self.__root_directory / self._generate_name()
        self.__last_directory.mkdir(parents=True, exist_ok=True)
        return self.__last_directory

    def _get_initial_counter(self) -> int:
        """Return the initial value of the counter for creating directories.

        This accounts for the already existing directories in :attr:`.__root_directory`.

        Returns:
             The initial value of the counter.
        """
        if not self.__root_directory.is_dir():
            return 0

        # Only keep directories which are a number.
        out_dirs = [
            path.name
            for path in self.__root_directory.iterdir()
            if path.is_dir() and path.name.isdigit()
        ]

        if not out_dirs:
            return 0
        return max(literal_eval(n) for n in out_dirs)
