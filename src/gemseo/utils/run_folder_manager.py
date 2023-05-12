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
"""Tools for unique run folder name generation."""
from __future__ import annotations

from ast import literal_eval
from multiprocessing import Lock
from multiprocessing import Value
from os import listdir
from pathlib import Path
from uuid import uuid1

from strenum import StrEnum

from gemseo.utils.portable_path import PLATFORM_IS_WINDOWS

FoldersIter = StrEnum("FoldersIter", "NUMBERED UUID")


class RunFolderManager:
    """Class generating unique names for run folders."""

    __counter: Value
    """The number of existing run folders."""
    __output_folder_basepath: Path
    """The path of the root of run folders."""
    __lock: Lock
    """The non-recursive lock object."""
    __use_shell: bool
    """Whether to run the command using the default shell."""
    _folders_iter: FoldersIter
    """The type of unique identifiers for the output folders."""

    def __init__(
        self,
        output_folder_basepath: str,
        folders_iter: FoldersIter = FoldersIter.NUMBERED,
        use_shell: bool = True,
    ) -> None:
        """

        Args:
            folders_iter: The type of unique identifiers for the output folders.
                If :attr:`~.FoldersIter.NUMBERED`,
                the generated output folders will be ``f"output_folder_basepath{i+1}"``,
                where ``i`` is the maximum value
                of the already existing ``f"output_folder_basepath{i}"`` folders.
                Otherwise, a unique number based on the UUID function is
                generated. This last option shall be used if multiple MDO
                processes are run in the same work folder.
            use_shell: If ``True``, run the command using the default shell.
                Otherwise, run directly the command.
            output_folder_basepath: The base path of the execution folders.

        """  # noqa:D205 D212 D415
        self.__use_shell = use_shell
        self.__lock = Lock()
        self._folders_iter = folders_iter
        self.__output_folder_basepath = Path(output_folder_basepath)
        self.__check_base_path_on_windows()
        self.__counter = Value("i", self.__get_max_outdir())

    def get_unique_run_folder_path(self) -> Path:
        """Generate a unique folder path for the run folder.

        Returns: A unique Path to be used as run folder.
        """
        return self.__output_folder_basepath / self.__generate_uid()

    def __generate_uid(self) -> str:
        """Generate a unique identifier for the execution folder.

        Generate a unique identifier for the current execution.
        If the _folders_iter strategy is :attr:`~.FoldersIter.NUMBERED`,
        the successive iterations are named by an integer 1, 2, 3 etc.
        This is multiprocess safe.
        Otherwise, a unique number based on the UUID function is generated.
        This last option shall be used if multiple MDO processes are run
        in the same workdir.

        Returns:
            An unique string identifier (either a number or a UUID).

        Raises:
            ValueError: If ``_folders_iter`` is not a :class:`.FoldersIter` object.
        """
        if self._folders_iter == FoldersIter.NUMBERED:
            with self.__lock:
                self.__counter.value += 1
                return str(self.__counter.value)
        elif self._folders_iter == FoldersIter.UUID:
            return str(uuid1()).split("-")[-1]
        raise ValueError(
            f"{self._folders_iter} is not a valid method "
            "for creating the execution folders."
        )

    def __get_max_outdir(self) -> int:
        """Get the maximum current index of output folders.

        Returns:
             The maximum index in the output folders.
        """
        outs = listdir(self.__output_folder_basepath)
        if not outs:
            return 0
        return max(literal_eval(n) for n in outs)

    def __check_base_path_on_windows(self) -> None:
        """Check that the base path can be used.

        Raises:
            ValueError: When the users use the shell under Windows
            and the base path is located on a network location.
        """
        if PLATFORM_IS_WINDOWS and self.__use_shell:
            output_folder_base_path = Path(self.__output_folder_basepath).resolve()
            if not output_folder_base_path.parts[0].startswith("\\\\"):
                return

            raise ValueError(
                "A network base path and use_shell cannot be used together"
                " under Windows, as cmd.exe cannot change the current folder"
                " to a UNC path."
                " Please try use_shell=False or use a local base path."
            )
