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
"""Base class to make a discipline from an executable."""

from __future__ import annotations

from abc import abstractmethod
from shutil import rmtree
from typing import TYPE_CHECKING

from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.core.discipline_data import Data
    from gemseo.wrappers._base_executable_runner import _BaseExecutableRunner


class _BaseDiscFromExe(MDODiscipline):
    """Base class for wrapping an executable in a discipline."""

    _executable_runner: _BaseExecutableRunner
    """The executable runner."""

    __clean_after_execution: bool
    """Whether to clean the working directory after execution."""

    def __init__(
        self,
        executable_runner: _BaseExecutableRunner,
        name: str | None = None,
        input_grammar_file: str | Path | None = None,
        output_grammar_file: str | Path | None = None,
        auto_detect_grammar_files: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        cache_type: MDODiscipline.CacheType = MDODiscipline.CacheType.SIMPLE,
        cache_file_path: str | Path | None = None,
        clean_after_execution: bool = False,
    ) -> None:
        """
        Args:
            executable_runner: The attached executable runner.
            clean_after_execution: Whether to clean the working directory after
                execution.
        """  # noqa: D205, D212, D415
        super().__init__(
            name,
            input_grammar_file,
            output_grammar_file,
            auto_detect_grammar_files,
            grammar_type,
            cache_type,
            cache_file_path,
        )
        self._executable_runner = executable_runner
        self.__clean_after_execution = clean_after_execution

    @abstractmethod
    def _create_inputs(self) -> None:
        """Create the input files."""

    @abstractmethod
    def _parse_outputs(self) -> Data:
        """Parse the output files.

        Returns:
            The output data for updating the discipline's ``local_data``.
        """

    def _run(self) -> None:
        self._executable_runner.create_directory()
        self._create_inputs()
        self._executable_runner.execute()
        self.store_local_data(**self._parse_outputs())

        if self.__clean_after_execution:
            rmtree(self.last_execution_directory)

    @property
    def last_execution_directory(self) -> Path | None:
        """The last directory wherein the executable was executed, or ``None``."""
        return self._executable_runner.last_execution_directory
