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

from gemseo.core.discipline import Discipline
from gemseo.util.global_configuration import _configuration

if TYPE_CHECKING:
    from gemseo.discipline.wrapper._base_executable_runner import _BaseExecutableRunner
    from gemseo.util.typing import StrKeyMapping


class _BaseDiscFromExe(Discipline):
    """Base class for wrapping an executable in a discipline."""

    _executable_runner: _BaseExecutableRunner
    """The executable runner."""

    __clean_after_execution: bool
    """Whether to clean the working directory after execution."""

    def __init__(
        self,
        executable_runner: _BaseExecutableRunner,
        name: str = "",
        clean_after_execution: bool = False,
    ) -> None:
        """
        Args:
            executable_runner: The attached executable runner.
            clean_after_execution: Whether to clean the last created directory after
                execution.
        """  # noqa: D205, D212, D415
        super().__init__(name)
        self._executable_runner = executable_runner
        self.__clean_after_execution = clean_after_execution

    @abstractmethod
    def _create_inputs(self, input_data: StrKeyMapping) -> None:
        """Create the input files.

        Args:
            input_data: The input data.
        """

    @abstractmethod
    def _parse_outputs(self) -> StrKeyMapping:
        """Parse the output files.

        Returns:
            The output data for updating the discipline's `data`.
        """

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        self._executable_runner.create_data_directory()
        self._create_inputs(input_data)
        self._executable_runner.execute()
        output_data = self._parse_outputs()
        self.__clean_data_directory()
        return output_data

    def __clean_data_directory(self) -> None:
        """Clean the data directory."""
        if not _configuration.directory_manager.enable and self.__clean_after_execution:
            rmtree(self._executable_runner.data_directory)
