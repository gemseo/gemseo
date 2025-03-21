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
"""Base class for a tool to generate names."""

from __future__ import annotations

from multiprocessing import Lock
from multiprocessing import Value
from uuid import uuid4

from strenum import StrEnum

from gemseo.core.serializable import Serializable


class BaseNameGenerator(Serializable):
    """A class to generate names."""

    class Naming(StrEnum):
        """The method to generate names."""

        NUMBERED = "NUMBERED"
        """The generated names use an integer i+1, i+2, i+3 etc, where ``i``
        is the maximum value of the already existing names."""

        UUID = "UUID"
        """A unique name based on the UUID function is generated.

        This last option shall be used if multiple MDO processes are run in the same
        working directory. This is multi-process safe.
        """

    __counter: Value
    """The number of generated files."""

    __lock: Lock
    """The non-recursive lock object."""

    __naming_method: Naming
    """The method to create the names."""

    def __init__(
        self,
        naming_method: Naming = Naming.NUMBERED,
    ) -> None:
        """
        Args:
            naming_method: The method to create the names.
        """  # noqa:D205 D212 D415
        self.__naming_method = naming_method
        self._init_shared_memory_attrs_after()

    def _init_shared_memory_attrs_after(self) -> None:
        if self.__naming_method == self.Naming.NUMBERED:
            self.__lock = Lock()
            self.__counter = Value("i", self._get_initial_counter())
        else:
            self.__counter = 0

    def _generate_name(self) -> str | None:
        """Generate a name.

        If the ``naming_method`` strategy is
        :attr:`~.Naming.NUMBERED`,
        the successive names are generated by an integer 1, 2, 3 etc.
        Otherwise, a unique number based on the UUID function is generated.
        This last option shall be used if multiple MDO processes are run
        in the same working directory, since it is multiprocess safe.

        Returns:
            A name.
        """
        if self.__naming_method == self.Naming.NUMBERED:
            with self.__lock:
                self.__counter.value += 1
                return str(self.__counter.value)
        elif self.__naming_method == self.Naming.UUID:
            return str(uuid4()).split("-")[-1]
        return None

    def _get_initial_counter(self) -> int:
        """Return the initial value of the counter for generating names.

        Returns:
             The initial value of the counter.
        """
        return 0
