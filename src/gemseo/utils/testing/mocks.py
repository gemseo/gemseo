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
"""Mocks."""

from __future__ import annotations


class SleepingCounter:
    """A class mocking ``time.perf_counter`` ."""

    __sleep_duration: float
    """The sleep duration."""

    __time: float
    """The current time."""

    def __init__(self, sleep_duration: float) -> None:
        """
        Args:
            sleep_duration: The sleep duration.
        """  # noqa: D205 D212
        self.__sleep_duration = sleep_duration
        self.__time = 0

    def __call__(self) -> float:
        """Sleep and return the current time."""
        self.__time += self.__sleep_duration
        return self.__time
