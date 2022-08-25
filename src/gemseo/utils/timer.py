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
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
"""A timer to measure the time spent within a ``with`` statement."""
from __future__ import annotations

import logging
from time import perf_counter

LOGGER = logging.getLogger(__name__)


class Timer:
    """A timer to measure the time spent within a ``with`` statement.

    Examples:
        with Timer() as timer:
            # do stuff

        elapsed_time = timer.elapsed_time
    """

    def __init__(self, log_level: str | int | None = None) -> None:
        """
        Args:
            log_level: The level of the logger.
                If ``None``, do not log the elapsed time.
        """
        if log_level is not None:
            log_level = logging.getLevelName(log_level)

        self.__log_level = log_level
        self.__elapsed_time = 0.0

    @property
    def elapsed_time(self) -> float:
        """The time spent within the ``with`` statement."""
        return self.__elapsed_time

    def __enter__(self) -> Timer:
        self.__elapsed_time = perf_counter()
        return self

    def __exit__(self, type, value, traceback) -> None:
        self.__elapsed_time = perf_counter() - self.__elapsed_time
        if self.__log_level is not None:
            LOGGER.log(self.__log_level, str(self))

    def __str__(self) -> str:
        return f"Elapsed time: {self.elapsed_time} s."
