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
from datetime import datetime
from datetime import timedelta
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import TracebackType

    from typing_extensions import Self

LOGGER = logging.getLogger(__name__)


class Timer:
    """A timer to measure the time spent within a ``with`` statement.

    Examples:
        with Timer() as timer:
            # do stuff

        elapsed_time = timer.elapsed_time
    """

    __elapsed_time: float
    """The elapsed time in seconds."""

    __entering_timestamp: datetime.datetime
    """The entering timestamp."""

    __exiting_timestamp: datetime.datetime
    """The exiting timestamp."""

    __log_level: int | None
    """The logging level, ``None`` means no logging."""

    def __init__(self, log_level: int | None = None) -> None:
        """
        Args:
            log_level: The level of the logger.
                If ``None``, do not log the elapsed time.
        """  # noqa:D205 D212 D415
        self.__elapsed_time = 0.0
        now = datetime.now()
        self.__entering_timestamp = now
        self.__exiting_timestamp = now
        self.__log_level = log_level

    @property
    def elapsed_time(self) -> float:
        """The time spent within the ``with`` statement."""
        return self.__elapsed_time

    @property
    def entering_timestamp(self) -> datetime:
        """The entering timestamp of the ``with`` statement."""
        return self.__entering_timestamp

    @property
    def exiting_timestamp(self) -> datetime:
        """The exiting timestamp of the ``with`` statement."""
        return self.__exiting_timestamp

    def __enter__(self) -> Self:
        self.__elapsed_time = perf_counter()
        self.__entering_timestamp = datetime.now()
        return self

    def __exit__(
        self,
        _: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.__elapsed_time = perf_counter() - self.__elapsed_time
        self.__exiting_timestamp = self.__entering_timestamp + timedelta(
            seconds=self.__elapsed_time
        )
        if self.__log_level is not None:
            LOGGER.log(self.__log_level, str(self))

    def __str__(self) -> str:
        return f"Elapsed time: {self.__elapsed_time} s."
