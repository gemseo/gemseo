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
"""Execution statistics."""

from __future__ import annotations

from contextlib import contextmanager
from multiprocessing import Value
from timeit import default_timer
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from docstring_inheritance import GoogleDocstringInheritanceMeta

from gemseo.core.serializable import Serializable
from gemseo.utils.multiprocessing.manager import get_multi_processing_manager
from gemseo.utils.timer import Timer

if TYPE_CHECKING:
    from collections.abc import Generator
    from multiprocessing.managers import DictProxy
    from multiprocessing.managers import ListProxy
    from multiprocessing.sharedctypes import Synchronized


class _Meta(GoogleDocstringInheritanceMeta):
    """Implement class properties for class attributes."""

    time_stamps: DictProxy[str, list[tuple[float, float, bool]]] | None

    @property
    def is_time_stamps_enabled(self) -> bool:
        """Whether to record the time stamps."""
        return self.time_stamps is not None

    @is_time_stamps_enabled.setter
    def is_time_stamps_enabled(self, value: bool) -> None:
        if value:
            self.time_stamps = get_multi_processing_manager().dict()
        else:
            self.time_stamps = None


class ExecutionStatistics(Serializable, metaclass=_Meta):
    """Record execution statistics of objects.

    This should be applied to objects such as :class:`.BaseMonitoredProcess`,
    hereafter referred to as the _measured object_.

    A measured object often has an execution method
    whose number of calls is counted and time measured.
    Some have also a linearization method whose number of calls is counted too.

    The recording of the statistics can be disabled all at once by setting
    :attr:`is_enabled` to ``False``. By default, it is set to ``True``.
    When enabled, the recording of time stamps can be enabled by setting
    :attr:`is_time_stamps_enabled` to ``True``. By default, it is set to ``False``.
    These switches are global and shall be modified from the class.

    If any of those switches are disabled, the recordings, if any, are not removed.

    The helper method :meth:`record` is a context manager that should be used to record
    statistics.

    The results of the recordings can be accessed with
    :attr:`.n_calls`,
    :attr:`.n_calls_linearize`,
    :attr:`.duration`,
    :attr:`.time_stamps`.
    The time stamps should be processed with :func:`create_gantt_chart`.

    The recorded statistics are not restored after pickling.
    """

    time_stamps: ClassVar[
        DictProxy[str, ListProxy[tuple[float, float, bool]]] | None
    ] = None
    """The mapping from the measured object names to their execution time stamps.

    It is ``None`` when time stamps recording is disabled.

    The structure is

    .. code-block::

       {
       "measure object name": [
           (start time, end time, whether it is for linearization),
           ...
           ],
       "other measure object name": [
           ...
           ],
       }

    """

    is_enabled: ClassVar[bool] = True
    """Whether to record all the statistics."""

    __duration: Synchronized[float]
    """The cumulated execution duration."""

    __n_calls: Synchronized[int]
    """The number of calls to the execution method."""

    __n_calls_linearize: Synchronized[int]
    """The number of calls to the linearization method."""

    __name: str
    """The name of the measured object."""

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = {
        "__duration",
        "__n_calls",
        "__n_calls_linearize",
    }

    def __init__(self, name: str):
        """
        Args:
            name: The name of the measured object.
        """  # noqa: D205, D212, D415
        self.__name = name
        self._init_shared_memory_attrs_before()

    @contextmanager
    def record(self, linearize: bool = False) -> Generator[Any, Any, Any]:
        """Record execution statistics while executing code in a context manager.

        Args:
            linearize: Whether measuring execution for linearization.
        """
        if self.is_enabled:
            if linearize:
                self.__increment_n_linearizations()
            else:
                self.__increment_n_executions()
            with Timer() as timer:
                yield
            self.__add_duration(timer.elapsed_time, linearize)
        else:
            yield

    def __increment_n_executions(self) -> None:
        """Increment the number of executions by 1."""
        with self.__n_calls.get_lock():
            self.__n_calls.value += 1

    def __increment_n_linearizations(self) -> None:
        """Increment the number of linearizations by 1."""
        with self.__n_calls_linearize.get_lock():
            self.__n_calls_linearize.value += 1

    def __add_duration(self, duration: float, linearize: bool) -> None:
        """Add execution duration.

        Args:
            duration: The time duration to add.
            linearize: Whether it is for linearization.
        """
        with self.__duration.get_lock():
            self.__duration.value += duration

        time_stamps = ExecutionStatistics.time_stamps
        if time_stamps is not None:
            _time_stamps = time_stamps.setdefault(
                self.__name, get_multi_processing_manager().list()
            )
            current_time = default_timer()
            _time_stamps.append((current_time - duration, current_time, linearize))

    def __check_is_enabled(self) -> None:
        if not self.is_enabled:
            msg = (
                f"The execution statistics of the object named "
                f"{self.__name} are disabled."
            )
            raise RuntimeError(msg)

    @property
    def n_calls(self) -> int | None:
        """The number of executions.

        This property is multiprocessing safe.

        Raises:
            RuntimeError: If the statistics are disabled.
        """
        if self.is_enabled:
            return self.__n_calls.value
        return None

    @n_calls.setter
    def n_calls(self, value: int) -> None:
        self.__check_is_enabled()
        self.__n_calls.value = value

    @property
    def duration(self) -> float | None:
        """The cumulated execution duration.

        This is property is multiprocessing safe.

        Raises:
            RuntimeError: If the statistics are disabled.
        """
        if self.is_enabled:
            return self.__duration.value
        return None

    @duration.setter
    def duration(self, value: float) -> None:
        self.__check_is_enabled()
        self.__duration.value = value

    @property
    def n_calls_linearize(self) -> int | None:
        """The number of linearizations.

        This property is multiprocessing safe.

        Raises:
            RuntimeError: If the statistics are disabled.
        """
        if self.is_enabled:
            return self.__n_calls_linearize.value
        return None

    @n_calls_linearize.setter
    def n_calls_linearize(self, value: int) -> None:
        self.__check_is_enabled()
        self.__n_calls_linearize.value = value

    def _init_shared_memory_attrs_before(self) -> None:
        self.__duration = Value("d", 0.0)
        self.__n_calls = Value("i", 0)
        self.__n_calls_linearize = Value("i", 0)
