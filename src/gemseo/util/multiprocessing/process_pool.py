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
"""Pools of worker processes."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.multiprocessing.start_method import MultiProcessingStartMethod
from gemseo.util.platform import platform_is_windows

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Iterable
    from typing import Any

max_windows_processes: Final[int] = 61
"""The maximum number of worker processes of a `ProcessPoolExecutor` on Windows.

It mirrors `concurrent.futures.process._MAX_WINDOWS_WORKERS`,
which is private and therefore not imported.
"""


def check_start_method(method: MultiProcessingStartMethod) -> None:
    """Check a multiprocessing start method with respect to the platform.

    Args:
        method: The start method.

    Raises:
        ValueError: If the start method is different from `spawn`
            on the Windows platform.
    """
    if (
        platform_is_windows and method != MultiProcessingStartMethod.SPAWN
    ):  # pragma: win32 cover
        msg = (
            f"The multiprocessing start method "
            f"{method.value} "
            f"cannot be used on the Windows platform. "
            f"Only {MultiProcessingStartMethod.SPAWN.value} is available."
        )
        raise ValueError(msg)


def get_number_of_processes(n_processes: int, n_tasks: int) -> int:
    """Return the number of worker processes of a pool.

    Never spawn more workers than tasks:
    extra workers cost a process creation and a pickling of the initializer
    arguments for no work.
    On Windows, `ProcessPoolExecutor` rejects more than 61 worker processes.

    Args:
        n_processes: The requested number of worker processes.
        n_tasks: The number of tasks to dispatch.

    Returns:
        The number of worker processes.
    """
    n_processes = min(n_tasks, n_processes)
    if platform_is_windows:
        n_processes = min(n_processes, max_windows_processes)

    return n_processes


def create_process_pool_executor(
    n_processes: int,
    method: MultiProcessingStartMethod,
    initializer: Callable[..., None],
    initargs: Iterable[Any],
) -> ProcessPoolExecutor:
    """Create a pool of worker processes.

    Args:
        n_processes: The number of worker processes.
        method: The multiprocessing start method.
        initializer: The function called at the start of each worker process.
        initargs: The arguments of the initializer.

    Returns:
        The pool of worker processes.

    Raises:
        ValueError: If the start method is different from `spawn`
            on the Windows platform.
    """
    check_start_method(method)
    return ProcessPoolExecutor(
        max_workers=n_processes,
        mp_context=get_context(method=method),
        initializer=initializer,
        initargs=tuple(initargs),
    )
