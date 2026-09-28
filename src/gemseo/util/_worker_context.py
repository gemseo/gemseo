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
"""Tools for the context of worker processes and threads.

This module owns the metadata protocol of the workers created by the parallel
execution machinery: a worker process or thread is tagged, by the pool
initializers, with the id (``parent_id``) and the working directory at
submission time (``parent_path``) of its parent, as attributes of the process
or thread object. The directory manager additionally records the working
directory of a worker thread as a ``cwd`` attribute, since the process-wide
working directory is shared among threads, thus no longer reliable there.

These attributes shall only be accessed through the functions below.
"""

from __future__ import annotations

from multiprocessing import current_process
from multiprocessing import parent_process
from os import getpid
from threading import current_thread
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def is_in_worker_process(initial_process_id: int | None = None) -> bool:
    """Return whether the current process is a worker process.

    A worker is recognized either from `multiprocessing`, whichever start
    method created it, including during the bootstrap phase of a spawned
    process (where [parent_process][multiprocessing.parent_process] is not
    set yet and ``_inheriting`` is the stdlib marker of that phase), or
    from a change of process id since ``initial_process_id`` was captured,
    which also covers a bare [fork][os.fork].

    Args:
        initial_process_id: The process id captured at a reference point in
            time, e.g. when the caller was constructed. If ``None``, the
            process id is not used for the detection.

    Returns:
        Whether the current process is a worker process.
    """
    return (
        parent_process() is not None
        or getattr(current_process(), "_inheriting", False)
        or (initial_process_id is not None and getpid() != initial_process_id)
    )


def tag_process_worker(parent_id: int, parent_path: Path) -> None:
    """Tag the current process with its parent metadata.

    Args:
        parent_id: The process id of the parent process.
        parent_path: The working directory of the parent at submission time.
    """
    process = current_process()
    process.parent_id = parent_id  # type: ignore[attr-defined]
    process.parent_path = parent_path  # type: ignore[attr-defined]


def tag_thread_worker(parent_id: int, parent_path: Path) -> None:
    """Tag the current thread with its parent metadata.

    Args:
        parent_id: The native id of the parent thread.
        parent_path: The working directory of the parent at submission time.
    """
    thread = current_thread()
    thread.parent_id = parent_id  # type: ignore[attr-defined]
    thread.parent_path = parent_path  # type: ignore[attr-defined]


def get_process_parent_path() -> Path | None:
    """Return the parent working directory of the current process.

    Returns:
        The working directory of the parent at submission time,
        or ``None`` if the current process is not a tagged worker.
    """
    return getattr(current_process(), "parent_path", None)


def get_thread_parent_path() -> Path | None:
    """Return the parent working directory of the current thread.

    Returns:
        The working directory of the parent at submission time,
        or ``None`` if the current thread is not a tagged worker.
    """
    return getattr(current_thread(), "parent_path", None)


def get_thread_parent_id() -> int | None:
    """Return the parent id of the current thread.

    Returns:
        The native id of the parent thread,
        or ``None`` if the current thread is not a tagged worker.
    """
    return getattr(current_thread(), "parent_id", None)


def get_thread_cwd() -> Path | None:
    """Return the working directory recorded on the current thread.

    Returns:
        The working directory recorded by
        [set_thread_cwd][gemseo.util._worker_context.set_thread_cwd],
        or ``None`` if none was recorded.
    """
    return getattr(current_thread(), "cwd", None)


def set_thread_cwd(path: Path) -> None:
    """Record the working directory on the current thread.

    The directory is only recorded when the current thread is a tagged
    worker: elsewhere, the process-wide working directory is reliable and
    thread-local tracking is not needed.

    Args:
        path: The path to record as the working directory of the thread.
    """
    thread = current_thread()
    if hasattr(thread, "parent_path"):
        thread.cwd = path  # type: ignore[attr-defined]
