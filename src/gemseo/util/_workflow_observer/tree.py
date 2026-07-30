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
"""A tree of workflow observers."""

from __future__ import annotations

from os import getpid
from os import getppid
from queue import LifoQueue
from threading import current_thread
from typing import TYPE_CHECKING
from typing import Final

if TYPE_CHECKING:
    from gemseo.util._workflow_observer.base_observer import BaseWorkflowObserver


class ObserverTree:
    """A tree of observers.

    This class is used to store the parent-child relationships between observers.
    It is supposed to be used as a global object,
    in multi-threading and multiprocessing contexts.
    A parent can have one or more child observers, a child can only have one parent.
    Branches are `LifoQueue` objects,
    the tree has a queue like interface, where observers can be `put` and `pop`.
    """

    __parent_id_to_observer: Final[dict[int, LifoQueue[BaseWorkflowObserver]]] = {}
    """The map from parent ids to parent observer queues."""

    @property
    def parent(self) -> BaseWorkflowObserver | None:
        """The parent observer or `None` if there is no parent."""
        if not self.__parent_id_to_observer:
            return None

        # Determine the potential parent ids from younger to older.
        parent_process_ids = (getpid(), getppid())
        if (tid := getattr(current_thread(), "parent_id", None)) is None:
            parent_ids = parent_process_ids
        else:
            parent_ids = (tid, *parent_process_ids)

        # Find the younger parent.
        for parent_id in parent_ids:
            observer_queue = self.__parent_id_to_observer.get(parent_id)
            if observer_queue is not None:
                return observer_queue.queue[-1]

        msg = f"No parent observer found for {self}"
        raise RuntimeError(msg)

    def put(self, observer: BaseWorkflowObserver) -> None:
        """Push an observer to the tree.

        Args:
            observer: The observer to add to the tree.
        """
        queue = self.__parent_id_to_observer.setdefault(self.__get_id(), LifoQueue())
        queue.put(observer)

    def pop(self) -> None:
        """Remove the last observer from the tree.

        The observer queue for the current context is popped. If the queue becomes
        empty, it is removed from the tree.
        """
        parent_id_to_handlers = self.__parent_id_to_observer
        id_ = self.__get_id()
        queue = parent_id_to_handlers[id_]
        # TODO: block = False?
        queue.get()
        if queue.empty():
            del parent_id_to_handlers[id_]

    @staticmethod
    def __get_id() -> int:
        """Return the ID of the current context (thread or process).

        In a multithreaded context, the thread ID is returned if it differs from
        the parent thread ID. Otherwise, the process ID is returned.

        Returns:
            The ID of the current thread or process.
        """
        thread = current_thread()
        if (parent_thread_id := getattr(thread, "parent_id", None)) and (
            thread_id := thread.native_id
        ) != parent_thread_id:
            return thread_id
        return getpid()
