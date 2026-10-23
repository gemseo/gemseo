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

"""Tests for the tracer object ids resolved inside a worker process."""

from __future__ import annotations

from multiprocessing import get_all_start_methods
from multiprocessing import get_context
from types import MappingProxyType
from types import SimpleNamespace
from typing import TYPE_CHECKING
from typing import Final

import pytest

from gemseo.util._tracer.base import BaseTracer
from gemseo.util._tracer.registry import TraceRegistry
from gemseo.util.base_multiton import BaseMultiton

if TYPE_CHECKING:
    from multiprocessing.queues import Queue

    from gemseo.util.typing import StrKeyMapping

_no_init_arguments: Final[StrKeyMapping] = MappingProxyType({})
"""No constructor arguments, the tracer under test reads none."""

_timeout: Final[float] = 60.0
"""The number of seconds to wait for the worker process."""


class _Observee:
    """A fake observee registered in the worker process."""


class _WorkerTracer(BaseTracer):
    """A minimal concrete tracer, only used for the id it resolves."""


def _put_object_ids(queue: Queue) -> None:
    """Resolve two object ids after a registry reset and send them back.

    Runs in the worker process. Two distinct observees of one class are
    registered around the eviction of the registry from the `BaseMultiton`
    cache, which `Settings.__reset_directory_manager` performs whenever the
    directory manager is (re-)enabled; the id of the first one is then stale
    and must not be reused.

    Args:
        queue: The queue carrying the resolved ids back to the parent process.
    """
    first_observee = _Observee()
    _WorkerTracer(SimpleNamespace(object_=first_observee), _no_init_arguments)

    # Simulate the reset performed by `Settings.__reset_directory_manager`.
    BaseMultiton.clear_cache(TraceRegistry)

    second_observee = _Observee()
    _WorkerTracer(SimpleNamespace(object_=second_observee), _no_init_arguments)
    # A tracer built for the first observee after the reset.
    _WorkerTracer(SimpleNamespace(object_=first_observee), _no_init_arguments)

    queue.put((
        first_observee._workflow_trace_id,
        second_observee._workflow_trace_id,
    ))


@pytest.mark.skip_under_windows
@pytest.mark.skipif(
    "fork" not in get_all_start_methods(), reason="requires the fork start method"
)
def test_resolve_object_id_re_registers_a_stale_id_in_a_worker_process():
    """Verify that a stale id is re-registered in a worker process too.

    The staleness of a stamped id is decided from the process that stamped it,
    not from whether the current process is a worker: in a worker, every
    stamped id would otherwise look fresh, and two distinct observees of one
    class would collide on one id after a registry reset.
    """
    context = get_context("fork")
    queue = context.Queue()
    process = context.Process(target=_put_object_ids, args=(queue,))
    process.start()
    try:
        first_id, second_id = queue.get(timeout=_timeout)
    finally:
        process.join(timeout=_timeout)

    # The second observee, registered first in the new registry, gets its
    # counter's first id; the first observee is re-registered instead of
    # keeping the id it collides with.
    assert second_id.startswith("Observee/")
    assert first_id != second_id
