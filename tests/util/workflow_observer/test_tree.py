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

"""Tests for the observer tree."""

from __future__ import annotations

from os import getpid
from queue import LifoQueue

import pytest

from gemseo.util._workflow_observer.tree import ObserverTree
from gemseo.util.testing.helper import assert_exception

_STATE_ATTR = "_ObserverTree__parent_id_to_observer"


@pytest.fixture
def isolated_tree(monkeypatch):
    monkeypatch.setattr(ObserverTree, _STATE_ATTR, {})
    return ObserverTree()


def test_parent_is_none_when_empty(isolated_tree):
    assert isolated_tree.parent is None


def test_parent_returns_top_of_current_pid_queue(monkeypatch):
    sentinel = object()
    queue: LifoQueue = LifoQueue()
    queue.put(object())
    queue.put(sentinel)
    monkeypatch.setattr(ObserverTree, _STATE_ATTR, {getpid(): queue})
    assert ObserverTree().parent is sentinel


def test_parent_raises_when_no_id_matches(monkeypatch, snapshot):
    queue: LifoQueue = LifoQueue()
    queue.put(object())
    monkeypatch.setattr(ObserverTree, _STATE_ATTR, {-1: queue})
    with assert_exception(RuntimeError, snapshot):
        _ = ObserverTree().parent


def test_parent_uses_thread_parent_id_when_set(monkeypatch):
    sentinel = object()
    queue: LifoQueue = LifoQueue()
    queue.put(sentinel)
    thread_parent_id = 424242
    monkeypatch.setattr(ObserverTree, _STATE_ATTR, {thread_parent_id: queue})

    class _FakeThread:
        parent_id = thread_parent_id

    monkeypatch.setattr(
        "gemseo.util._workflow_observer.tree.current_thread", lambda: _FakeThread()
    )
    assert ObserverTree().parent is sentinel


def test_put_adds_observer_to_current_pid_queue(isolated_tree):
    sentinel = object()
    isolated_tree.put(sentinel)
    state = getattr(ObserverTree, _STATE_ATTR)
    assert getpid() in state
    assert state[getpid()].queue[-1] is sentinel


def test_pop_removes_top_observer(isolated_tree):
    first = object()
    second = object()
    isolated_tree.put(first)
    isolated_tree.put(second)
    isolated_tree.pop()
    state = getattr(ObserverTree, _STATE_ATTR)
    assert state[getpid()].queue[-1] is first


def test_pop_deletes_empty_queue_entry(isolated_tree):
    isolated_tree.put(object())
    isolated_tree.pop()
    assert getpid() not in getattr(ObserverTree, _STATE_ATTR)
