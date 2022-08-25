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
"""Tests for the timer."""
from __future__ import annotations

import re
from logging import getLevelName
from time import sleep

import pytest
from gemseo.utils.timer import Timer


@pytest.fixture(scope="module")
def timer() -> Timer:
    """A timer."""
    with Timer() as timer:
        sleep(1)

    return timer


def test_elapsed_time(timer):
    """Check the elapsed_time attribute."""
    assert timer.elapsed_time == pytest.approx(1.0, abs=0.1)


def test_str(timer):
    """Check the string representation of the timer."""
    assert re.compile(r"Elapsed time: .* s\.").match(str(timer))


def test_no_log(caplog):
    """Check that there is no log by default."""
    with Timer():
        sleep(0.1)

    assert not caplog.records


@pytest.mark.parametrize("level", ["INFO", "DEBUG"])
def test_log(caplog, level):
    """Check the logs."""
    caplog.set_level(level)
    with Timer(level):
        sleep(0.1)

    log = caplog.record_tuples[0]
    assert log[1] == getLevelName(level)
    assert re.compile(r"Elapsed time: .* s\.").match(log[2])
