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
"""Fixtures for the tests of the caches."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gemseo.util.singleton import SingleInstancePerFileAttribute

if TYPE_CHECKING:
    from collections.abc import Callable


@pytest.fixture
def forget_file_singletons(monkeypatch) -> Callable[[], None]:
    """A function making the next caches use new file singletons.

    The process-wide registry of the singletons is replaced, not emptied,
    and restored at the end of the test,
    so that the other tests of the same process keep their singletons.
    """

    def forget() -> None:
        """Replace the registry of the file singletons, for the current test only."""
        monkeypatch.setattr(SingleInstancePerFileAttribute, "instances", {})

    return forget
