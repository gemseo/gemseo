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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Gilberto RUIZ
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ones

from gemseo.core.cache.factory import CacheFactory

if TYPE_CHECKING:
    from gemseo.core.cache.memory_full import MemoryFullCache


def create_cache() -> MemoryFullCache:
    """Create a MemoryFullCache.

    Returns:
        A MemoryFullCache.
    """
    return CacheFactory().create(
        "MemoryFullCache",
    )


@pytest.mark.parametrize("tolerance", [0, 0.01])
@pytest.mark.parametrize(
    ("cached", "queried"),
    [
        (array(["some_string"]), array(["some_string"])),
        ("some_string", "some_string"),
        (Path("some_path"), Path("some_path")),
        (Path("a") / "b", "a/b"),
        ("a/b", Path("a") / "b"),
    ],
    ids=["array", "str", "path", "path-to-str", "str-to-path"],
)
def test_cache_str(cached, queried, tolerance) -> None:
    """Check that a str, a path and a one-item string array hit the cache entry of
    the same string, with numeric variables.
    """
    cache = create_cache()
    cache.tolerance = tolerance
    outputs = {"o": ones(1)}
    cache.cache_outputs({"i": cached, "var_1": ones(1)}, outputs)
    assert cache.get({"i": queried, "var_1": ones(1)}).outputs == outputs


@pytest.mark.parametrize("tolerance", [0, 0.01])
def test_cache_path_miss(tolerance) -> None:
    """Check that a different path input is a cache miss."""
    cache = create_cache()
    cache.tolerance = tolerance
    cache.cache_outputs({"p": Path("some_path"), "x": ones(1)}, {"o": ones(1)})
    assert not cache.get({"p": Path("other_path"), "x": ones(1)}).outputs
