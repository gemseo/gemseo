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

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ones

from gemseo.caches.factory import CacheFactory

if TYPE_CHECKING:
    from gemseo.caches.simple_cache import SimpleCache


def create_cache() -> SimpleCache:
    """Create a SimpleCache.

    Returns:
        A SimpleCache.
    """
    return CacheFactory().create("SimpleCache")


@pytest.mark.parametrize("input_string", [array(["some_string"]), "some_string"])
@pytest.mark.parametrize("tolerance", [0, 0.01])
def test_cache_str(input_string, tolerance) -> None:
    """Test a cache with strings in different formats and with numeric variables.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    cache = create_cache()
    cache.tolerance = tolerance
    inputs = {"i": input_string, "var_1": ones(1)}
    outputs = {"o": ones(1)}
    cache.cache_outputs(inputs, outputs)
    cache.cache_outputs(inputs, outputs)
    assert cache.get(inputs)[0] == inputs
    assert cache.get(inputs)[1] == outputs
