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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy import ones

from gemseo.caches.factory import CacheFactory
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

if TYPE_CHECKING:
    from gemseo.caches.hdf5_cache import HDF5Cache


def create_cache(hdf_node_path="Dummy") -> HDF5Cache:
    """Create an HDF5 cache.

    Args:
        hdf_node_path: The name of the HDF node.

    Returns:
        An HDF5 cache.
    """
    return CacheFactory().create(
        "HDF5Cache", hdf_file_path="dummy.h5", hdf_node_path=hdf_node_path
    )


def test_runtimerror(tmp_wd) -> None:
    h_cache1 = create_cache()
    h_cache2 = create_cache()
    data = {"a": ones(1)}
    group_name = h_cache1.Group.INPUTS
    hdf_node_path = "DummyCache"
    group_num = 1
    h_cache1.hdf_file.write_data(data, group_name, group_num, hdf_node_path)
    with pytest.raises((RuntimeError, IOError)):
        h_cache2.hdf_file.write_data(data, group_name, group_num, hdf_node_path)


def test_hasgroup(tmp_wd) -> None:
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    hasgrp = cache.hdf_file.has_group(1, cache.Group.INPUTS, "Dummy")
    assert hasgrp

    hasgrp = cache.hdf_file.has_group(2, cache.Group.INPUTS, "Dummy")
    assert not hasgrp

    cache.hdf_file.write_data(
        {"i": ones(1)},
        cache.Group.OUTPUTS,
        2,
        "Dummy",
    )


def test_repr(tmp_wd) -> None:
    """Check string representation."""
    cache = create_cache()
    cache[{"i": ones(1)}] = ({"o": ones(1)}, None)
    cache[{"i": ones(2)}] = ({"o": ones(2)}, None)
    expected = """Name: Dummy
   Type: HDF5Cache
   Tolerance: 0.0
   Input names: ['i']
   Output names: ['o']
   Length: 2
   HDF file path: dummy.h5
   HDF node path: Dummy"""
    assert repr(cache) == str(cache) == expected


def test_repr_html(tmp_wd) -> None:
    """Check HDF5Cache._repr_html."""
    cache = create_cache()
    cache[{"i": ones(1)}] = ({"o": ones(1)}, None)
    cache[{"i": ones(2)}] = ({"o": ones(2)}, None)
    assert cache._repr_html_() == REPR_HTML_WRAPPER.format(
        "Name: Dummy<br/>"
        "<ul>"
        "<li>Type: HDF5Cache</li>"
        "<li>Tolerance: 0.0</li>"
        "<li>Input names: [&#x27;i&#x27;]</li>"
        "<li>Output names: [&#x27;o&#x27;]</li>"
        "<li>Length: 2</li>"
        "<li>HDF file path: dummy.h5</li>"
        "<li>HDF node path: Dummy</li>"
        "</ul>"
    )


@pytest.mark.parametrize("tolerance", [0, 0.01])
def test_cache_str(tmp_wd, tolerance) -> None:
    """Test a cache with strings in different formats and with numeric variables.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    cache = create_cache()
    cache.tolerance = tolerance
    inputs = {"i": array(["some_string"]), "var_1": ones(1)}
    outputs = {"o": ones(1)}
    cache.cache_outputs(inputs, outputs)
    assert cache.get(inputs)[0] == inputs
    assert cache.get(inputs)[1] == outputs


def test_hdf_node_path(tmp_wd) -> None:
    """Check the property hdf_node_path."""
    assert create_cache().hdf_node_path == "Dummy"
    assert create_cache("foo").hdf_node_path == "foo"
