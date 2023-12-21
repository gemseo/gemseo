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

import h5py
import pytest
from numpy import array
from numpy import ones

from gemseo.caches.cache_factory import CacheFactory
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


def test_runtimerror(tmp_wd):
    h_cache1 = create_cache()
    h_cache2 = create_cache()
    data = {"a": ones(1)}
    group_name = h_cache1._INPUTS_GROUP
    hdf_node_path = "DummyCache"
    group_num = 1
    h_cache1._HDF5Cache__hdf_file.write_data(data, group_name, group_num, hdf_node_path)
    with pytest.raises((RuntimeError, IOError)):
        h_cache2._HDF5Cache__hdf_file.write_data(
            data, group_name, group_num, hdf_node_path
        )


def test_hasgroup(tmp_wd):
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    h5file_sing = cache._HDF5Cache__hdf_file

    h5file = h5py.File(h5file_sing.hdf_file_path, "a")

    hasgrp = h5file_sing._has_group(
        1, cache._INPUTS_GROUP, "Dummy", h5_open_file=h5file
    )
    assert hasgrp

    hasgrp = h5file_sing._has_group(
        2, cache._INPUTS_GROUP, "Dummy", h5_open_file=h5file
    )
    assert not hasgrp

    h5file_sing.write_data(
        {"i": ones(1)},
        cache._OUTPUTS_GROUP,
        2,
        "Dummy",
        h5_open_file=h5file,
    )


def test_repr(tmp_wd):
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


def test_repr_html(tmp_wd):
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


def test_cache_array_str(tmp_wd):
    """Test a cache with arrays of strings.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    cache = create_cache()
    inputs = {"i": array(["some_string"])}
    outputs = {"o": ones(1)}
    cache.cache_outputs(inputs, outputs)
    assert cache.last_entry[0] == inputs
    assert cache.last_entry[1] == outputs


def test_hdf_node_path(tmp_wd):
    """Check the property hdf_node_path."""
    assert create_cache().hdf_node_path == "Dummy"
    assert create_cache("foo").hdf_node_path == "foo"
