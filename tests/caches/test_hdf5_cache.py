# -*- coding: utf-8 -*-
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
from __future__ import division, unicode_literals

from os.path import join

import h5py
import pytest
from numpy import ones

from gemseo.caches.cache_factory import CacheFactory


def create_cache(tmp_path, h5_node="Dummy"):
    factory = CacheFactory()
    hdf_file_path = join(str(tmp_path), "dummy.h5")
    return factory.create(
        "HDF5Cache", hdf_file_path=hdf_file_path, hdf_node_path=h5_node
    )


def test_runtimerror(tmp_path):
    h_cache1 = create_cache(tmp_path, "Dummy")
    h_cache2 = create_cache(tmp_path, "Dummy")
    data = {"a": ones(1)}
    data_names = ["a"]
    group_name = h_cache1.INPUTS_GROUP
    hdf_node_path = "DummyCache"
    group_num = 1
    h_cache1._hdf_file.write_data(
        data, data_names, group_name, group_num, hdf_node_path
    )
    with pytest.raises((RuntimeError, IOError)):
        h_cache2._hdf_file.write_data(
            data, data_names, group_name, group_num, hdf_node_path
        )


def test_hasgroup(tmp_path):
    cache = create_cache(tmp_path, "Dummy")
    cache.cache_outputs({"i": ones(1)}, ["i"], {"o": ones(1)})
    h5file_sing = cache._hdf_file

    h5file = h5py.File(h5file_sing.hdf_file_path, "a")

    hasgrp = h5file_sing._has_group(1, cache.INPUTS_GROUP, "Dummy", h5file=h5file)
    assert hasgrp

    hasgrp = h5file_sing._has_group(2, cache.INPUTS_GROUP, "Dummy", h5file=h5file)
    assert not hasgrp

    h5file_sing.write_data(
        {"i": ones(1)},
        ["i"],
        group_name=cache.OUTPUTS_GROUP,
        group_num=2,
        hdf_node_path="Dummy",
        h5_open_file=h5file,
    )
