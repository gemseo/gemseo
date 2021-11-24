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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import re
import shutil
import sys
from typing import Generator

import h5py
import pytest
from numpy import arange, array, eye, float64, zeros
from numpy.linalg import norm
from six import PY2

from gemseo.api import create_scenario
from gemseo.caches.cache_factory import CacheFactory
from gemseo.caches.hdf5_cache import HDF5Cache, HDF5FileSingleton
from gemseo.core.cache import hash_data_dict, to_real
from gemseo.core.chain import MDOParallelChain
from gemseo.problems.sellar.sellar import Sellar1, SellarSystem
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
from gemseo.utils.py23_compat import Path, long, string_array, xrange


@pytest.fixture(scope="module")
def factory():
    return CacheFactory()


@pytest.fixture
def simple_cache(factory):
    return factory.create("SimpleCache", tolerance=0.0)


@pytest.fixture
def memory_full_cache(factory):
    return factory.create("MemoryFullCache")


@pytest.fixture
def memory_full_cache_loc(factory):
    return factory.create("MemoryFullCache", is_memory_shared=False)


@pytest.fixture
def hdf5_cache(factory, tmp_wd):
    return factory.create(
        "HDF5Cache", hdf_file_path="dummy.h5", hdf_node_path="DummyCache"
    )


def test_jac_and_outputs_caching(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
):
    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        assert cache.get_length() == 0
        assert not cache
        assert cache.get_last_cached_inputs() is None
        assert cache.get_last_cached_outputs() is None
        input_data = {"i": arange(3)}
        data_out = {"o": arange(4)}
        cache.cache_outputs(input_data, input_data.keys(), data_out)
        out, jac = cache.get_outputs(input_data, input_data.keys())
        assert out is not None
        assert jac is None
        assert_items_equal(out, data_out)

        jac = {"o": {"i": eye(4, 3)}}
        cache.cache_jacobian(input_data, input_data.keys(), jac)

        out, jac_l = cache.get_outputs(input_data, input_data.keys())
        assert jac_l is not None
        assert norm(jac_l["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
        assert out is not None

        cache.tolerance = 1e-6
        out_t, jac_t = cache.get_outputs(input_data, input_data.keys())
        assert jac_t is not None
        assert norm(jac_t["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
        assert out_t is not None

        cache.tolerance = 0.0

        # Cache jac again, to check it still works
        cache.cache_jacobian(input_data, input_data.keys(), jac)

    assert simple_cache.get_length() == 1

    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:

        assert cache.get_last_cached_inputs() is not None
        assert cache.get_last_cached_outputs() is not None

        cache.clear()
        assert cache.get_length() == 0


def test_hdf_cache_read(tmp_wd):
    cache = HDF5Cache("dummy.h5", "DummyCache")

    with pytest.raises(ValueError):
        cache.export_to_ggobi(
            "out.ggobi",
            inputs_names=["i", "j"],
            outputs_names=["o"],
        )

    n = 10
    for i in xrange(1, n + 1):
        input_data = {"i": i * arange(3), "j": array([1.0])}
        data_out = {"o": i * arange(4)}
        cache.cache_outputs(input_data, input_data.keys(), data_out)
    # Cache it again, just to check
    cache.cache_outputs(input_data, input_data.keys(), data_out)

    assert len(cache) == n
    cache_read = HDF5Cache("dummy.h5", "DummyCache")

    assert cache_read.get_length() == n

    cache.export_to_ggobi(
        "out1.ggobi",
        inputs_names=["i", "j"],
        outputs_names=["o"],
    )

    input_data = {"i": zeros(3), "k": zeros(3)}
    data_out = {"o": arange(4), "t": zeros(3)}
    cache.cache_outputs(input_data, input_data.keys(), data_out, data_out.keys())

    assert len(cache) == n + 1
    exp_ggobi = tmp_wd / "out2.ggobi"
    cache.export_to_ggobi(
        str(exp_ggobi), inputs_names=["i", "k"], outputs_names=["o", "t"]
    )
    assert exp_ggobi.exists()

    with pytest.raises(ValueError):
        cache.export_to_ggobi(
            "out3.ggobi",
            inputs_names=["izgz"],
            outputs_names=["o"],
        )


def test_merge_caches(tmp_wd):
    c1 = HDF5Cache("out11.h5", "DummyCache")
    c2 = HDF5Cache("out21.h5", "DummyCache")

    keys = ["i"]
    input_data = {"i": 10 * arange(3)}
    c1.cache_outputs(input_data, keys, {"o": arange(4)})

    input_data2 = {"i": 2 * arange(3)}
    c2.cache_outputs(input_data2, keys, {"o": 2 * arange(4)})
    c2.cache_jacobian(input_data2, keys, {"o": {"i": 3 * arange(4)}})

    c1.merge(c2)
    assert c1.get_length() == 2

    outs, jacs = c1.get_outputs(input_data2, keys)
    assert (outs["o"] == 2 * arange(4)).all()
    assert (jacs["o"]["i"] == 3 * arange(4)).all()


def test_collision(tmp_wd):
    c1 = HDF5Cache("out7.h5", "DummyCache")
    input_data = {"i": arange(3)}
    output_data = {"o": arange(3)}
    c1.cache_outputs(input_data, input_data.keys(), output_data)

    # Fake a collision of hashes
    c1.cache_outputs(input_data, input_data.keys(), output_data)
    hash_0 = next(iter(c1._hashes.keys()))
    groups = c1._hashes.pop(hash_0)
    input_data2 = {"i": 2 * arange(3)}
    hash_1 = hash_data_dict(input_data2, input_data2.keys())
    c1._hashes[hash_1] = groups
    output_data2 = {"o": 2.0 * arange(3)}
    c1.cache_outputs(input_data2, input_data2.keys(), output_data2)

    c1_outs = c1.get_outputs(input_data2, input_data2.keys())
    assert (c1_outs[0]["o"] == output_data2["o"]).all()


def test_hash_data_dict():
    input_data = {"i": 10 * arange(3)}
    hash_0 = hash_data_dict(input_data)
    if PY2:
        assert isinstance(hash_0, int)
    else:
        assert isinstance(hash_0, long)
    assert hash_0 == hash_data_dict(input_data)
    assert hash_0 == hash_data_dict({"i": 10 * arange(3), "t": None})
    assert hash_0 == hash_data_dict({"i": 10 * arange(3), "j": None})
    hash_data_dict({"i": 10 * arange(3)})
    # Discontiguous array
    hash_data_dict({"i": arange(10)[::3]})


def test_to_real():
    assert to_real(1.0j * arange(3)).dtype == float64
    assert to_real(1.0 * arange(3)).dtype == float64
    assert (to_real(1.0j * arange(3) + arange(3)) == 1.0 * arange(3)).all()


def test_write_data(tmp_wd):
    file_sing = HDF5FileSingleton("out2.h5")
    input_data = {"i": arange(3)}
    file_sing.write_data(input_data, input_data.keys(), "group", 1, "node", None)


#             h5_file = h5py.File(join(tmp_out, "out51.h5"), "a")
#             file_sing.write_data(input_data, input_data.keys(), "group2",
#                                  2, "node", h5_file)
#             assert not file_sing._has_group(3, "group", "node", h5_file)
#             file_sing.write_data(input_data, input_data.keys(), "group2",
#                                  2, "node", h5_file)
#             self.assertRaises((RuntimeError, IOError), file_sing.write_data,
#                               input_data, input_data.keys(), "group2",
#                               2, "node", h5_file)


def test_read_hashes(tmp_wd):
    file_sing = HDF5FileSingleton("out1.h5")
    input_data = {"i": arange(3)}
    file_sing.write_data(input_data, input_data.keys(), "group", 1, "node", None)
    assert file_sing.read_hashes({}, "unknown") == 0
    hashes_dict = {}
    file_sing.read_hashes(hashes_dict, "unknown")
    n_0 = len(hashes_dict)
    file_sing.read_hashes(hashes_dict, "unknown")
    assert n_0 == len(hashes_dict)
    hashes_dict = {}
    hashes_dict[977299934065931519957167197057685376965897664534] = []
    file_sing.read_hashes(hashes_dict, "node")


def test_read_group(tmp_wd):
    cache = HDF5Cache("out3.h5", "node")
    cache.cache_outputs(
        {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
    )
    input_data = {"x": arange(3), "y": arange(2)}
    cache._read_group([1], input_data)


def test_get_all_data(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
):
    inputs = {"x": arange(3), "y": arange(3)}
    outputs = {"f": array([1])}
    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:

        cache.cache_outputs(inputs, ["x", "y"], outputs, ["f"])
        all_data = cache.get_all_data()
        assert len(all_data) == 1
        assert_items_equal(all_data[1]["inputs"], inputs)
        assert_items_equal(all_data[1]["outputs"], outputs)
        assert all_data[1]["jacobian"] is None

        assert cache.get_outputs(inputs)[0]["f"][0] == outputs["f"][0]

    jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
    hdf5_cache.cache_jacobian({"x": arange(3), "y": arange(3)}, ["x", "y"], jac)


def test_all_data(memory_full_cache, memory_full_cache_loc, hdf5_cache):
    for cache in [hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        cache.cache_outputs(
            {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
        )
        jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
        cache.cache_jacobian({"x": arange(3), "y": arange(3)}, ["x", "y"], jac)
        cache.cache_outputs(
            {"x": arange(3) * 2, "y": arange(3)},
            ["x", "y"],
            {"f": array([2])},
            ["f"],
        )
        all_data = cache.get_all_data(True)
        data = next(all_data)
        assert_items_equal(data["inputs"], {"x": arange(3), "y": arange(3)})
        assert_items_equal(data["outputs"], {"f": array([1])})
        data = next(all_data)
        assert_items_equal(data["inputs"], {"x": arange(3) * 2, "y": arange(3)})
        assert_items_equal(data["outputs"], {"f": array([2])})


def assert_items_equal(data1, data2):
    assert len(data1) == len(data2)
    for key, val in data1.items():
        assert key in data2
        assert (val == data2[key]).all()


def test_addition():
    cache1 = CacheFactory().create("MemoryFullCache")
    cache1.cache_outputs(
        {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
    )
    cache2 = CacheFactory().create("MemoryFullCache")
    cache2.cache_outputs(
        {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
    )
    cache3 = cache1 + cache2
    assert cache3.get_length() == 1
    cache2.cache_outputs(
        {"x": arange(3) * 2, "y": arange(3)},
        ["x", "y"],
        {"f": array([1]) * 2},
        ["f"],
    )
    cache3 = cache1 + cache2
    assert cache3.get_length() == 2
    all_data = cache3.get_all_data(True)
    data = next(all_data)
    assert_items_equal(data["inputs"], {"x": arange(3), "y": arange(3)})
    assert_items_equal(data["outputs"], {"f": array([1])})
    data = next(all_data)
    assert_items_equal(data["inputs"], {"x": arange(3) * 2, "y": arange(3)})
    assert_items_equal(data["outputs"], {"f": array([2])})


def test_duplicate_from_scratch(memory_full_cache, hdf5_cache):
    hdf5_cache._duplicate_from_scratch()
    memory_full_cache._duplicate_from_scratch()


def test_multithreading(memory_full_cache, memory_full_cache_loc):
    caches = (memory_full_cache, memory_full_cache_loc)
    for c_1, c_2 in zip(caches, caches):
        s_s = SellarSystem()
        s_1 = Sellar1()
        s_1.cache = c_1
        s_s.cache = c_2
        assert len(c_1) == 0
        assert len(c_2) == 0
        par = MDOParallelChain([s_1, s_s], use_threading=True)
        ds = SellarDesignSpace("float64")
        scen = create_scenario(par, "DisciplinaryOpt", "obj", ds, scenario_type="DOE")

        options = {"algo": "fullfact", "n_samples": 10, "n_processes": 4}
        scen.execute(options)

        nexec_1 = s_1.n_calls
        nexec_2 = s_s.n_calls

        scen = create_scenario(par, "DisciplinaryOpt", "obj", ds, scenario_type="DOE")
        scen.execute(options)

        assert nexec_1 == s_1.n_calls
        assert nexec_2 == s_s.n_calls


def test_copy(memory_full_cache):
    input_data = {"i": arange(3)}
    data_out = {"o": arange(4)}
    memory_full_cache.cache_outputs(input_data, input_data.keys(), data_out)
    input_data = {"i": arange(4)}
    data_out = {"o": arange(5)}
    memory_full_cache.cache_outputs(input_data, input_data.keys(), data_out)
    copy = memory_full_cache.copy
    assert copy.get_length() == 2
    input_data = {"i": arange(5)}
    data_out = {"o": arange(6)}
    copy.cache_outputs(input_data, input_data.keys(), data_out)
    assert copy.get_length() == 3
    assert memory_full_cache.get_length() == 2


def test_hdf5singleton(tmp_wd):
    node = "node"
    file_path = "singleton.h5"
    CacheFactory().create("HDF5Cache", hdf_file_path=file_path, hdf_node_path=node)
    singleton = HDF5FileSingleton(file_path)
    data = {"x": array([0.0])}
    singleton.write_data(data, ["x"], HDF5Cache.INPUTS_GROUP, 1, node)
    with pytest.raises(RuntimeError):
        singleton.write_data(
            data,
            ["x"],
            HDF5Cache.INPUTS_GROUP,
            1,
            node,
        )


def test_cache_max_length(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
):
    """Tests the maximum length getter."""
    assert simple_cache.max_length == 1
    assert hdf5_cache.max_length == sys.maxsize
    assert memory_full_cache.max_length == sys.maxsize
    assert memory_full_cache_loc.max_length == sys.maxsize


def test_hash_data_dict_keys():
    """Check that hash considers the keys of the dictionary."""
    data = {"a": array([1]), "b": array([2])}
    assert hash_data_dict(data) == hash_data_dict({"a": array([1]), "b": array([2])})
    assert hash_data_dict(data) != hash_data_dict({"a": array([1]), "c": array([2])})
    assert hash_data_dict(data) != hash_data_dict({"a": array([1]), "b": array([3])})

    data = {"a": array([1]), "b": array([1])}
    assert hash_data_dict(data) != hash_data_dict({"a": array([1]), "c": array([1])})


CACHE_FILE_NAME = "cache.h5"


@pytest.fixture
def h5_file(tmp_wd):  # type: (...) -> Generator[h5py.File]
    """Provide an empty h5 file object and close it afterward."""
    h5_file = h5py.File(CACHE_FILE_NAME, mode="a")
    yield h5_file
    h5_file.close()


def test_check_version_new_file():
    """Verify that a non existing file passes the file format version check."""
    HDF5FileSingleton("foo")


def test_check_version_empty_file(h5_file):
    """Verify that an empty file passes the file format version check."""
    HDF5FileSingleton(CACHE_FILE_NAME)


def test_check_version_missing(h5_file):
    """Verify that an non empty file with no file format version raises."""
    h5_file["foo"] = "bar"

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The file cache.h5 cannot be used because it has no file format version: "
            "see HDFCache.update_file_format to convert it."
        ),
    ):
        HDF5FileSingleton(CACHE_FILE_NAME)


def test_check_version_greater(h5_file):
    """Verify that an non empty file with greater file format version raises."""
    h5_file["foo"] = "bar"
    h5_file.attrs["version"] = HDF5FileSingleton.FILE_FORMAT_VERSION + 1

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The file cache.h5 cannot be used because its file format version is 2 "
            "while the expected version is 1: "
            "see HDFCache.update_file_format to convert it."
        ),
    ):
        HDF5FileSingleton(CACHE_FILE_NAME)


def test_update_file_format(tmp_wd):
    """Check that updating format changes both hashes and version tag."""
    singleton = HDF5FileSingleton(CACHE_FILE_NAME)
    singleton.write_data(
        {"x": array([1.0]), "y": array([2.0, 3.0])},
        ["x", "y"],
        HDF5FileSingleton.INPUTS_GROUP,
        1,
        "foo",
    )
    singleton.write_data(
        {"x": array([2.0]), "y": array([3.0, 4.0])},
        ["x", "y"],
        HDF5FileSingleton.INPUTS_GROUP,
        2,
        "foo",
    )
    with h5py.File(CACHE_FILE_NAME, mode="r+") as h5_file:
        old_hash_1 = string_array([1])
        old_hash_2 = string_array([2])
        h5_file["foo"]["1"][HDF5FileSingleton.HASH_TAG][0] = old_hash_1
        h5_file["foo"]["2"][HDF5FileSingleton.HASH_TAG][0] = old_hash_2

    HDF5Cache.update_file_format(CACHE_FILE_NAME)
    with h5py.File(CACHE_FILE_NAME, mode="a") as h5_file:
        assert h5_file.attrs["version"] == HDF5FileSingleton.FILE_FORMAT_VERSION
        assert h5_file["foo"]["1"][HDF5FileSingleton.HASH_TAG][0] != old_hash_1
        assert h5_file["foo"]["2"][HDF5FileSingleton.HASH_TAG][0] != old_hash_2


def test_update_file_format_from_deprecated_file(tmp_wd):
    """Check that a file with a deprecated format can be correctly updated."""
    deprecated_cache_path = "cache_with_deprecated_format.h5"
    # This file has been obtained with GEMSEO 3.1.0:
    #     cache = HDF5Cache("cache_with_deprecated_format.h5", "node")
    #     cache.cache_outputs({'x': array([1.])}, ['x'], {'y': array([2.])}, ['y'])

    shutil.copy(
        str(Path(__file__).parent / deprecated_cache_path), deprecated_cache_path
    )
    HDF5Cache.update_file_format(deprecated_cache_path)

    cache_path = tmp_wd / "cache.h5"
    cache = HDF5Cache(str(cache_path), "node")
    input_data = {"x": array([1.0])}
    cache.cache_outputs(input_data, ["x"], {"y": array([2.0])}, ["y"])

    file_format_version = HDF5FileSingleton.FILE_FORMAT_VERSION
    hash_tag = HDF5FileSingleton.HASH_TAG
    inputs_group = HDF5FileSingleton.INPUTS_GROUP
    outputs_group = HDF5FileSingleton.OUTPUTS_GROUP
    with h5py.File(str(cache_path), mode="a") as h5_file:
        with h5py.File(str(deprecated_cache_path), mode="a") as deprecated_h5_file:
            assert h5_file.attrs["version"] == file_format_version
            assert deprecated_h5_file.attrs["version"] == file_format_version
            new_h5 = h5_file["node"]["1"]
            old_h5 = h5_file["node"]["1"]
            assert new_h5[hash_tag][0] == old_h5[hash_tag][0]
            assert new_h5[inputs_group]["x"][0] == old_h5[inputs_group]["x"][0]
            assert new_h5[outputs_group]["y"][0] == old_h5[outputs_group]["y"][0]
