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
from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Iterator

import h5py
import pytest
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.caches.cache_factory import CacheFactory
from gemseo.caches.hdf5_cache import HDF5Cache
from gemseo.caches.hdf5_file_singleton import HDF5FileSingleton
from gemseo.core.cache import CacheEntry
from gemseo.core.cache import hash_data_dict
from gemseo.core.cache import to_real
from gemseo.core.chain import MDOParallelChain
from gemseo.problems.sellar.sellar import Sellar1
from gemseo.problems.sellar.sellar import SellarSystem
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
from numpy import arange
from numpy import array
from numpy import eye
from numpy import float64
from numpy import zeros
from numpy.linalg import norm

DIR_PATH = Path(__file__).parent


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
        assert not cache
        assert not cache.last_entry.inputs
        input_data = {"i": arange(3)}
        output_data = {"o": arange(4)}
        cache.cache_outputs(input_data, output_data)
        _, out, jac = cache[input_data]
        assert out
        assert not jac
        assert_items_equal(out, output_data)

        jac = {"o": {"i": eye(4, 3)}}
        cache.cache_jacobian(input_data, jac)

        _, out, jac_l = cache[input_data]
        assert jac_l is not None
        assert norm(jac_l["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
        assert out is not None

        cache.tolerance = 1e-6
        _, out_t, jac_t = cache[input_data]
        assert jac_t is not None
        assert norm(jac_t["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
        assert out_t is not None

        cache.tolerance = 0.0

        # Cache jac again, to check it still works
        cache.cache_jacobian(input_data, jac)

    assert len(simple_cache) == 1

    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:

        last_entry = cache.last_entry
        assert last_entry.inputs
        assert last_entry.outputs

        cache.clear()
        assert not cache


def test_hdf_cache_read(tmp_wd):
    cache = HDF5Cache("dummy.h5", "DummyCache")

    with pytest.raises(ValueError):
        cache.export_to_ggobi(
            "out.ggobi",
            input_names=["i", "j"],
            output_names=["o"],
        )

    n = 10
    for i in range(1, n + 1):
        cache.cache_outputs(
            {"i": i * arange(3), "j": array([1.0])}, {"o": i * arange(4)}
        )
    # Cache it again, just to check
    cache.cache_outputs({"i": n * arange(3), "j": array([1.0])}, {"o": n * arange(4)})

    assert len(cache) == n
    cache_read = HDF5Cache("dummy.h5", "DummyCache")

    assert len(cache_read) == n

    cache.export_to_ggobi(
        "out1.ggobi",
        input_names=["i", "j"],
        output_names=["o"],
    )

    input_data = {"i": zeros(3), "k": zeros(3)}
    output_data = {"o": arange(4), "t": zeros(3)}
    cache.cache_outputs(input_data, output_data)

    assert len(cache) == n + 1
    exp_ggobi = Path("out2.ggobi")
    cache.export_to_ggobi(
        str(exp_ggobi), input_names=["i", "k"], output_names=["o", "t"]
    )
    assert exp_ggobi.exists()

    with pytest.raises(ValueError):
        cache.export_to_ggobi(
            "out3.ggobi",
            input_names=["izgz"],
            output_names=["o"],
        )


def test_update_caches(tmp_wd):
    c1 = HDF5Cache("out11.h5", "DummyCache")
    c2 = HDF5Cache("out21.h5", "DummyCache")

    c1.cache_outputs({"i": 10 * arange(3)}, {"o": arange(4)})

    input_data2 = {"i": 2 * arange(3)}
    c2.cache_outputs(input_data2, {"o": 2 * arange(4)})
    c2.cache_jacobian(input_data2, {"o": {"i": 3 * arange(4)}})

    c1.update(c2)
    assert len(c1) == 2

    _, outs, jacs = c1[input_data2]
    assert (outs["o"] == 2 * arange(4)).all()
    assert (jacs["o"]["i"] == 3 * arange(4)).all()


def test_collision(tmp_wd):
    c1 = HDF5Cache("out7.h5", "DummyCache")
    input_data = {"i": arange(3)}
    output_data = {"o": arange(3)}
    c1.cache_outputs(input_data, output_data)

    # Fake a collision of hashes
    c1.cache_outputs(input_data, output_data)
    hash_0 = next(iter(c1._hashes_to_indices.keys()))
    groups = c1._hashes_to_indices.pop(hash_0)
    input_data2 = {"i": 2 * arange(3)}
    hash_1 = hash_data_dict(input_data2)
    c1._hashes_to_indices[hash_1] = groups
    output_data2 = {"o": 2.0 * arange(3)}
    c1.cache_outputs(input_data2, output_data2)

    _, c1_outs, _ = c1[input_data2]
    assert (c1_outs["o"] == output_data2["o"]).all()


def test_hash_data_dict():
    input_data = {"i": 10 * arange(3)}
    hash_0 = hash_data_dict(input_data)
    assert isinstance(hash_0, int)
    assert hash_0 == hash_data_dict(input_data)
    assert hash_0 == hash_data_dict({"i": 10 * arange(3), "t": None})
    assert hash_0 == hash_data_dict({"i": 10 * arange(3), "j": None})
    hash_data_dict({"i": 10 * arange(3)})
    # Discontiguous array
    hash_data_dict({"i": arange(10)[::3]})


@pytest.mark.parametrize(
    "input_c,input_f",
    [
        (array([[1, 2], [3, 4]], order="C"), array([[1, 2], [3, 4]], order="F")),
        (
            array([[1.0, 2.0], [3.0, 4.0]], order="C"),
            array([[1.0, 2.0], [3.0, 4.0]], order="F"),
        ),
    ],
)
def test_hash_discontiguous_array(input_c, input_f):
    """Test that the hashes are the same for discontiguous arrays.

    Args:
        input_c: A C-contiguous array.
        input_f: A Fortran ordered array.
    """
    assert hash_data_dict({"i": input_c}) == hash_data_dict({"i": input_f})


def func(x: int | float) -> int | float:
    """Dummy function to test the cache."""
    y = x
    return y


@pytest.mark.parametrize(
    "hdf_name,inputs,expected",
    [
        ("int_win.h5", array([1, 2, 3]), array([1, 2, 3])),
        ("int_linux.h5", array([1, 2, 3]), array([1, 2, 3])),
        ("float_win.h5", array([1.0, 2.0, 3.0]), array([1.0, 2.0, 3.0])),
        ("float_linux.h5", array([1.0, 2.0, 3.0]), array([1.0, 2.0, 3.0])),
    ],
)
def test_det_hash(tmp_wd, hdf_name, inputs, expected):
    """Test that hashed values are the same across sessions.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    # Use a temporary copy of the file in case the test fails.
    shutil.copy(str(DIR_PATH / hdf_name), tmp_wd)
    disc = create_discipline("AutoPyDiscipline", py_func=func)
    disc.set_cache_policy("HDF5Cache", cache_hdf_file=hdf_name)
    out = disc.execute({"x": inputs})

    assert disc.n_calls == 0
    assert out["y"].all() == expected.all()


def test_to_real():
    assert to_real(1.0j * arange(3)).dtype == float64
    assert to_real(1.0 * arange(3)).dtype == float64
    assert (to_real(1.0j * arange(3) + arange(3)) == 1.0 * arange(3)).all()


def test_write_data(tmp_wd):
    file_sing = HDF5FileSingleton("out2.h5")
    input_data = {"i": arange(3)}
    file_sing.write_data(input_data, "group", 1, "node")


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
    file_sing.write_data(input_data, "group", 1, "node")
    assert file_sing.read_hashes({}, "unknown") == 0
    hashes_dict = {}
    file_sing.read_hashes(hashes_dict, "unknown")
    n_0 = len(hashes_dict)
    file_sing.read_hashes(hashes_dict, "unknown")
    assert n_0 == len(hashes_dict)
    hashes_dict = {977299934065931519957167197057685376965897664534: []}
    file_sing.read_hashes(hashes_dict, "node")


def test_read_group(tmp_wd):
    cache = HDF5Cache("out3.h5")
    cache.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})
    cache._read_input_output_data([1], {"x": arange(3), "y": arange(2)})


def test_get_all_data(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
):
    inputs = {"x": arange(3), "y": arange(3)}
    outputs = {"f": array([1])}
    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        cache.cache_outputs(inputs, outputs)
        all_data = list(cache)
        assert len(all_data) == 1
        assert_items_equal(all_data[0].inputs, inputs)
        assert_items_equal(all_data[0].outputs, outputs)
        assert not all_data[0].jacobian
        assert cache[inputs].outputs["f"] == outputs["f"]

    jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
    hdf5_cache.cache_jacobian({"x": arange(3), "y": arange(3)}, jac)


def test_all_data(memory_full_cache, memory_full_cache_loc, hdf5_cache):
    for cache in [hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
        cache.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})
        cache.cache_jacobian({"x": arange(3), "y": arange(3)}, jac)
        cache.cache_outputs({"x": arange(3) * 2, "y": arange(3)}, {"f": array([2])})
        all_data = iter(cache)
        data = next(all_data)
        assert_items_equal(data.inputs, {"x": arange(3), "y": arange(3)})
        assert_items_equal(data.outputs, {"f": array([1])})
        data = next(all_data)
        assert_items_equal(data.inputs, {"x": arange(3) * 2, "y": arange(3)})
        assert_items_equal(data.outputs, {"f": array([2])})


def assert_items_equal(data1, data2):
    assert len(data1) == len(data2)
    for key, val in data1.items():
        assert key in data2
        assert (val == data2[key]).all()


def test_addition():
    cache1 = CacheFactory().create("MemoryFullCache")
    cache1.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})

    cache2 = CacheFactory().create("MemoryFullCache")
    cache2.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})

    cache3 = cache1 + cache2
    assert len(cache3) == 1

    cache2.cache_outputs({"x": arange(3) * 2, "y": arange(3)}, {"f": array([1]) * 2})
    cache3 = cache1 + cache2
    assert len(cache3) == 2
    all_data = iter(cache3)
    data = next(all_data)
    assert_items_equal(data.inputs, {"x": arange(3), "y": arange(3)})
    assert_items_equal(data.outputs, {"f": array([1])})
    data = next(all_data)
    assert_items_equal(data.inputs, {"x": arange(3) * 2, "y": arange(3)})
    assert_items_equal(data.outputs, {"f": array([2])})


def test_duplicate_from_scratch(memory_full_cache, hdf5_cache):
    hdf5_cache._copy_empty_cache()
    memory_full_cache._copy_empty_cache()


def test_multithreading(memory_full_cache, memory_full_cache_loc):
    caches = (memory_full_cache, memory_full_cache_loc)
    for c_1, c_2 in zip(caches, caches):
        s_s = SellarSystem()
        s_1 = Sellar1()
        s_1.cache = c_1
        s_s.cache = c_2
        assert len(c_1) == 0
        assert len(c_2) == 0
        par = MDOParallelChain([s_1, s_s])
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
    memory_full_cache.cache_outputs(input_data, data_out)
    input_data = {"i": arange(4)}
    data_out = {"o": arange(5)}
    memory_full_cache.cache_outputs(input_data, data_out)
    copy = memory_full_cache.copy
    assert len(copy) == 2
    copy.cache_outputs({"i": arange(5)}, {"o": arange(6)})
    assert len(copy) == 3
    assert len(memory_full_cache) == 2


def test_hdf5singleton(tmp_wd):
    node = "node"
    file_path = "singleton.h5"
    CacheFactory().create("HDF5Cache", hdf_file_path=file_path, hdf_node_path=node)
    singleton = HDF5FileSingleton(file_path)
    data = {"x": array([0.0])}
    singleton.write_data(data, HDF5Cache._INPUTS_GROUP, 1, node)
    with pytest.raises(RuntimeError):
        singleton.write_data(
            data,
            HDF5Cache._INPUTS_GROUP,
            1,
            node,
        )


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
def h5_file(tmp_wd) -> Iterator[h5py.File]:
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
            "The file cache.h5 cannot be used because its file format version is {} "
            "while the expected version is {}: "
            "see HDFCache.update_file_format to convert it.".format(
                HDF5FileSingleton.FILE_FORMAT_VERSION + 1,
                HDF5FileSingleton.FILE_FORMAT_VERSION,
            )
        ),
    ):
        HDF5FileSingleton(CACHE_FILE_NAME)


def test_update_file_format(tmp_wd):
    """Check that updating format changes both hashes and version tag."""
    singleton = HDF5FileSingleton(CACHE_FILE_NAME)
    singleton.write_data(
        {"x": array([1.0]), "y": array([2.0, 3.0])},
        HDF5Cache._INPUTS_GROUP,
        1,
        "foo",
    )
    singleton.write_data(
        {"x": array([2.0]), "y": array([3.0, 4.0])},
        HDF5Cache._INPUTS_GROUP,
        2,
        "foo",
    )
    with h5py.File(CACHE_FILE_NAME, mode="r+") as h5_file:
        old_hash_1 = array([1], dtype="bytes")
        old_hash_2 = array([2], dtype="bytes")
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

    shutil.copy(str(DIR_PATH / deprecated_cache_path), deprecated_cache_path)
    HDF5Cache.update_file_format(deprecated_cache_path)

    cache_path = Path("cache.h5")
    cache = HDF5Cache(cache_path)
    cache.cache_outputs({"x": array([1.0])}, {"y": array([2.0])})

    file_format_version = HDF5FileSingleton.FILE_FORMAT_VERSION
    hash_tag = HDF5FileSingleton.HASH_TAG
    inputs_group = HDF5Cache._INPUTS_GROUP
    outputs_group = HDF5Cache._OUTPUTS_GROUP
    with h5py.File(str(cache_path), mode="a") as h5_file:
        with h5py.File(str(deprecated_cache_path), mode="a") as deprecated_h5_file:
            assert h5_file.attrs["version"] == file_format_version
            assert deprecated_h5_file.attrs["version"] == file_format_version
            new_h5 = h5_file["node"]["1"]
            old_h5 = h5_file["node"]["1"]
            assert new_h5[hash_tag][0] == old_h5[hash_tag][0]
            assert new_h5[inputs_group]["x"][0] == old_h5[inputs_group]["x"][0]
            assert new_h5[outputs_group]["y"][0] == old_h5[outputs_group]["y"][0]


def test_iter(memory_full_cache):
    """Check that a cache can be iterated with namedtuple values."""
    memory_full_cache.cache_outputs({"x": array([0])}, {"y": array([0])})
    memory_full_cache.cache_outputs({"x": array([1])}, {"y": array([1])})
    for index, data in enumerate(memory_full_cache):
        assert data[0]["x"] == array([index])
        assert data[1]["y"] == array([index])
        assert data[2] is None
        assert data.inputs["x"] == array([index])
        assert data.outputs["y"] == array([index])
        assert data.jacobian is None


@pytest.mark.parametrize(
    "jacobian_data",
    [None, {"y": {"x": array([[1]])}}],
)
def test_setitem_getitem(simple_cache, jacobian_data):
    """Check that a cache can be read and set with __getitem__ and __setitem__."""
    simple_cache[{"x": array([0])}] = ({"y": array([1])}, jacobian_data)
    assert simple_cache.last_entry.inputs["x"] == array([0])
    assert simple_cache.last_entry.outputs["y"] == array([1])

    if jacobian_data:
        assert simple_cache.last_entry.jacobian == {"y": {"x": array([[1]])}}
    else:
        assert not simple_cache.last_entry.jacobian


def test_getitem_missing_entry(simple_cache):
    """Verify that querying a missing entry returns an empty CacheEntry."""
    data = simple_cache[{"x": array([2])}]
    assert data.inputs["x"] == array([2])
    assert not data.outputs
    assert not data.jacobian


def test_setitem_empty_data(simple_cache, caplog):
    """Verify that adding an empty data to the cache logs a warning."""
    with caplog.at_level(logging.WARNING):
        simple_cache[{"x": array([1.0])}] = (None, None)
        assert (
            "Cannot add the entry to the cache "
            "as both output data and Jacobian data are missing."
        ) in caplog.text


@pytest.mark.parametrize("first_jacobian", [None, {"y": {"x": array([[4.0]])}}])
@pytest.mark.parametrize("second_jacobian", [None, {"y": {"x": array([[4.0]])}}])
@pytest.mark.parametrize("first_outputs", [None, {"y": array([3.0])}])
def test_export_to_dataset_and_entries(
    simple_cache, first_jacobian, second_jacobian, first_outputs
):
    """Check exporting a simple cache to a dataset and entries.

    Only the last observation should be exported, without its Jacobian.
    """
    first_inputs = {"x": array([2.0])}
    second_inputs = {"x": array([1.0])}
    second_outputs = {"y": array([2.0])}
    simple_cache[first_inputs] = (first_outputs, first_jacobian)
    simple_cache[second_inputs] = (second_outputs, second_jacobian)
    dataset = simple_cache.export_to_dataset()
    assert len(dataset) == 1
    assert dataset["x"][0, 0] == 1.0
    assert dataset["y"][0, 0] == 2.0

    # Check penultimate_entry and last_entry
    first_jacobian = first_jacobian or {}
    second_jacobian = second_jacobian or {}
    if second_outputs:
        first_outputs = {}
    if second_jacobian:
        first_jacobian = {}
    if not first_jacobian and not first_outputs:
        first_inputs = {}
    penultimate_entry = CacheEntry(
        first_inputs,
        first_outputs,
        first_jacobian,
    )
    last_entry = CacheEntry(second_inputs, second_outputs, second_jacobian or {})
    assert simple_cache.penultimate_entry == penultimate_entry
    assert simple_cache.last_entry == last_entry

    # Check __iter__
    entries = [entry for entry in simple_cache]
    if first_inputs:
        assert len(entries) == 2
        assert entries[0] == penultimate_entry
        assert entries[1] == last_entry
    else:
        assert len(entries) == 1
        assert entries[0] == last_entry


@pytest.mark.parametrize(
    "data",
    (
        arange(2),
        [0, 0],
        {
            0: None,
            1: None,
        },
    ),
)
def test_names_to_sizes(simple_cache, data):
    """Verify the ``names_to_sizes`` attribute."""
    simple_cache.cache_outputs({"index": 1}, {"o": data})
    assert simple_cache.names_to_sizes == {"index": 1, "o": 2}
