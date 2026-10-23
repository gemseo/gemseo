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
import shutil
from errno import EACCES
from itertools import count
from pathlib import Path
from pathlib import PureWindowsPath
from typing import TYPE_CHECKING
from typing import ClassVar

import h5py
import pytest
from numpy import arange
from numpy import array
from numpy import eye
from numpy import float64
from numpy import zeros
from scipy.sparse import eye as speye

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.core.cache import _hdf5_file_singleton
from gemseo.core.cache import base_full
from gemseo.core.cache._hdf5_file_singleton import HDF5FileSingleton
from gemseo.core.cache.cache_entry import CacheEntry
from gemseo.core.cache.cache_item import CacheItem
from gemseo.core.cache.factory import CacheFactory
from gemseo.core.cache.hdf5 import HDF5Cache
from gemseo.core.cache.simple import SimpleCache
from gemseo.core.cache.util import hash_data
from gemseo.core.cache.util import to_real
from gemseo.dataset.io_dataset import IODataset
from gemseo.discipline.chain.parallel_chain import ParallelDisciplineChain
from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.problem.mdo.sellar.sellar_design_space import SellarDesignSpace
from gemseo.util._numpy import int32_dtype
from gemseo.util._numpy import int64_dtype
from gemseo.util._numpy import uint32_dtype
from gemseo.util._numpy import uint64_dtype
from gemseo.util.comparison import compare_dict_of_arrays
from gemseo.util.singleton import SingleInstancePerFileAttribute
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy import ndarray

    from gemseo.core.cache.memory_full import MemoryFullCache
    from gemseo.util.typing import StrKeyMapping

dir_path = Path(__file__).parent

factory = CacheFactory()


@pytest.fixture
def simple_cache():
    return factory.create("SimpleCache")


@pytest.fixture
def memory_full_cache():
    return factory.create("MemoryFullCache")


@pytest.fixture
def memory_full_cache_loc():
    return factory.create("MemoryFullCache", is_memory_shared=False)


@pytest.fixture
def hdf5_cache(tmp_wd):
    return factory.create(
        "HDF5Cache", hdf_file_path="dummy.h5", hdf_node_path="DummyCache"
    )


def test_io_names(simple_cache):
    """Verify input_names and output_names properties."""
    # NO IO names at instantiation.
    assert simple_cache.input_names == []
    assert simple_cache.output_names == []

    # After a first entry.
    simple_cache[{"b": 0, "a": 1}] = ({"d": 2, "c": 3}, {})
    assert simple_cache.input_names == ["a", "b"]
    assert simple_cache.output_names == ["c", "d"]

    # After a new entry: the names are the same as those in the first entry.
    simple_cache[{"b_": 0, "a_": 1}] = ({"d_": 2, "c_": 3}, {})
    assert simple_cache.input_names == ["a", "b"]
    assert simple_cache.output_names == ["c", "d"]


@pytest.mark.parametrize("cache", map(factory.create, factory.class_names))
def test_tolerance(cache, snapshot):
    """Verify tolerance property."""
    assert cache.tolerance == 0.0
    cache.tolerance = 1.0
    assert cache.tolerance == 1.0
    with assert_exception(ValueError, snapshot):
        cache.tolerance = -1.0


def test_jac_and_outputs_caching(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
) -> None:
    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        # Test the cache are empty
        assert not cache
        assert not cache.last_item.outputs

        # Create input, output and jacobian
        input_data = {"i1": arange(3), "i2": arange(3)}
        output_data = {"o": arange(4)}
        jac = {"o": {"i1": eye(4, 3), "i2": speye(4, 3)}}

        # Cache and retrieve output
        cache.cache_outputs(input_data, output_data)
        out_loaded, jac_loaded = cache[input_data]
        assert out_loaded
        assert not jac_loaded
        assert_items_equal(out_loaded, output_data)

        # Cache and retrieve jacobian
        cache.cache_jacobian(input_data, jac)
        out_loaded, jac_loaded = cache[input_data]
        assert out_loaded is not None
        assert compare_dict_of_arrays(jac, jac_loaded)

        cache.tolerance = 1e-6
        cache[input_data]
        assert out_loaded is not None
        assert compare_dict_of_arrays(jac, jac_loaded)

        cache.tolerance = 0.0

        # Cache jac again, to check it still works
        cache.cache_jacobian(input_data, jac)

    assert len(simple_cache) == 1

    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        last_item = cache.last_item
        assert last_item.inputs
        assert last_item.outputs

        cache.clear()
        assert not cache


def test_hdf_cache_read(tmp_wd) -> None:
    cache = HDF5Cache(hdf_file_path="dummy.h5", hdf_node_path="DummyCache")

    with pytest.raises(ValueError):
        cache.to_ggobi(
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
    cache_read = HDF5Cache(hdf_file_path="dummy.h5", hdf_node_path="DummyCache")

    assert len(cache_read) == n

    cache.to_ggobi(
        "out1.ggobi",
        input_names=["i", "j"],
        output_names=["o"],
    )

    input_data = {"i": zeros(3), "k": zeros(3)}
    output_data = {"o": arange(4), "t": zeros(3)}
    cache.cache_outputs(input_data, output_data)

    assert len(cache) == n + 1
    exp_ggobi = Path("out2.ggobi")
    cache.to_ggobi(str(exp_ggobi), input_names=["i", "k"], output_names=["o", "t"])
    assert exp_ggobi.exists()

    with pytest.raises(ValueError):
        cache.to_ggobi(
            "out3.ggobi",
            input_names=["izgz"],
            output_names=["o"],
        )


def test_update_caches(tmp_wd) -> None:
    c1 = HDF5Cache(hdf_file_path="out11.h5", hdf_node_path="DummyCache")
    c2 = HDF5Cache(hdf_file_path="out21.h5", hdf_node_path="DummyCache")

    c1.cache_outputs({"i": 10 * arange(3)}, {"o": arange(4)})

    input_data2 = {"i": 2 * arange(3)}
    c2.cache_outputs(input_data2, {"o": 2 * arange(4)})
    c2.cache_jacobian(input_data2, {"o": {"i": 3 * arange(4)}})
    # This empty entry will not be used to update c1.
    c2.cache_outputs({"i": 3 * arange(3)}, {})

    c1.update(c2)
    assert len(c1) == 2

    outs, jacs = c1[input_data2]
    assert (outs["o"] == 2 * arange(4)).all()
    assert (jacs["o"]["i"] == 3 * arange(4)).all()


def test_collision(tmp_wd) -> None:
    c1 = HDF5Cache(hdf_file_path="out7.h5", hdf_node_path="DummyCache")
    input_data = {"i": arange(3)}
    output_data = {"o": arange(3)}
    c1.cache_outputs(input_data, output_data)

    # Fake a collision of hashes
    c1.cache_outputs(input_data, output_data)
    hash_0 = next(iter(c1._hash_to_indices.keys()))
    groups = c1._hash_to_indices.pop(hash_0)
    input_data2 = {"i": 2 * arange(3)}
    hash_1 = hash_data(input_data2)
    c1._hash_to_indices[hash_1] = groups
    output_data2 = {"o": 2.0 * arange(3)}
    c1.cache_outputs(input_data2, output_data2)

    c1_outs, _ = c1[input_data2]
    assert (c1_outs["o"] == output_data2["o"]).all()


@pytest.mark.parametrize(
    ("other_value", "are_equal"),
    [
        (Path("a") / "b", True),
        ("a/b", True),
        (Path("a") / "c", False),
        (PureWindowsPath("a\\b"), True),
    ],
    ids=["same-path", "str", "other-path", "windows-path"],
)
def test_hash_data_path(other_value, are_equal) -> None:
    """Check that a path is hashed as its string with forward slashes."""
    assert (
        hash_data({"p": Path("a") / "b"}) == hash_data({"p": other_value})
    ) is are_equal


@pytest.mark.parametrize(
    "value",
    [[Path("a")], (Path("a"),), array([Path("a")], dtype=object)],
    ids=["list", "tuple", "object-array"],
)
def test_hash_data_several_paths(value, snapshot) -> None:
    """Check that a list, a tuple or an array of paths cannot be hashed."""
    with assert_exception(TypeError, snapshot):
        hash_data({"p": value})


@pytest.mark.parametrize(
    "value", [{"k": 1}, array([object()])], ids=["dict", "object-array"]
)
def test_hash_data_object_array(value, snapshot) -> None:
    """Check that an array of Python objects, not paths, cannot be hashed."""
    with assert_exception(TypeError, snapshot):
        hash_data({"p": value})


def test_hash_data_dict() -> None:
    input_data = {"i": 10 * arange(3)}
    hash_0 = hash_data(input_data)
    assert isinstance(hash_0, int)
    assert hash_0 == hash_data(input_data)
    assert hash_0 == hash_data({"i": 10 * arange(3), "t": None})
    assert hash_0 == hash_data({"i": 10 * arange(3), "j": None})
    hash_data({"i": 10 * arange(3)})
    # Discontiguous array
    hash_data({"i": arange(10)[::3]})
    # Zero-dimensional array
    assert isinstance(hash_data({"i": array(1.0)}), int)


@pytest.mark.parametrize(
    ("dtype_32", "dtype_64"), [(int32_dtype, int64_dtype), (uint32_dtype, uint64_dtype)]
)
def test_hash_data_32_bit_integers(dtype_32, dtype_64) -> None:
    """Check that a 32-bit integer array is hashed as its 64-bit equivalent."""
    assert hash_data({"i": array([1, 2], dtype=dtype_32)}) == hash_data({
        "i": array([1, 2], dtype=dtype_64)
    })


@pytest.mark.parametrize(
    ("array_c", "array_non_c"),
    [
        (array([[1, 2], [3, 4]], order="C"), array([[1, 2], [3, 4]], order="F")),
        (
            array([[1.0, 2.0], [3.0, 4.0]], order="C"),
            array([[1.0, 2.0], [3.0, 4.0]], order="F"),
        ),
        (array([0, 3, 6, 9]), arange(10)[::3]),
        (array([0.0, 3.0, 6.0, 9.0]), arange(10.0)[::3]),
    ],
)
def test_hash_discontiguous_array(array_c, array_non_c) -> None:
    """Test that the hashes are the same for discontiguous arrays.

    Args:
        array_c: A C-contiguous array.
        array_non_c: An array that is not C-contiguous.
    """
    assert hash_data({"i": array_c}) == hash_data({"i": array_non_c})


def func(x):
    """Dummy function to test the cache."""
    y = x
    return y  # noqa: RET504


@pytest.mark.parametrize(
    ("hdf_name", "inputs", "expected"),
    [
        ("int_win.h5", array([1, 2, 3]), array([1, 2, 3])),
        ("int_linux.h5", array([1, 2, 3]), array([1, 2, 3])),
        ("float_win.h5", array([1.0, 2.0, 3.0]), array([1.0, 2.0, 3.0])),
        ("float_linux.h5", array([1.0, 2.0, 3.0]), array([1.0, 2.0, 3.0])),
        # The hashes of the files above are computed again when they are opened,
        # as their version is older; these ones are read as they are stored,
        # and hashed the same way on every platform.
        ("int_v3.h5", array([1, 2, 3]), array([1, 2, 3])),
        ("float_v3.h5", array([1.0, 2.0, 3.0]), array([1.0, 2.0, 3.0])),
    ],
)
def test_det_hash(
    tmp_wd, hdf_name, inputs, expected, enable_discipline_statistics
) -> None:
    """Test that hashed values are the same across sessions.

    Args:
        tmp_wd: Fixture to move into a temporary directory.
    """
    # Use a temporary copy of the file in case the test fails.
    shutil.copy(str(dir_path / hdf_name), tmp_wd)
    disc = create_discipline("AutoPyDiscipline", py_func=func)
    disc.set_cache("HDF5Cache", hdf_file_path=hdf_name)
    out = disc.execute({"x": inputs})

    assert disc.execution_statistics.n_executions == 0
    assert out["y"].all() == expected.all()


def test_to_real() -> None:
    assert to_real(1.0j * arange(3)).dtype == float64
    assert to_real(1.0 * arange(3)).dtype == float64
    assert (to_real(1.0j * arange(3) + arange(3)) == 1.0 * arange(3)).all()


def test_write_data(tmp_wd) -> None:
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


def test_read_hashes(tmp_wd) -> None:
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


def test_read_group(tmp_wd) -> None:
    """Check that a group is correctly read."""
    cache = HDF5Cache(hdf_file_path="out3.h5")
    input_data = {"x": arange(3), "y": arange(3)}
    output_data = {"f": array([1])}
    cache.cache_outputs(input_data, output_data)
    cache_entry = cache[{"x": arange(3), "y": arange(3)}]
    assert_items_equal(cache_entry.outputs, output_data)


def test_get_all_data(
    simple_cache, memory_full_cache, memory_full_cache_loc, hdf5_cache
) -> None:
    inputs = {"x": arange(3), "y": arange(3)}
    outputs = {"f": array([1])}
    for cache in [simple_cache, hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        cache.cache_outputs(inputs, outputs)
        all_data = list(cache.items())
        assert len(all_data) == 1
        input_data_0, cache_entry_0 = all_data[0]
        assert_items_equal(input_data_0, inputs)
        assert_items_equal(cache_entry_0.outputs, outputs)
        assert not cache_entry_0.jacobian
        assert cache[inputs].outputs["f"] == outputs["f"]

    jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
    hdf5_cache.cache_jacobian({"x": arange(3), "y": arange(3)}, jac)


def test_all_data(memory_full_cache, memory_full_cache_loc, hdf5_cache) -> None:
    for cache in [hdf5_cache, memory_full_cache, memory_full_cache_loc]:
        jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
        cache.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})
        cache.cache_jacobian({"x": arange(3), "y": arange(3)}, jac)
        cache.cache_outputs({"x": arange(3) * 2, "y": arange(3)}, {"f": array([2])})
        all_data = iter(cache.items())
        input_data_0, entry_0 = next(all_data)
        assert_items_equal(input_data_0, {"x": arange(3), "y": arange(3)})
        assert_items_equal(entry_0.outputs, {"f": array([1])})
        input_data_1, entry_1 = next(all_data)
        assert_items_equal(input_data_1, {"x": arange(3) * 2, "y": arange(3)})
        assert_items_equal(entry_1.outputs, {"f": array([2])})


def assert_items_equal(data1, data2) -> None:
    assert len(data1) == len(data2)
    for key, val in data1.items():
        assert key in data2
        assert (val == data2[key]).all()


def test_addition() -> None:
    cache1 = CacheFactory().create("MemoryFullCache")
    cache1.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})

    cache2 = CacheFactory().create("MemoryFullCache")
    cache2.cache_outputs({"x": arange(3), "y": arange(3)}, {"f": array([1])})

    cache3 = cache1 + cache2
    assert len(cache3) == 1

    cache2.cache_outputs({"x": arange(3) * 2, "y": arange(3)}, {"f": array([1]) * 2})
    cache3 = cache1 + cache2
    assert len(cache3) == 2
    all_data = iter(cache3.items())
    input_data_0, entry_0 = next(all_data)
    assert_items_equal(input_data_0, {"x": arange(3), "y": arange(3)})
    assert_items_equal(entry_0.outputs, {"f": array([1])})
    input_data_1, entry_1 = next(all_data)
    assert_items_equal(input_data_1, {"x": arange(3) * 2, "y": arange(3)})
    assert_items_equal(entry_1.outputs, {"f": array([2])})


def test_duplicate_from_scratch(memory_full_cache, hdf5_cache) -> None:
    hdf5_cache._copy_empty_cache()
    memory_full_cache._copy_empty_cache()


@pytest.fixture(params=range(2))
def memory_cache(memory_full_cache, memory_full_cache_loc, request) -> MemoryFullCache:
    """A parametrized fixture to iterate over memory_full_cache and
    memory_full_cache_loc."""
    return (memory_full_cache, memory_full_cache_loc)[request.param]


def test_multithreading(
    memory_cache, sellar_with_2d_array, sellar_disciplines, enable_function_statistics
) -> None:
    s_1 = sellar_disciplines.sellar1
    s_s = sellar_disciplines.sellar_system
    s_1.cache = memory_cache
    s_s.cache = memory_cache
    assert len(memory_cache) == 0
    par = ParallelDisciplineChain([s_1, s_s])
    ds = SellarDesignSpace()
    scen = create_scenario(par, "obj", ds, formulation_name="DisciplinaryOpt")

    settings = PYDOE_FULLFACT_Settings(n_samples=10, n_processes=4)
    scen.execute(settings)

    nexec_1 = s_1.execution_statistics.n_executions
    nexec_2 = s_s.execution_statistics.n_executions

    scen = create_scenario(par, "obj", ds, formulation_name="DisciplinaryOpt")
    scen.execute(settings)

    assert nexec_1 == s_1.execution_statistics.n_executions
    assert nexec_2 == s_s.execution_statistics.n_executions


def test_copy(memory_full_cache) -> None:
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


def test_hdf5singleton(tmp_wd) -> None:
    node = "node"
    file_path = "singleton.h5"
    CacheFactory().create("HDF5Cache", hdf_file_path=file_path, hdf_node_path=node)
    singleton = HDF5FileSingleton(file_path)
    data = {"x": array([0.0])}
    singleton.write_data(data, HDF5Cache.Group.INPUTS, 1, node)
    with pytest.raises(RuntimeError):
        singleton.write_data(
            data,
            HDF5Cache.Group.INPUTS,
            1,
            node,
        )


def test_hash_data_dict_keys() -> None:
    """Check that hash considers the keys of the dictionary."""
    data = {"a": array([1]), "b": array([2])}
    assert hash_data(data) == hash_data({"a": array([1]), "b": array([2])})
    assert hash_data(data) != hash_data({"a": array([1]), "c": array([2])})
    assert hash_data(data) != hash_data({"a": array([1]), "b": array([3])})

    data = {"a": array([1]), "b": array([1])}
    assert hash_data(data) != hash_data({"a": array([1]), "c": array([1])})


cache_file_name = "cache.h5"


@pytest.fixture
def h5_file(tmp_wd) -> Iterator[h5py.File]:
    """Provide an empty h5 file object and close it afterward."""
    h5_file = h5py.File(cache_file_name, mode="a")
    yield h5_file
    h5_file.close()


def test_check_version_new_file() -> None:
    """Verify that a non-existing file passes the file format version check."""
    HDF5FileSingleton("foo")


def test_check_version_empty_file(h5_file) -> None:
    """Verify that an empty file passes the file format version check."""
    HDF5FileSingleton(cache_file_name)


def test_check_version_missing(h5_file, snapshot) -> None:
    """Verify that a non-empty file with no file format version raises."""
    h5_file["foo"] = "bar"

    with assert_exception(ValueError, snapshot):
        HDF5FileSingleton(cache_file_name)


def test_check_version_greater(h5_file, snapshot, monkeypatch) -> None:
    """Verify that a non-empty file with greater file format version raises."""
    h5_file["foo"] = "bar"
    h5_file.attrs["version"] = HDF5FileSingleton.file_format_version + 1
    # The message names the running version of GEMSEO,
    # which changes with every commit of a development version.
    monkeypatch.setattr(_hdf5_file_singleton.metadata, "version", lambda _: "7.0.0")

    with assert_exception(ValueError, snapshot):
        HDF5FileSingleton(cache_file_name)


def _find_input_data_with_long_hash() -> dict[str, ndarray]:
    """Return input data whose hash has 20 digits, the most for a 64-bit hash.

    Returns:
        The input data.
    """
    for index in count():
        input_data = {"x": array([float(index)])}
        if len(str(hash_data(input_data))) == 20:
            return input_data

    return {}  # pragma: no cover


def _write_older_file(
    file_path: str, hdf_node_path: str, input_data: StrKeyMapping
) -> None:
    """Write an entry in an HDF5 file looking like one of an older version.

    Its stored hash has 19 digits, as if computed by another hash function,
    in a dataset of type `S19` that cannot hold the 20 digits of its current hash.

    Args:
        file_path: The path to the HDF5 file.
        hdf_node_path: The path to the node of the entry.
        input_data: The input data of the entry.
    """
    cache = HDF5Cache(hdf_file_path=file_path, hdf_node_path=hdf_node_path)
    cache.cache_outputs(input_data, {"y": array([2.0])})
    with h5py.File(file_path, "a") as h5_file:
        h5_file.attrs["version"] = HDF5FileSingleton.file_format_version - 1
        entry = h5_file[hdf_node_path]["1"]
        del entry[HDF5FileSingleton.hash_tag]
        entry.create_dataset(HDF5FileSingleton.hash_tag, data=array([b"1" * 19]))


def _forget_file_singletons(monkeypatch) -> None:
    """Make the next caches use new file singletons, for the current test only.

    The process-wide registry of the singletons is replaced, not emptied,
    so that the other tests of the same process keep their singletons.

    Args:
        monkeypatch: The pytest fixture restoring the registry.
    """
    monkeypatch.setattr(SingleInstancePerFileAttribute, "instances", {})


def test_check_version_older(tmp_wd, monkeypatch) -> None:
    """Verify that the hashes of a file of an older version are computed again."""
    input_data = _find_input_data_with_long_hash()
    _write_older_file(cache_file_name, "node", input_data)

    _forget_file_singletons(monkeypatch)
    new_cache = HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path="node")

    # The entry is found again although its stored hash was stale,
    # and its new hash is stored in full, although longer than the old one.
    assert len(new_cache) == 1
    assert compare_dict_of_arrays(new_cache[input_data].outputs, {"y": array([2.0])})
    with h5py.File(cache_file_name, "r") as h5_file:
        stored_hash = h5_file["node"]["1"][HDF5FileSingleton.hash_tag][0]
    assert int(stored_hash) == hash_data(input_data)


def test_check_version_older_with_nested_node(tmp_wd, monkeypatch) -> None:
    """Verify that the hashes of the entries at a nested node are computed again."""
    input_data = _find_input_data_with_long_hash()
    _write_older_file(cache_file_name, "flat", input_data)
    _write_older_file(cache_file_name, "root/sub/node", input_data)

    _forget_file_singletons(monkeypatch)
    caches = [
        HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path=hdf_node_path)
        for hdf_node_path in ("flat", "root/sub/node")
    ]

    for cache in caches:
        assert compare_dict_of_arrays(cache[input_data].outputs, {"y": array([2.0])})
    with h5py.File(cache_file_name, "r") as h5_file:
        assert h5_file.attrs["version"] == HDF5FileSingleton.file_format_version


def test_check_version_older_read_only(tmp_wd, caplog, monkeypatch) -> None:
    """Verify that a file of an older version that cannot be written can be used."""
    input_data = _find_input_data_with_long_hash()
    _write_older_file(cache_file_name, "node", input_data)

    # The file is made read-only for h5py rather than for the operating system,
    # whose permissions do not apply to root, the user running the CI jobs.
    open_file = h5py.File

    def open_read_only(name, mode="r", *args, **kwargs):
        if mode != "r":
            raise PermissionError(EACCES, "Permission denied", name)
        return open_file(name, mode, *args, **kwargs)

    monkeypatch.setattr(h5py, "File", open_read_only)
    _forget_file_singletons(monkeypatch)
    cache = HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path="node")

    # The hashes are computed from the input data, without writing them.
    assert compare_dict_of_arrays(cache[input_data].outputs, {"y": array([2.0])})
    with h5py.File(cache_file_name, "r") as h5_file:
        version = h5_file.attrs["version"]
    assert version == HDF5FileSingleton.file_format_version - 1
    assert "cannot be updated" in caplog.text


def test_check_version_older_with_missing_hash(tmp_wd, monkeypatch) -> None:
    """Verify that the update repairs an entry whose hash is missing.

    An update interrupted between the deletion of a hash and its creation
    leaves such an entry.
    """
    cache = HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path="node")
    first_input_data = {"x": array([1.0])}
    second_input_data = {"x": array([2.0])}
    cache.cache_outputs(first_input_data, {"y": array([1.0])})
    cache.cache_outputs(second_input_data, {"y": array([2.0])})
    with h5py.File(cache_file_name, "a") as h5_file:
        h5_file.attrs["version"] = HDF5FileSingleton.file_format_version - 1
        del h5_file["node"]["2"][HDF5FileSingleton.hash_tag]

    _forget_file_singletons(monkeypatch)
    new_cache = HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path="node")

    assert len(new_cache) == 2
    for input_data, output_value in [(first_input_data, 1.0), (second_input_data, 2.0)]:
        cache_entry = new_cache[input_data]
        assert compare_dict_of_arrays(cache_entry.outputs, {"y": array([output_value])})
    with h5py.File(cache_file_name, "r") as h5_file:
        stored_hash = h5_file["node"]["2"][HDF5FileSingleton.hash_tag][0]
    assert int(stored_hash) == hash_data(second_input_data)


def test_check_version_older_with_node_named_inputs(tmp_wd, monkeypatch) -> None:
    """Verify that a node named like the input data of an entry is not an entry."""
    input_data = _find_input_data_with_long_hash()
    _write_older_file(cache_file_name, "root/inputs", input_data)

    _forget_file_singletons(monkeypatch)
    cache = HDF5Cache(hdf_file_path=cache_file_name, hdf_node_path="root/inputs")

    assert compare_dict_of_arrays(cache[input_data].outputs, {"y": array([2.0])})
    with h5py.File(cache_file_name, "r") as h5_file:
        assert HDF5FileSingleton.hash_tag not in h5_file["root"]


def test_update_file_format_failure_keeps_the_version(tmp_wd, monkeypatch) -> None:
    """Verify that a file whose update fails keeps its old version."""
    _write_older_file(cache_file_name, "node", {"x": array([1.0])})

    def fail(data):
        msg = "Something went wrong."
        raise ValueError(msg)

    monkeypatch.setattr("gemseo.core.cache._hdf5_file_singleton.hash_data", fail)

    with pytest.raises(ValueError):  # noqa: PT011
        HDF5FileSingleton.update_file_format(cache_file_name)

    with h5py.File(cache_file_name, "r") as h5_file:
        version = h5_file.attrs["version"]
    assert version == HDF5FileSingleton.file_format_version - 1


def test_update_file_format(tmp_wd) -> None:
    """Check that updating format changes both hashes and version tag."""
    singleton = HDF5FileSingleton(cache_file_name)
    singleton.write_data(
        {"x": array([1.0]), "y": array([2.0, 3.0])},
        HDF5Cache.Group.INPUTS,
        1,
        "foo",
    )
    singleton.write_data(
        {"x": array([2.0]), "y": array([3.0, 4.0])},
        HDF5Cache.Group.INPUTS,
        2,
        "foo",
    )
    with h5py.File(cache_file_name, mode="r+") as h5_file:
        old_hash_1 = array([1], dtype="bytes")
        old_hash_2 = array([2], dtype="bytes")
        h5_file["foo"]["1"][HDF5FileSingleton.hash_tag][0] = old_hash_1
        h5_file["foo"]["2"][HDF5FileSingleton.hash_tag][0] = old_hash_2

    HDF5Cache.update_file_format(cache_file_name)
    with h5py.File(cache_file_name, mode="a") as h5_file:
        assert h5_file.attrs["version"] == HDF5FileSingleton.file_format_version
        assert h5_file["foo"]["1"][HDF5FileSingleton.hash_tag][0] != old_hash_1
        assert h5_file["foo"]["2"][HDF5FileSingleton.hash_tag][0] != old_hash_2


def test_update_file_format_from_deprecated_file(tmp_wd) -> None:
    """Check that a file with a deprecated format can be correctly updated."""
    deprecated_cache_path = "cache_with_deprecated_format.h5"
    # This file has been obtained with GEMSEO 3.1.0:
    #     cache = HDF5Cache("cache_with_deprecated_format.h5", "node")
    #     cache.cache_outputs({'x': array([1.])}, ['x'], {'y': array([2.])}, ['y'])

    shutil.copy(str(dir_path / deprecated_cache_path), deprecated_cache_path)
    HDF5Cache.update_file_format(deprecated_cache_path)

    cache_path = Path("cache.h5")
    cache = HDF5Cache(hdf_file_path=cache_path)
    cache.cache_outputs({"x": array([1.0])}, {"y": array([2.0])})

    file_format_version = HDF5FileSingleton.file_format_version
    hash_tag = HDF5FileSingleton.hash_tag
    inputs_group = HDF5Cache.Group.INPUTS
    outputs_group = HDF5Cache.Group.OUTPUTS
    with h5py.File(str(cache_path), mode="a") as h5_file:  # noqa: SIM117
        with h5py.File(str(deprecated_cache_path), mode="a") as deprecated_h5_file:
            assert h5_file.attrs["version"] == file_format_version
            assert deprecated_h5_file.attrs["version"] == file_format_version
            new_h5 = h5_file["node"]["1"]
            old_h5 = h5_file["node"]["1"]
            assert new_h5[hash_tag][0] == old_h5[hash_tag][0]
            assert new_h5[inputs_group]["x"][0] == old_h5[inputs_group]["x"][0]
            assert new_h5[outputs_group]["y"][0] == old_h5[outputs_group]["y"][0]


def test_get_entries(memory_full_cache) -> None:
    """Check that a cache can be iterated with namedtuple values."""
    memory_full_cache.cache_outputs({"x": array([0])}, {"y": array([0])})
    memory_full_cache.cache_outputs({"x": array([1])}, {"y": array([1])})
    for index, (input_data, entry) in enumerate(memory_full_cache.items()):
        assert input_data["x"] == array([index])
        assert entry.outputs["y"] == array([index])
        assert not entry.jacobian


@pytest.mark.parametrize(
    "jacobian_data",
    [None, {"y": {"x": array([[1]])}}],
)
def test_setitem_getitem(simple_cache, jacobian_data) -> None:
    """Check that a cache can be read and set with __getitem__ and __setitem__."""
    simple_cache[{"x": array([0])}] = ({"y": array([1])}, jacobian_data)
    assert next(iter(simple_cache))["x"] == array([0])
    assert simple_cache.last_item.outputs["y"] == array([1])

    if jacobian_data:
        assert simple_cache.last_item.jacobian == {"y": {"x": array([[1]])}}
    else:
        assert not simple_cache.last_item.jacobian


def test_getitem_missing_entry(simple_cache) -> None:
    """Verify that querying a missing entry returns an empty CacheEntry."""
    data = simple_cache[{"x": array([2])}]
    assert not data.outputs
    assert not data.jacobian


def test_setitem_empty_data(simple_cache, caplog) -> None:
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
@pytest.mark.parametrize("categorize", [False, True])
def test_export_to_dataset_and_entries(
    simple_cache, first_jacobian, second_jacobian, first_outputs, categorize
) -> None:
    """Check exporting a simple cache to a dataset and entries.

    Only the last observation should be exported, without its Jacobian.
    """
    first_inputs = {"x": array([2.0])}
    second_inputs = {"x": array([1.0])}
    second_outputs = {"y": array([2.0])}
    simple_cache[first_inputs] = (first_outputs, first_jacobian)
    simple_cache[second_inputs] = (second_outputs, second_jacobian)
    dataset = simple_cache.to_dataset(categorize=categorize)
    assert len(dataset) == 1
    if categorize:
        assert isinstance(dataset, IODataset)
        assert "x" in dataset.input_names
        assert "y" in dataset.output_names
    else:
        assert dataset.group_names == [dataset.parameter_group]

    assert dataset.get_view(variable_names="x").to_numpy()[0, 0] == 1.0
    assert dataset.get_view(variable_names="y").to_numpy()[0, 0] == 2.0

    second_jacobian = second_jacobian or {}
    last_item = CacheItem(second_inputs, second_outputs, second_jacobian or {})
    assert simple_cache.last_item == last_item

    # Check get_all_entries
    entries = list(simple_cache.items())
    assert len(entries) == 1
    assert entries[0] == (
        second_inputs,
        CacheEntry(last_item.outputs, last_item.jacobian),
    )


@pytest.mark.parametrize(
    "data",
    [
        arange(2),
        [0, 0],
        {
            0: None,
            1: None,
        },
    ],
)
def test_names_to_sizes(simple_cache, data) -> None:
    """Verify the `name_to_size` attribute."""
    assert not simple_cache.name_to_size
    simple_cache.cache_outputs({"index": 1}, {"o": data})
    assert simple_cache.name_to_size == {"index": 1, "o": 2}


def _compare_dict_of_arrays(*args, **kwargs):
    """Dummy dict comparator that just raises when called."""
    raise RuntimeError


def test_dict_comparator():
    """Verify that a different dict comparator can be injected."""
    cache = SimpleCache()
    cache.compare_dict_of_arrays = _compare_dict_of_arrays
    cache.cache_outputs({"i": arange(1)}, {"o": arange(1)})
    with pytest.raises(RuntimeError):
        cache.get("i")


def _count_comparisons(cache, monkeypatch) -> list[int]:
    """Count the comparisons of input data done by a cache.

    The input data of an entry are read to be compared,
    so the reads are counted rather than the calls to the comparator,
    which the cache would no longer take for the default one if it were replaced.

    Args:
        cache: The cache to spy on.
        monkeypatch: The pytest fixture restoring the method.

    Returns:
        A single-item list containing the number of comparisons.
    """
    n_comparisons = [0]
    read_data = cache._read_data

    def count(index, group):
        if group == cache.Group.INPUTS:
            n_comparisons[0] += 1
        return read_data(index, group)

    monkeypatch.setattr(cache, "_read_data", count)
    return n_comparisons


def _fill(cache, n_entries: int = 10) -> None:
    """Cache the outputs of several input data.

    Args:
        cache: The cache to fill.
        n_entries: The number of entries to cache.
    """
    for index in range(n_entries):
        cache.cache_outputs({"x": array([index])}, {"y": array([index])})


def test_getitem_does_not_compare_the_entries(memory_full_cache, monkeypatch) -> None:
    """Check that a missing entry is not compared to the entries of the cache."""
    _fill(memory_full_cache)
    n_comparisons = _count_comparisons(memory_full_cache, monkeypatch)

    # The hash of the input data is enough to conclude that the entry is missing.
    assert not memory_full_cache[{"x": array([10])}].outputs

    assert n_comparisons[0] == 0


def test_getitem_returns_the_entry_found(memory_full_cache) -> None:
    """Check that a lookup returns the entry of the input data."""
    _fill(memory_full_cache)

    cache_entry = memory_full_cache[{"x": array([5])}]

    assert compare_dict_of_arrays(cache_entry.outputs, {"y": array([5])})


def test_getitem_compares_the_entries_with_a_tolerance(
    memory_full_cache, monkeypatch
) -> None:
    """Check that all the entries are compared when the tolerance is not zero."""
    _fill(memory_full_cache)
    memory_full_cache.tolerance = 1e-6
    n_comparisons = _count_comparisons(memory_full_cache, monkeypatch)

    # Input data close to each other can have different hashes,
    # so all the entries are compared to the missing one.
    assert not memory_full_cache[{"x": array([10.5])}].outputs
    assert n_comparisons[0] == 10


def test_getitem_in_a_child_process(memory_full_cache, monkeypatch) -> None:
    """Check that a lookup from a child process finds the entry of the input data."""
    _fill(memory_full_cache)
    monkeypatch.setattr(base_full, "parent_process", object)

    cache_entry = memory_full_cache[{"x": array([5])}]

    assert compare_dict_of_arrays(cache_entry.outputs, {"y": array([5])})


def _compare_names_and_shapes(dict_of_arrays, other_dict_of_arrays, tolerance=0.0):
    """Compare two dictionaries of arrays from their names and shapes only.

    Args:
        dict_of_arrays: A dictionary of arrays.
        other_dict_of_arrays: Another dictionary of arrays.
        tolerance: The tolerance, ignored.

    Returns:
        Whether the dictionaries have the same names and shapes.
    """
    return {name: value.shape for name, value in dict_of_arrays.items()} == {
        name: value.shape for name, value in other_dict_of_arrays.items()
    }


def test_getitem_with_another_comparator(memory_full_cache, monkeypatch) -> None:
    """Check that a lookup finds an entry equal for a comparator replacing the default.

    Two input data equal for such a comparator can have different hashes,
    so the hash of the input data is not enough to conclude that the entry is missing.
    """
    memory_full_cache.cache_outputs({"x": array([1.0])}, {"y": array([1.0])})
    monkeypatch.setattr(
        memory_full_cache, "compare_dict_of_arrays", _compare_names_and_shapes
    )

    cache_entry = memory_full_cache[{"x": array([2.0])}]

    assert compare_dict_of_arrays(cache_entry.outputs, {"y": array([1.0])})


@pytest.mark.parametrize(
    ("value", "other_value"),
    [
        (array([1]), array([1.0])),
        (array([1.0]), array([1.0 + 0.0j])),
        (array([1.0], dtype="float32"), array([1.0])),
        (array([True]), array([1])),
        (array([0.0]), array([-0.0])),
        (array([complex(1.0, -0.0)]), array([1.0])),
        (array([1.0 + 1.0j]), array([1.0 + 1.0j])),
        (array(["a"]), array(["a"], dtype="<U5")),
        (array([b"\xe9"]), array([b"\xe9"], dtype="S5")),
        (
            array(["2020-01-01"], dtype="datetime64[D]"),
            array(["2020-01-01"], dtype="datetime64[s]"),
        ),
        (
            array(["3000-01-01"], dtype="datetime64[D]"),
            array(["3000-01-01"], dtype="datetime64[s]"),
        ),
        (array([1], dtype="timedelta64[D]"), array([24], dtype="timedelta64[h]")),
    ],
    ids=[
        "integer",
        "complex",
        "single",
        "boolean",
        "signed zero",
        "complex signed zero",
        "complex imaginary",
        "text",
        "non-ASCII bytes",
        "datetime unit",
        "datetime out of the nanosecond range",
        "timedelta unit",
    ],
)
def test_hash_data_of_equal_values(value, other_value) -> None:
    """Check that data equal for a cache have the same hash."""
    data = {"x": value}
    other_data = {"x": other_value}
    assert compare_dict_of_arrays(data, other_data)
    assert hash_data(data) == hash_data(other_data)


@pytest.mark.parametrize(
    ("value", "other_value"),
    [
        (array([1]), array([2])),
        (array([1.0]), array([1.0 + 1.0j])),
        (array([1.0 + 1.0j]), array([1.0 + 2.0j])),
        (array(["a"]), array(["ab"])),
        (array(["ab", "c"]), array(["a", "bc"])),
    ],
    ids=["integer", "complex", "complex imaginary", "text", "text items"],
)
def test_getitem_of_different_values(memory_full_cache, value, other_value) -> None:
    """Check that data differing in value are not mistaken for one another."""
    memory_full_cache.cache_outputs({"x": value}, {"y": array([1.0])})

    assert not memory_full_cache[{"x": other_value}].outputs


@pytest.mark.parametrize(
    ("value", "other_value"),
    [
        (array([1]), array([1.0])),
        (array([0.0]), array([-0.0])),
        (
            array(["2020-01-01"], dtype="datetime64[D]"),
            array(["2020-01-01"], dtype="datetime64[s]"),
        ),
    ],
    ids=["integer", "signed zero", "datetime unit"],
)
def test_getitem_of_another_data_type(memory_full_cache, value, other_value) -> None:
    """Check that an entry is found with input data of another data type."""
    output_data = {"y": array([2.0])}
    memory_full_cache.cache_outputs({"x": value}, output_data)

    cache_entry = memory_full_cache[{"x": other_value}]

    assert compare_dict_of_arrays(cache_entry.outputs, output_data)


class _ColorConverter:
    """A data converter of color names to numbers."""

    colors: ClassVar[tuple[str, ...]] = ("red", "green")
    """The colors, whose numbers are their positions."""

    def convert_value_to_array(self, name: str, value: str) -> ndarray:  # noqa: D102
        return array([float(self.colors.index(value))])

    def convert_array_to_value(self, name: str, value: ndarray) -> str:  # noqa: D102
        return self.colors[int(value[0])]


def test_reopen_with_input_data_converter(tmp_wd, monkeypatch) -> None:
    """Check that an entry is found again in a reopened file with input converter.

    The file stores the hash of the input data converted to arrays.
    """
    converter = _ColorConverter()
    cache = HDF5Cache(
        hdf_file_path=cache_file_name,
        hdf_node_path="node",
        input_data_converter=converter,
    )
    cache.cache_outputs({"color": "red"}, {"y": array([2.0])})

    _forget_file_singletons(monkeypatch)
    new_cache = HDF5Cache(
        hdf_file_path=cache_file_name,
        hdf_node_path="node",
        input_data_converter=converter,
    )

    assert len(new_cache) == 1
    cache_entry = new_cache[{"color": "red"}]
    assert compare_dict_of_arrays(cache_entry.outputs, {"y": array([2.0])})
