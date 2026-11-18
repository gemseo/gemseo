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

import pickle
import subprocess
import sys
from contextlib import contextmanager
from multiprocessing.util import _finalizer_registry
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING
from typing import Any

import h5py
import pytest
from numpy import array
from numpy import ndarray
from numpy import ones
from numpy.testing import assert_array_equal
from pydantic import BaseModel
from scipy.sparse import csr_array

from gemseo.core.cache._hdf5_file_singleton import HDF5FileSingleton
from gemseo.core.cache._hdf5_file_singleton import _close_files
from gemseo.core.cache._hdf5_file_singleton import _close_idle_files
from gemseo.core.cache._hdf5_file_singleton import _forget_parent_process_state
from gemseo.core.cache._hdf5_file_singleton import _open_files
from gemseo.core.cache.factory import CacheFactory
from gemseo.core.cache.util import hash_data
from gemseo.core.data_converter.base import BaseDataConverter
from gemseo.core.grammar.simple import SimpleGrammar
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import Discipline
from gemseo.discipline import ParallelDisciplineChain
from gemseo.doe import PYDOE_LHS_Settings
from gemseo.enum import CacheType
from gemseo.formulation import DisciplinaryOpt_Settings
from gemseo.scenario import MDOScenario
from gemseo.space import DesignSpace
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.repr_html import repr_html_wrapper
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Generator
    from collections.abc import Iterator
    from contextlib import AbstractContextManager

    from gemseo.core.cache.hdf5 import HDF5Cache


@pytest.fixture(autouse=True)
def close_files() -> Iterator[None]:
    """Close the HDF files left open by a test, e.g. with file_kept_open.

    An open file would leak into the next tests
    and prevent the removal of the temporary directory on Windows.
    """
    yield
    _close_files()


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
    assert cache._repr_html_() == repr_html_wrapper.format(
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
    assert cache.get(inputs).outputs == outputs


def test_hdf_node_path(tmp_wd) -> None:
    """Check the property hdf_node_path."""
    assert create_cache().hdf_node_path == "Dummy"
    assert create_cache("foo").hdf_node_path == "foo"


class _PathOutputs(BaseModel):
    y: NDArrayPydantic = ones(1)
    some_path: Path = Path("some_path")


class _PathDiscipline(Discipline):
    """A discipline with a path input and a path output."""

    default_grammar_type = Discipline.GrammarType.PYDANTIC

    def __init__(self, hdf_file_path: str, path_type: Any = Path) -> None:
        super().__init__()
        self.input_grammar.update_from_types({"x": NDArrayPydantic, "p": path_type})
        self.output_grammar.update_from_model(_PathOutputs)
        self.default_input_data = {"x": ones(1), "p": Path("in")}
        self.set_cache(Discipline.CacheType.HDF5, hdf_file_path=hdf_file_path)
        self.n_calls = 0

    def _run(self, input_data):
        self.n_calls += 1
        return {"y": input_data["x"], "some_path": Path("out") / input_data["p"]}


@pytest.fixture
def path_cache_file(tmp_wd) -> str:
    """The path to an HDF5 file filled by a discipline with path input and output."""
    file_path = "path_cache.h5"
    _PathDiscipline(file_path).execute()
    return file_path


def test_cache_path_hit(path_cache_file) -> None:
    """Check that the same path input reads the path output from an HDF5Cache."""
    discipline = _PathDiscipline(path_cache_file)
    discipline.execute()
    assert discipline.n_calls == 0
    output = discipline.io.output_data["some_path"]
    assert isinstance(output, Path)
    assert output == Path("out") / "in"


def test_cache_path_miss(path_cache_file) -> None:
    """Check that a different path input is a cache miss for an HDF5Cache."""
    discipline = _PathDiscipline(path_cache_file)
    discipline.execute({"p": Path("other")})
    assert discipline.n_calls == 1
    assert discipline.io.output_data["some_path"] == Path("out") / "other"


def test_cache_addition_with_paths(tmp_wd) -> None:
    """Check that adding HDF5 caches keeps the path values."""
    discipline = _PathDiscipline("cache_d.h5")
    other = _PathDiscipline("cache_e.h5")
    discipline.execute({"p": Path("a")})
    other.execute({"p": Path("b")})
    cache = discipline.cache + other.cache
    assert len(cache) == 2
    for p, q in (("a", "out/a"), ("b", "out/b")):
        outputs = cache[{"x": ones(1), "p": Path(p)}].outputs
        assert isinstance(outputs["some_path"], Path)
        assert outputs["some_path"] == Path(q)


def test_cache_str_or_path_hit(tmp_wd) -> None:
    """Check that a path input hits an HDF5Cache entry read back as a str."""
    discipline = _PathDiscipline("str_or_path_cache.h5", str | Path)
    discipline.execute()
    discipline.execute({"x": 2 * ones(1)})
    discipline.execute({"x": ones(1)})
    assert discipline.n_calls == 2


def test_cache_str_array_without_type(tmp_wd) -> None:
    """Check that a one-item string array of an item without type hits the cache."""
    input_grammar = SimpleGrammar("inputs")
    input_grammar.update_from_types({"x": None})
    output_grammar = SimpleGrammar("outputs")
    output_grammar.update_from_types({"y": None})
    cache = CacheFactory().create(
        "HDF5Cache",
        hdf_file_path="dummy.h5",
        input_data_converter=input_grammar.data_converter,
        output_data_converter=output_grammar.data_converter,
    )
    cache[{"x": array(["a"])}] = ({"y": array(["b"])}, None)
    outputs = cache[{"x": array(["a"])}].outputs
    assert isinstance(outputs["y"], ndarray)
    assert_array_equal(outputs["y"], array(["b"]))


@pytest.mark.parametrize(
    ("path_type", "value", "expected"),
    [(Path, Path("dé/x"), Path("out/dé/x")), (str, "é", Path("out/é"))],
    ids=["path", "str"],
)
def test_cache_non_ascii_round_trip(tmp_wd, path_type, value, expected) -> None:
    """Check that a non-ASCII path or str is stored, read back and hit."""
    discipline = _PathDiscipline("non_ascii.h5", path_type)
    discipline.execute({"x": ones(1), "p": value})
    outputs = discipline.cache[{"x": ones(1), "p": value}].outputs
    assert outputs["some_path"] == expected


def test_cache_non_ascii_reopen_with_str_or_path(tmp_wd) -> None:
    """Check that a reopened cache with non-ASCII paths and strs is hit."""
    file_path = "non_ascii.h5"
    _PathDiscipline(file_path).execute({"x": ones(1), "p": Path("dé/x")})
    _PathDiscipline(file_path, str).execute({"x": ones(1), "p": "é"})

    new_discipline = _PathDiscipline(file_path, str | Path)
    assert len(new_discipline.cache) == 2
    new_discipline.execute({"x": ones(1), "p": Path("dé/x")})
    new_discipline.execute({"x": ones(1), "p": "é"})
    assert new_discipline.n_calls == 0
    assert len(new_discipline.cache) == 2


def test_cache_optional_path_none(tmp_wd, snapshot) -> None:
    """Check that a None value of a path variable raises an error, without entry."""
    discipline = _PathDiscipline("optional_path.h5", Path | None)
    cache = discipline.cache
    with assert_exception(TypeError, snapshot):
        cache.cache_outputs({"x": ones(1), "p": None}, {"y": ones(1)})
    assert len(cache) == 0


def test_cache_object_array(tmp_wd, snapshot) -> None:
    """Check that an array of Python objects cannot be cached."""
    cache = create_cache()
    with assert_exception(TypeError, snapshot):
        cache.cache_outputs({"x": ones(1)}, {"y": array([object(), object()])})


def test_cache_path_with_custom_converter(tmp_wd, monkeypatch) -> None:
    """Check that the hash does not depend on the converter of a path variable."""
    monkeypatch.setattr(
        BaseDataConverter,
        "value_to_array_converters",
        {"p": lambda value: array([f"X{value.as_posix()}"])},
    )
    monkeypatch.setattr(
        BaseDataConverter,
        "array_to_value_converters",
        {"p": lambda value: Path(str(value[0])[1:])},
    )
    file_path = "custom_converter.h5"
    _PathDiscipline(file_path).execute()
    new_discipline = _PathDiscipline(file_path)
    new_discipline.execute()
    assert new_discipline.n_calls == 0


def test_failed_write_keeps_entry_with_same_hash(tmp_wd, monkeypatch) -> None:
    """Check that a failed write keeps the other entry sharing its hash."""
    for module in ("base_full", "hdf5"):
        monkeypatch.setattr(f"gemseo.core.cache.{module}.hash_data", lambda _: 1)
    cache = create_cache()
    cache.cache_outputs({"x": array([1.0])}, {"y": array([1.0])})
    with pytest.raises(TypeError):
        cache.cache_outputs(
            {"x": array([2.0]), "p": array([None])}, {"y": array([2.0])}
        )
    assert len(cache) == 1
    assert_array_equal(cache[{"x": array([1.0])}].outputs["y"], array([1.0]))


def reopen_cache(
    forget_file_singletons: Callable[[], None], hdf_node_path: str = "Dummy"
) -> HDF5Cache:
    """Create an HDF5 cache from an existing file, with a brand new singleton.

    Args:
        forget_file_singletons: The function replacing the registry of the singletons.
        hdf_node_path: The name of the HDF node.

    Returns:
        An HDF5 cache reading the entries already stored in the file.
    """
    forget_file_singletons()
    return create_cache(hdf_node_path)


def _create_doe_scenario() -> tuple[AnalyticDiscipline, MDOScenario]:
    """Create a DOE scenario whose discipline has an HDF5 cache.

    Returns:
        The discipline and the scenario.
    """
    discipline = AnalyticDiscipline({"y": "2*x"})
    discipline.set_cache(CacheType.HDF5, hdf_file_path="dummy.h5", hdf_node_path="doe")
    design_space = DesignSpace()
    design_space.add_variable("x", lower_bound=0.0, upper_bound=1.0)
    scenario = MDOScenario(
        [discipline], design_space, formulation_settings=DisciplinaryOpt_Settings()
    )
    scenario.add_objective("y")
    return discipline, scenario


def test_doe_multiprocessing_with_shared_cache(tmp_wd, forget_file_singletons) -> None:
    """Check a DOE using multiprocessing with a cache shared by the processes."""
    n_samples = 100
    _, scenario = _create_doe_scenario()
    scenario.execute(PYDOE_LHS_Settings(n_samples=n_samples, seed=1, n_processes=2))

    # The entries have been written by the two processes in the same node.
    new_cache = reopen_cache(forget_file_singletons, "doe")
    assert len(new_cache) == n_samples
    for input_data, cache_entry in new_cache.items():
        assert_array_equal(cache_entry.outputs["y"], 2 * input_data["x"])


def _keep_open_context(cache: HDF5Cache) -> AbstractContextManager[None]:
    """Keep the file of a cache open with the dedicated context manager.

    Args:
        cache: The cache.

    Returns:
        The context manager keeping the file open.
    """
    return cache.hdf_file.keep_open()


@contextmanager
def _keep_file_open(cache: HDF5Cache) -> Generator[None, None, None]:
    """Keep the file of a cache open with the dedicated option.

    Args:
        cache: The cache.

    Yields:
        Nothing.
    """
    cache.file_kept_open = True
    try:
        yield
    finally:
        cache.file_kept_open = False
        cache.close_file()


@pytest.mark.parametrize(
    "keep_open",
    [_keep_open_context, _keep_file_open],
    ids=["keep_open() context", "file_kept_open option"],
)
@pytest.mark.parametrize(
    "execute_first", [False, True], ids=["DOE only", "execution then DOE"]
)
def test_doe_multiprocessing_with_file_kept_open(
    tmp_wd, keep_open, execute_first, forget_file_singletons
) -> None:
    """Check a DOE using multiprocessing while the parent keeps the file open.

    The worker processes shall not keep the file open,
    otherwise they would lock it out of each other
    and lose the points evaluated by the other ones,
    and the parent process closes the file before forking them,
    otherwise they would inherit the open file and its HDF5 lock.
    """
    n_samples = 6
    discipline, scenario = _create_doe_scenario()
    with keep_open(discipline.cache):
        if execute_first:
            discipline.execute({"x": array([5.0])})

        scenario.execute(PYDOE_LHS_Settings(n_samples=n_samples, seed=1, n_processes=2))

    assert len(reopen_cache(forget_file_singletons, "doe")) == n_samples + execute_first


def test_close_idle_files_before_fork(tmp_wd) -> None:
    """Check that the files no operation uses are closed before forking."""
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    _close_idle_files()

    assert cache.hdf_file not in _open_files


def test_close_idle_files_before_fork_keeps_file_in_use(tmp_wd) -> None:
    """Check that a file an operation uses is not closed before forking."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    hdf_file = cache.hdf_file
    with hdf_file._HDF5FileSingleton__open():
        _close_idle_files()

        assert hdf_file in _open_files
        assert hdf_file.has_group(1, cache.Group.INPUTS, "Dummy")


def test_forget_open_file_in_forked_process(tmp_wd, monkeypatch) -> None:
    """Check that a forked process drops the file of its parent without closing it."""
    parent_files_kept_alive = []
    monkeypatch.setattr(
        "gemseo.core.cache._hdf5_file_singleton._parent_files_kept_alive",
        parent_files_kept_alive,
    )
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    handle = cache.hdf_file._HDF5FileSingleton__file

    _forget_parent_process_state()

    assert cache.hdf_file not in _open_files
    assert parent_files_kept_alive == [handle]
    assert handle.id.valid
    handle.close()


def test_forget_open_file_of_a_copy_in_forked_process(tmp_wd, monkeypatch) -> None:
    """Check that a forked process drops the open file of a copy of a singleton.

    A copy, e.g. unpickled, is not in the registry of the singletons.
    """
    parent_files_kept_alive = []
    monkeypatch.setattr(
        "gemseo.core.cache._hdf5_file_singleton._parent_files_kept_alive",
        parent_files_kept_alive,
    )
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    copy = pickle.loads(pickle.dumps(cache.hdf_file))
    copy._file_kept_open = True
    assert copy.has_group(1, cache.Group.INPUTS, "Dummy")
    handle = copy._HDF5FileSingleton__file

    _forget_parent_process_state()

    assert copy not in _open_files
    assert not copy._file_kept_open
    assert parent_files_kept_alive == [handle]
    handle.close()


@pytest.mark.parametrize(
    "keep_open",
    [_keep_open_context, _keep_file_open],
    ids=["keep_open() context", "file_kept_open option"],
)
def test_forget_keeping_file_open_in_forked_process(tmp_wd, keep_open) -> None:
    """Check that a forked process does not keep the file open as its parent."""
    cache = create_cache()
    with keep_open(cache):
        _forget_parent_process_state()
        cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
        assert cache.hdf_file not in _open_files


def test_parallel_chain_multiprocessing_with_shared_cache(
    tmp_wd, forget_file_singletons
) -> None:
    """Check a parallel chain using multiprocessing with a shared cache file."""
    n_disciplines = 4
    disciplines = []
    for index in range(n_disciplines):
        discipline = AnalyticDiscipline({f"y_{index}": f"{index + 1}*x"})
        discipline.set_cache(
            CacheType.HDF5, hdf_file_path="dummy.h5", hdf_node_path=f"node_{index}"
        )
        disciplines.append(discipline)

    chain = ParallelDisciplineChain(disciplines, use_threading=False, n_processes=2)
    for x in (1.0, 2.0):
        chain.execute({"x": array([x])})

    # The entries have been written by the sub-processes;
    # re-read them from the file with brand new caches.
    for index in range(n_disciplines):
        new_cache = reopen_cache(forget_file_singletons, f"node_{index}")
        assert len(new_cache) == 2, f"node_{index}"
        for x in (1.0, 2.0):
            assert_array_equal(
                new_cache[{"x": array([x])}].outputs[f"y_{index}"],
                array([(index + 1) * x]),
            )


_input_data = {"i": ones(1)}
"""The input data of the entry of a cache keeping its file open."""

_output_data = {"o": ones(2)}
"""The output data of the entry of a cache keeping its file open."""


@pytest.fixture
def cache_keeping_file_open(tmp_wd) -> HDF5Cache:
    """An HDF5 cache keeping its file open, with an entry."""
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs(_input_data, _output_data)
    return cache


def test_keep_file_open_after_write(cache_keeping_file_open) -> None:
    """Check that the file is kept open after a write."""
    assert cache_keeping_file_open.hdf_file in _open_files


def test_keep_file_open_read(cache_keeping_file_open) -> None:
    """Check that an entry is read while the file is kept open."""
    outputs = cache_keeping_file_open[_input_data].outputs

    assert_array_equal(outputs["o"], _output_data["o"])
    assert cache_keeping_file_open.hdf_file in _open_files


def test_keep_file_open_close(cache_keeping_file_open) -> None:
    """Check that a file kept open can be closed."""
    cache_keeping_file_open.hdf_file.close()

    assert cache_keeping_file_open.hdf_file not in _open_files


def test_keep_file_open_close_twice(cache_keeping_file_open) -> None:
    """Check that closing a file already closed does nothing."""
    cache_keeping_file_open.hdf_file.close()

    cache_keeping_file_open.hdf_file.close()

    assert cache_keeping_file_open.hdf_file not in _open_files


def test_keep_file_open_read_after_close(cache_keeping_file_open) -> None:
    """Check that a file kept open and closed is opened again by the next read."""
    cache_keeping_file_open.hdf_file.close()

    outputs = cache_keeping_file_open[_input_data].outputs

    assert_array_equal(outputs["o"], _output_data["o"])
    assert cache_keeping_file_open.hdf_file in _open_files


def test_file_closed_by_default(tmp_wd) -> None:
    """Check that the file is closed after an operation by default."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    assert cache.hdf_file not in _open_files


def test_close_files(tmp_wd) -> None:
    """Check that the function run at the end of the process closes the open files."""
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    handle = cache.hdf_file._HDF5FileSingleton__file
    assert handle.id.valid

    _close_files()

    assert cache.hdf_file not in _open_files
    # The HDF5 file itself is closed, not only forgotten.
    assert not handle.id.valid


def test_close_on_exception(tmp_wd, forget_file_singletons) -> None:
    """Check that the file is closed when the process raises an exception."""
    script_path = tmp_wd / "raise_after_caching.py"
    script_path.write_text(
        dedent(
            """
            from __future__ import annotations

            from numpy import ones

            from gemseo.core.cache.hdf5 import HDF5Cache

            # The cache relies on multiprocessing features,
            # which require this guard on Windows.
            if __name__ == "__main__":
                cache = HDF5Cache(hdf_file_path="dummy.h5", hdf_node_path="Dummy")
                cache.file_kept_open = True
                cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
                raise ValueError("Something went wrong.")
            """
        )
    )
    process = subprocess.run(  # noqa: S603
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_wd,
    )
    assert process.returncode != 0
    assert "Something went wrong." in process.stderr

    # The file has been closed, so it can be read and the entry is there.
    cache = reopen_cache(forget_file_singletons)
    assert len(cache) == 1
    assert_array_equal(cache[{"i": ones(1)}].outputs["o"], ones(1))


def test_open_retries_while_locked(tmp_wd, monkeypatch) -> None:
    """Check that opening a file locked by another process is retried."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    open_file = h5py.File
    n_calls = 0

    def fail_twice(*args, **kwargs):
        nonlocal n_calls
        n_calls += 1
        if n_calls <= 2:
            msg = "Unable to open file (unable to lock file, errno = 11)"
            raise OSError(msg)
        return open_file(*args, **kwargs)

    monkeypatch.setattr(h5py, "File", fail_twice)

    # A single read, hence a single opening of the file, is retried twice.
    assert cache.hdf_file.read_data(1, cache.Group.OUTPUTS, "Dummy")
    assert n_calls == 3


def test_open_gives_up_when_locked(tmp_wd, monkeypatch, caplog) -> None:
    """Check the error raised when a file stays locked by another process."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    def always_locked(*args, **kwargs):
        msg = "Unable to open file (unable to lock file, errno = 11)"
        raise OSError(msg)

    monkeypatch.setattr(h5py, "File", always_locked)
    cache.timeout = 0.0

    with pytest.raises(OSError, match=r"unable to lock file"):
        cache.hdf_file.read_data(1, cache.Group.OUTPUTS, "Dummy")

    assert "is still locked after 0.0 seconds" in caplog.text


def test_open_does_not_retry_other_errors(tmp_wd, monkeypatch) -> None:
    """Check that an error other than a lock one is not retried."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    n_calls = 0

    def fail(*args, **kwargs):
        nonlocal n_calls
        n_calls += 1
        msg = "No space left on device"
        raise OSError(msg)

    monkeypatch.setattr(h5py, "File", fail)

    with pytest.raises(OSError, match=r"No space left on device"):
        cache.hdf_file.read_data(1, cache.Group.OUTPUTS, "Dummy")

    assert n_calls == 1


def test_keep_file_open_switches_to_append_mode(tmp_wd) -> None:
    """Check that a file kept open for reading is reopened to be written."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    assert cache.hdf_file not in _open_files

    cache.file_kept_open = True
    # The file is read, hence opened in read-only mode, and kept open.
    assert cache[{"i": ones(1)}].outputs
    assert cache.hdf_file in _open_files

    # Writing requires the append mode, so the file is opened again.
    cache.cache_outputs({"i": ones(2)}, {"o": ones(2)})
    assert cache.hdf_file in _open_files
    assert len(cache) == 2
    assert_array_equal(cache[{"i": ones(2)}].outputs["o"], ones(2))


def test_index_allocated_from_file(tmp_wd) -> None:
    """Check that an entry is written at the index allocated from the file.

    The counter of a cache is not shared between the processes, so another process
    may have already used the index it counted; this is simulated here by resetting
    the counter of the cache after a first entry has been written.
    """
    cache = create_cache()
    first_input_data = {"i": ones(1)}
    cache.cache_outputs(first_input_data, {"o": ones(1)})
    assert cache.hdf_file.has_group(1, cache.Group.INPUTS, "Dummy")

    # Another process would count this entry from 0 and pick the index 1 again.
    cache._max_index.value = 0
    cache._last_accessed_index.value = 0

    second_input_data = {"i": ones(2)}
    cache.cache_outputs(second_input_data, {"o": ones(3)})

    # The file allocated the index 2 and the cache took it into account.
    assert cache.hdf_file.has_group(2, cache.Group.INPUTS, "Dummy")
    assert cache._max_index.value == 2
    assert cache._last_accessed_index.value == 2
    assert cache._hash_to_indices[hash_data(second_input_data)].tolist() == [2]
    assert len(cache) == 2
    assert_array_equal(cache[first_input_data].outputs["o"], ones(1))
    assert_array_equal(cache[second_input_data].outputs["o"], ones(3))


def test_keep_open_context(tmp_wd) -> None:
    """Check that the file is kept open in the dedicated context, and closed after."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    assert cache.hdf_file not in _open_files

    with cache.hdf_file.keep_open():
        assert_array_equal(cache[{"i": ones(1)}].outputs["o"], ones(1))
        assert cache.hdf_file in _open_files

    assert cache.hdf_file not in _open_files


def test_keep_open_context_on_exception(tmp_wd) -> None:
    """Check that the file is closed when the dedicated context raises."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    def read_then_raise() -> None:
        """Read the cache in the context and raise."""
        with cache.hdf_file.keep_open():
            assert cache[{"i": ones(1)}].outputs
            msg = "Something went wrong."
            raise ValueError(msg)

    with pytest.raises(ValueError, match=r"Something went wrong"):
        read_then_raise()

    assert cache.hdf_file not in _open_files


def test_pickle(tmp_wd) -> None:
    """Check that a cache can be pickled and keeps its entries."""
    cache = create_cache()
    input_data = {"i": ones(1)}
    output_data = {"o": ones(2)}
    cache.cache_outputs(input_data, output_data)

    cache.file_kept_open = True
    assert cache[input_data].outputs
    assert cache.hdf_file in _open_files

    new_cache = pickle.loads(pickle.dumps(cache))

    # The file of the original cache has been closed when it was pickled,
    # so that the process using the copy can open it.
    assert cache.hdf_file not in _open_files
    assert len(new_cache) == 1
    assert_array_equal(new_cache[input_data].outputs["o"], output_data["o"])
    # The copy, e.g. in a worker process, does not keep the file open,
    # otherwise the worker processes would lock it out of each other.
    assert not new_cache.file_kept_open
    assert new_cache.hdf_file not in _open_files


@pytest.mark.parametrize("file_kept_open", [False, True], ids=["closed", "kept open"])
def test_no_flush(tmp_wd, monkeypatch, file_kept_open) -> None:
    """Check that a write does not flush the file, which closing it does."""
    flushes = []
    flush = h5py.File.flush
    monkeypatch.setattr(
        h5py.File, "flush", lambda file: flushes.append(None) or flush(file)
    )
    cache = create_cache()
    cache.file_kept_open = file_kept_open

    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    assert not flushes


def test_keep_open_context_with_keep_file_open(tmp_wd) -> None:
    """Check that leaving keep_open() does not close a file kept open."""
    cache = create_cache()
    cache.file_kept_open = True
    with cache.hdf_file.keep_open():
        cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    assert cache.hdf_file in _open_files
    cache.hdf_file.close()


def test_close_files_at_process_exit(tmp_wd, monkeypatch) -> None:
    """Check that a process opening a file closes it with a finalizer at exit.

    A worker process forked by multiprocessing does not run atexit but these ones.
    """
    monkeypatch.setattr(
        "gemseo.core.cache._hdf5_file_singleton._processes_closing_files", set()
    )
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    finalizers = [
        finalizer
        for finalizer in list(_finalizer_registry.values())
        if finalizer._callback is _close_files
    ]
    assert finalizers
    assert cache.hdf_file in _open_files

    finalizers[-1]()

    assert cache.hdf_file not in _open_files


def test_allocate_index_reads_no_entry_names(tmp_wd, monkeypatch) -> None:
    """Check that allocating an index does not read the names of all the entries."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    n_iterations = []
    iterate = h5py.Group.__iter__
    monkeypatch.setattr(
        h5py.Group,
        "__iter__",
        lambda group: n_iterations.append(None) or iterate(group),
    )

    cache.cache_outputs({"i": ones(2)}, {"o": ones(1)})

    assert not n_iterations
    with h5py.File("dummy.h5", "r") as h5_file:
        assert h5_file["Dummy"].attrs[HDF5FileSingleton.last_index_tag] == 2


@pytest.mark.parametrize(
    ("written_index", "remove_last_index", "expected_index"),
    [(3, True, 4), (2, False, 3)],
    ids=["node without last index", "index taken without allocation"],
)
def test_allocate_index(
    tmp_wd, written_index, remove_last_index, expected_index
) -> None:
    """Check the allocation of an index.

    The last index is missing from a node written by an older version,
    and an index can be taken by an entry written without allocation.
    """
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    cache.hdf_file.write_data(
        {"i": ones(2)}, cache.Group.INPUTS, written_index, "Dummy"
    )
    if remove_last_index:
        with h5py.File("dummy.h5", "a") as h5_file:
            del h5_file["Dummy"].attrs[HDF5FileSingleton.last_index_tag]

    index = cache.hdf_file.write_new_entry({"i": ones(4)}, "Dummy")

    assert index == expected_index


def test_pickle_during_an_operation(tmp_wd) -> None:
    """Check that pickling the file while an operation uses it does not close it."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    hdf_file = cache.hdf_file
    with hdf_file._HDF5FileSingleton__open():
        pickle.dumps(hdf_file)
        assert hdf_file in _open_files
        assert hdf_file.has_group(1, cache.Group.INPUTS, "Dummy")
        # The nested operation does not close the file either.
        assert hdf_file in _open_files

    assert hdf_file not in _open_files


def test_len_with_another_cache_writing_in_the_node(tmp_wd) -> None:
    """Check that the length is the number of entries, not the largest index."""
    cache = create_cache()
    other_cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    other_cache.cache_outputs({"i": ones(2)}, {"o": ones(1)})

    # The file allocated the index 2 to the only entry of the other cache.
    assert other_cache.hdf_file.has_group(2, other_cache.Group.INPUTS, "Dummy")
    assert len(other_cache) == 1
    assert len(list(other_cache)) == 1


def test_write_while_reading(tmp_wd, snapshot) -> None:
    """Check that the file cannot be opened again to be written while it is read."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    hdf_file = cache.hdf_file
    with (
        hdf_file._HDF5FileSingleton__open(),
        assert_exception(RuntimeError, snapshot),
    ):
        hdf_file.write_data({"i": ones(2)}, cache.Group.INPUTS, 2, "Dummy")

    assert hdf_file not in _open_files


def test_keep_open_context_during_an_operation(tmp_wd) -> None:
    """Check that leaving keep_open() does not close a file an operation uses."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    hdf_file = cache.hdf_file
    with hdf_file._HDF5FileSingleton__open():
        with hdf_file.keep_open():
            pass

        assert hdf_file in _open_files
        assert hdf_file.has_group(1, cache.Group.INPUTS, "Dummy")

    assert hdf_file not in _open_files


def test_nested_keep_open_contexts(tmp_wd) -> None:
    """Check that leaving a nested keep_open() keeps the file open for the outer one."""
    cache = create_cache()
    hdf_file = cache.hdf_file
    with hdf_file.keep_open():
        with hdf_file.keep_open():
            cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

        assert hdf_file in _open_files
        cache.cache_outputs({"i": ones(2)}, {"o": ones(1)})
        assert hdf_file in _open_files

    assert hdf_file not in _open_files


def test_failed_write_of_a_new_entry(tmp_wd, monkeypatch) -> None:
    """Check that a failed write of a new entry leaves no partial entry."""
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    def fail(*args):
        raise OSError

    monkeypatch.setattr(
        HDF5FileSingleton, "_HDF5FileSingleton__write_sparse_array", fail
    )
    with pytest.raises(RuntimeError):
        cache.hdf_file.write_new_entry({"i": csr_array(ones((1, 1)))}, "Dummy", 1)

    with h5py.File("dummy.h5", "r") as h5_file:
        node = h5_file["Dummy"]
        assert list(node) == ["1"]
        assert node.attrs[HDF5FileSingleton.last_index_tag] == 1


def test_keep_file_open_from_the_cache(tmp_wd) -> None:
    """Check that the cache keeps its file open and closes it."""
    cache = create_cache()
    assert not cache.file_kept_open

    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})

    assert cache.hdf_file._file_kept_open
    assert cache.hdf_file in _open_files

    cache.close_file()

    assert cache.hdf_file not in _open_files


def test_timeout_from_the_cache(tmp_wd) -> None:
    """Check that the timeout set from a cache applies to the file and its copies."""
    cache = create_cache()
    assert cache.timeout == 30.0

    cache.timeout = 1.5

    assert cache.hdf_file._timeout == 1.5
    assert create_cache().timeout == 1.5
    assert pickle.loads(pickle.dumps(cache.hdf_file))._timeout == 1.5


def test_close_failure_forgets_the_file(tmp_wd, monkeypatch) -> None:
    """Check that a file failing to close is forgotten all the same."""
    cache = create_cache()
    cache.file_kept_open = True
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    handle = cache.hdf_file._HDF5FileSingleton__file

    def fail(file):
        raise OSError

    with monkeypatch.context() as context:
        context.setattr(h5py.File, "close", fail)
        with pytest.raises(OSError):
            cache.close_file()

    assert cache.hdf_file not in _open_files
    assert cache.hdf_file._HDF5FileSingleton__file is None
    handle.close()


def test_read_hashes_of_entries_sharing_a_hash(
    tmp_wd, monkeypatch, forget_file_singletons
) -> None:
    """Check that a reopened cache indexes all the entries sharing a hash."""
    for module in ("base_full", "hdf5"):
        monkeypatch.setattr(f"gemseo.core.cache.{module}.hash_data", lambda _: 1)
    cache = create_cache()
    cache.cache_outputs({"i": ones(1)}, {"o": ones(1)})
    cache.cache_outputs({"i": ones(2)}, {"o": ones(2)})

    new_cache = reopen_cache(forget_file_singletons)

    assert new_cache._hash_to_indices[1].tolist() == [1, 2]
    assert_array_equal(new_cache[{"i": ones(2)}].outputs["o"], ones(2))
