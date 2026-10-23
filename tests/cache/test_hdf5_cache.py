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

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import array
from numpy import ndarray
from numpy import ones
from numpy.testing import assert_array_equal
from pydantic import BaseModel

from gemseo.core.cache.factory import CacheFactory
from gemseo.core.data_converter.base import BaseDataConverter
from gemseo.core.grammar.simple import SimpleGrammar
from gemseo.discipline import Discipline
from gemseo.util.pydantic_ndarray import NDArrayPydantic
from gemseo.util.repr_html import repr_html_wrapper
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.core.cache.hdf5 import HDF5Cache


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
