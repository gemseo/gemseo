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
# Antoine DECHAUME
from __future__ import annotations

import json
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy.testing import assert_array_equal
from pandas import DataFrame

from gemseo.core.discipline_data import DisciplineData

if TYPE_CHECKING:
    from collections.abc import Mapping


def to_df_key(
    key1: str,
    key2: str,
) -> str:
    return f"{key1}{DisciplineData.SEPARATOR}{key2}"


def assert_equal(disc_data_1, disc_data_2, shallow_copy: bool) -> None:
    """Assert that 2 disciplines data object are equal."""
    assert disc_data_2 == disc_data_1

    if shallow_copy:
        assert disc_data_1["z"] is disc_data_2["z"]
    else:
        assert disc_data_1["z"] is not disc_data_2["z"]

    # For data frames 'is' cannot be used, we check indirectly.
    disc_data_1["x~a"][0] = 1
    if shallow_copy:
        assert disc_data_2["x~a"] == disc_data_1["x~a"]
    else:
        assert disc_data_2["x~a"] != disc_data_1["x~a"]

    # Adding a key is not propagated to the copy.
    disc_data_1["w"] = 0
    assert "w" not in disc_data_2


def test_copy():
    """Verify copy()."""
    df = DataFrame(data={"a": [0]})
    data = {"x": df, "y": 0, "z": [0]}
    d = DisciplineData(data)
    d_copy = d.copy()

    assert_equal(d, d_copy, True)


def test_deepcopy():
    """Verify deepcopy()."""
    df = DataFrame(data={"a": [0]})
    data = {"x": df, "y": 0, "z": [0]}
    d = DisciplineData(data)
    d_copy = deepcopy(d)

    assert_equal(d, d_copy, False)


@pytest.mark.parametrize("with_namespace", [True, False])
def test_copy_keys_namespace(with_namespace):
    """Verify the copy with keys and namespace."""
    data = DisciplineData()
    data["ns:name"] = 0
    data["other_name"] = 0
    data_copy = data.copy(
        keys=("ns:name", "non-existing-name"), with_namespace=with_namespace
    )
    prefix = "ns:" if with_namespace else ""
    assert data_copy.keys() == {f"{prefix}name"}


def assert_getitem(
    d: Mapping[str, Any],
    df: DataFrame,
) -> None:
    assert d["x"].equals(df)
    assert_array_equal(d[to_df_key("x", "a")], df["a"])
    assert d["y"] == 0

    with pytest.raises(KeyError, match="foo"):
        d["foo"]

    with pytest.raises(KeyError, match="foo"):
        d[to_df_key("x", "foo")]


@pytest.mark.parametrize("namespace_mapping", [{}, {"z": "ns:z"}])
def test_getitem(namespace_mapping):
    """Verify __getitem__()."""
    df = DataFrame(data={"a": [0]})
    data = {"x": df, "y": 0}
    d = DisciplineData(
        data,
        input_to_namespaced=namespace_mapping,
        output_to_namespaced=namespace_mapping,
    )

    assert_getitem(d, df)


def test_len():
    """Verify len()."""
    length = 2
    df = DataFrame(data=dict.fromkeys(range(length), [0]))
    data = {"x": df, "y": None}
    d = DisciplineData(data)
    assert len(d) == length + 1


def test_iter():
    """Verify __iter__()."""
    length = 2
    df = DataFrame(data=dict.fromkeys(map(str, range(length)), [0]))
    data = {"x": df, "y": None}
    d = DisciplineData(data)

    ref_keys = (to_df_key("x", "0"), to_df_key("x", "1"), "y")

    assert tuple(d) == ref_keys


def test_delitem():
    """Verify __delitem__()."""
    leaf_data = [0]
    df = DataFrame(data={"a": leaf_data, "b": leaf_data})
    data = {"x": df, "y": 0}
    d = DisciplineData(data)

    with pytest.raises(KeyError, match="foo"):
        del d["foo"]

    with pytest.raises(KeyError, match="foo"):
        del d[to_df_key("x", "foo")]

    del d[to_df_key("x", "a")]
    assert "x~a" not in d
    del d[to_df_key("x", "b")]
    assert "x" not in d
    del d["y"]


def test_setitem():
    """Verify __setitem__()."""
    d = DisciplineData({})

    d["y"] = 0
    assert d["y"] == 0

    # Create a new data frame.
    d[to_df_key("x", "a")] = [0]
    assert d["x"].equals(DataFrame(data={"a": [0]}))

    # Extend the data frame.
    d[to_df_key("x", "b")] = [0]
    assert d["x"].equals(DataFrame(data={"a": [0], "b": [0]}))

    msg = "Cannot set {} because y is not bound to a pandas DataFrame.".format(
        to_df_key("y", "a")
    )
    with pytest.raises(KeyError, match=msg):
        d[to_df_key("y", "a")] = 0


def test_repr():
    """Verify __repr__()."""
    df = DataFrame(data={"a": [0]})
    data = {"x": df, "y": 0}
    d = DisciplineData(data)

    assert repr(d) == repr(data)


def test_keys():
    """Verify the shared dict keys have no separator."""
    msg = "{} shall not contain {}.".format(
        to_df_key("x", "x"), DisciplineData.SEPARATOR
    )
    with pytest.raises(KeyError, match=msg):
        DisciplineData({to_df_key("x", "x"): 0})


def test_init():
    """Verify that creating a DisciplineData from another one shares contents."""
    assert not DisciplineData()
    data = {}
    d1 = DisciplineData(data)
    d2 = DisciplineData(d1)
    d2["x"] = 0
    assert data["x"] == d2["x"]


def test_init_error():
    """Verify that creating a DisciplineData with a bad key raises an error."""
    msg = r"The key bad~key shall not contain ~\."
    with pytest.raises(KeyError, match=msg):
        DisciplineData({"bad~key": 0})


def test_getitem_ns():
    """Verify access of data from keys without namespaces."""
    data = DisciplineData(
        {"ns:x": 1, "ns:y": 2, "z": 0},
        input_to_namespaced={"x": "ns:x"},
        output_to_namespaced={"y": "ns:y"},
    )
    assert data["ns:x"] == 1
    assert data["x"] == 1
    assert data.get("x") == 1
    assert data.get("ns:x") == 1

    assert data["ns:y"] == 2
    assert data["y"] == 2
    assert data.get("y") == 2
    assert data.get("ns:y") == 2


def test_contains():
    """Verify that the in operator works."""
    d = DisciplineData({"x": 0})
    assert "x" in d
    assert "y" not in d


exclude_names = pytest.mark.parametrize(
    "exclude_names",
    [
        (),
        ["dummy"],
        ["name"],
    ],
)


@exclude_names
def test_update(exclude_names):
    """Verify the defaults update."""
    data = DisciplineData({"name": 0})
    data_before = dict(data)
    other_data = {"name": 1}
    data.update(other_data, exclude=exclude_names)
    for name, value in data.items():
        if name in exclude_names:
            assert value == data_before[name]
        else:
            assert value == other_data[name]


def test_restrict():
    """Verify the restriction."""
    data = DisciplineData()
    data["name"] = 0
    data["other_name"] = 0
    data.restrict("name", "non-existing-name")
    assert data.keys() == {"name"}


def test_wrong_data_type():
    """Verify that the type of the initial data is well checked."""
    data = ("a",)
    with pytest.raises(TypeError, match=f"Invalid type for data, got {type(data)}."):
        DisciplineData(data)


def test_serialization():
    """Verify serialization of nested data."""
    assert json.dumps(dict(DisciplineData({"a": {"b": "c"}}))) == '{"a": {"b": "c"}}'
