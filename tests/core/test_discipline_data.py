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

from copy import deepcopy
from typing import Any
from typing import Mapping

import pandas as pd
import pytest
from gemseo.core.discipline_data import DisciplineData
from numpy.testing import assert_array_equal


def to_df_key(
    key1: str,
    key2: str,
) -> str:
    return f"{key1}{DisciplineData.SEPARATOR}{key2}"


def test_copy():
    """Verify copy()."""
    leaf_data = [0]
    df = pd.DataFrame(data={"a": leaf_data})
    data = {"x": df, "y": 0, "z": [0]}
    d = DisciplineData(deepcopy(data))
    d_copy = d.copy()

    assert d.keys() == d_copy.keys()

    # Adding a key is not propagated to the copy.
    d["w"] = 0
    assert "w" not in d_copy

    # Item's values are shared.
    assert d["z"] is d_copy["z"]
    assert d["x"] is d_copy["x"]
    assert d["x"]["a"] is d_copy["x"]["a"]


def assert_getitem(
    d: Mapping[str, Any],
    df: pd.DataFrame,
) -> None:
    assert d["x"].equals(df)
    assert_array_equal(d[to_df_key("x", "a")], df["a"])
    assert d["y"] == 0

    with pytest.raises(KeyError, match="foo"):
        d["foo"]

    with pytest.raises(KeyError, match="foo"):
        d[to_df_key("x", "foo")]


def test_getitem():
    """Verify __getitem__()."""
    df = pd.DataFrame(data={"a": [0]})
    data = {"x": df, "y": 0}
    d = DisciplineData(data)

    assert_getitem(d, df)

    # With nested dictionary.
    data.update({"z": data.copy()})
    assert_getitem(d["z"], df)


def test_len():
    """Verify len()."""
    length = 2
    df = pd.DataFrame(data=dict.fromkeys(range(length), [0]))
    data = {"x": df, "y": None}
    d = DisciplineData(data)
    assert len(d) == length + 1


def test_iter():
    """Verify __iter__()."""
    length = 2
    df = pd.DataFrame(data=dict.fromkeys(map(str, range(length)), [0]))
    data = {"x": df, "y": None}
    d = DisciplineData(data)

    ref_keys = (to_df_key("x", "0"), to_df_key("x", "1"), "y")

    assert tuple(d) == ref_keys


def test_delitem():
    """Verify __delitem__()."""
    leaf_data = [0]
    df = pd.DataFrame(data={"a": leaf_data})
    data = {"x": df, "y": 0}
    d = DisciplineData(data)

    with pytest.raises(KeyError, match="foo"):
        del d["foo"]

    with pytest.raises(KeyError, match="foo"):
        del d[to_df_key("x", "foo")]

    del d[to_df_key("x", "a")]
    del d["x"]
    del d["y"]


def test_setitem():
    """Verify __setitem__()."""
    d = DisciplineData({})

    d["y"] = 0
    assert d["y"] == 0

    # Create a new data frame.
    d[to_df_key("x", "a")] = [0]
    assert d["x"].equals(pd.DataFrame(data={"a": [0]}))

    # Extend the data frame.
    d[to_df_key("x", "b")] = [0]
    assert d["x"].equals(pd.DataFrame(data={"a": [0], "b": [0]}))

    msg = "Cannot set {} because y is not bound to a pandas DataFrame.".format(
        to_df_key("y", "a")
    )
    with pytest.raises(KeyError, match=msg):
        d[to_df_key("y", "a")] = 0


def test_repr():
    """Verify __repr__()."""
    df = pd.DataFrame(data={"a": [0]})
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
    data = {}
    d1 = DisciplineData(data)
    d2 = DisciplineData(d1)
    d2["x"] = 0
    assert data["x"] == d2["x"]


def test_get_ns():
    """Verify access of data from keys without namespaces."""
    data = DisciplineData(
        {"ns:x": 1, "ns:y": 2},
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
