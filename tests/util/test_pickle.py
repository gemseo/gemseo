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
from __future__ import annotations

import pickle
from pathlib import Path

from numpy import array
from numpy.testing import assert_array_equal

from gemseo.util.pickle import from_pickle
from gemseo.util.pickle import to_pickle


def test_to_and_from_pickle(tmp_wd):
    """Check the round trip between to_pickle and from_pickle."""
    obj = {"x": array([1.0, 2.0]), "name": "foo"}
    file_path = Path("obj.pkl")
    to_pickle(obj, file_path)
    loaded = from_pickle(file_path)
    assert loaded["name"] == "foo"
    assert_array_equal(loaded["x"], obj["x"])


def test_from_pickle_with_legacy_numpy_core(tmp_wd):
    """Check that a pickle referencing numpy.core can be loaded.

    Pickles created with NumPy 1 reference the numpy.core package,
    which was renamed to numpy._core in NumPy 2.
    """
    data = array([1.0, 2.0])
    # Protocol 0 stores the module names as newline-terminated text,
    # so they can be rewritten without invalidating the pickle stream.
    legacy_bytes = pickle.dumps(data, protocol=0).replace(b"numpy._core", b"numpy.core")
    assert b"numpy.core" in legacy_bytes
    file_path = Path("legacy.pkl")
    file_path.write_bytes(legacy_bytes)
    assert_array_equal(from_pickle(file_path), data)
