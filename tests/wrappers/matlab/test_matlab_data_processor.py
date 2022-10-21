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

import os

import pytest
from numpy import array
from numpy import ndarray

# skip if matlab API is not found
matlab = pytest.importorskip("matlab")

from gemseo.wrappers.matlab.matlab_data_processor import (  # noqa: E402
    MatlabDataProcessor,
    array2double,
    convert_array_from_matlab,
    convert_array_to_matlab,
    double2array,
    load_matlab_file,
    save_matlab_file,
)

from .matlab_files import MATLAB_FILES_DIR_PATH  # noqa: E402


def test_pre_process_data():
    """Test that data are correctly pre-processed from dict of array to dict of
    matlab.double."""
    data_proc = MatlabDataProcessor()
    d = {"x": array([2]), "y": array([2j], dtype="complex")}
    res = data_proc.pre_process_data(d)
    assert isinstance(res["x"], matlab.double)
    assert res["x"][0] == pytest.approx(2)
    assert isinstance(res["y"], matlab.double)
    assert res["y"][0] == pytest.approx(2j)


def test_post_process_data():
    """Test that data are correctly post-processed from dict of matlab.double to dict of
    array."""
    data_proc = MatlabDataProcessor()
    d = {"y": matlab.double([2, 3]), "x": array([2j], dtype="complex")}
    res = data_proc.post_process_data(d)

    assert isinstance(res["x"], ndarray)
    assert res["x"][0] == pytest.approx(2j)
    assert isinstance(res["y"], ndarray)
    assert res["y"][0] == pytest.approx(2)
    assert res["y"][1] == pytest.approx(3)


def test_load_matlab_file():
    """Test matlab file loading."""
    res = load_matlab_file(MATLAB_FILES_DIR_PATH / "dummy_file.mat")
    assert array(res["a"]) == pytest.approx(2.0)
    assert array(res["b"]) == pytest.approx(3.0)
    assert array(res["c"]) == pytest.approx(4.0)
    assert array(res["d"]) == pytest.approx(array([[4.0, 5.0, 6.0]]))
    assert array(res["e"]) == pytest.approx(array([[10], [20], [30]]))
    assert array(res["f"]) == pytest.approx(array([[1, 2, 3], [4, 5, 6], [7, 8, 99]]))


def test_save_matlab_file():
    """Test matlab file saving."""
    name_file = "test_file.mat"
    test_dict = {
        "test1": array([3.0 + 1j, 2.0], dtype="complex"),
        "test2": array([3.0]),
        "test3": array([[3.0, 2.0], [10.0, 22.0]]),
    }
    save_matlab_file(test_dict, name_file)
    load_dat = load_matlab_file(name_file)

    assert array(load_dat["test1"][0]) == pytest.approx(test_dict["test1"])
    assert array(load_dat["test2"][0]) == pytest.approx(test_dict["test2"])
    assert array(load_dat["test3"]) == pytest.approx(test_dict["test3"])
    os.remove(name_file)


def test_array2double_1d():
    """Test that a 1d array is correctly converted to double."""
    a = array([3.0, 2.0])
    r = array2double(a)
    assert a[0] == pytest.approx(r[0])
    assert a[1] == pytest.approx(r[1])


def test_array2double_1d_complex():
    """Test that a 1d complex array is correctly converted to double."""
    a = array([3.0, 2.0 + 5.2j])
    r = array2double(a)
    assert a[0] == pytest.approx(r[0])
    assert a[1] == pytest.approx(r[1])


def test_array2double_2d():
    """Test that a 2d array is correctly converted to double."""
    a = array([[3.0, 2.0], [11, 22.0]])
    r = array2double(a)
    assert a[0][0] - r[0][0] == pytest.approx(0)
    assert a[0][1] - r[0][1] == pytest.approx(0)
    assert a[1][0] - r[1][0] == pytest.approx(0)
    assert a[1][1] - r[1][1] == pytest.approx(0)


def test_double2array_1d():
    """Test that matlab.double is correctly converted to ndarray."""
    a = matlab.double([2.0, 3.0])
    r = double2array(a)
    assert a[0][0] - r[0] == pytest.approx(0)
    assert a[0][1] - r[1] == pytest.approx(0)


def test_double2array_1d_complex():
    """Test that 1d complex matlab.double is correctly converted to ndarray."""
    a = matlab.double([2.0, 3.0 + 6.3j], is_complex=True)
    r = double2array(a)
    assert a[0][0] - r[0] == pytest.approx(0)
    assert a[0][1] - r[1] == pytest.approx(0)


def test_double2array_2d():
    """Test that 2d matlab.double is correctly converted to ndarray."""
    a = matlab.double([[3.0, 2.0], [11, 22.0]])
    r = double2array(a)
    assert a[0][0] - r[0][0] == pytest.approx(0)
    assert a[0][1] - r[0][1] == pytest.approx(0)
    assert a[1][0] - r[1][0] == pytest.approx(0)
    assert a[1][1] - r[1][1] == pytest.approx(0)


def test_matlab2gems():
    """Test that matlab dict is correctly converted to gems-like dict."""
    d = {
        "test1": matlab.double([3.0 + 1j, 2.0], is_complex=True),
        "test2": matlab.double([3.0]),
        "test3": matlab.double([[3.0, 2.0], [10.0, 22.0]]),
    }
    r = convert_array_from_matlab(d)
    assert r["test1"][0] - d["test1"][0][0] == pytest.approx(0)
    assert r["test1"][1] - d["test1"][0][1] == pytest.approx(0)
    assert r["test2"][0] - d["test2"][0][0] == pytest.approx(0)
    assert sum(r["test3"][0] - d["test3"][0]) == pytest.approx(0)
    assert sum(r["test3"][1] - d["test3"][1]) == pytest.approx(0)


def test_gems2matlab():
    """Test that gems-like dict is correctly converted to matlab dict."""
    d = {
        "test1": array([3.0 + 1j, 2.0]),
        "test2": array([3.0]),
        "test3": array([[3.0, 2.0], [10.0, 22.0]]),
    }
    r = convert_array_to_matlab(d)
    assert sum(r["test1"] - d["test1"]) == pytest.approx(0)
    assert r["test2"] - d["test2"] == pytest.approx(0)
    assert sum(r["test3"][0] - d["test3"][0]) == pytest.approx(0)
    assert sum(r["test3"][1] - d["test3"][1]) == pytest.approx(0)
