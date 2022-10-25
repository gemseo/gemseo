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

import re

import pytest

# skip if matlab API is not found
pytest.importorskip("matlab")

from gemseo.wrappers.matlab.matlab_parser import MatlabParser  # noqa: E402

from .matlab_files import MATLAB_FILES_DIR_PATH  # noqa: E402

# TODO: change message because scripts not allowed


@pytest.mark.parametrize(
    "error, match_pattern, path",
    (
        (
            ValueError,
            "The given file {} should either be a matlab function or script.",
            MATLAB_FILES_DIR_PATH,
        ),
        (
            IOError,
            "The function directory for Matlab sources {} does not exists.",
            MATLAB_FILES_DIR_PATH / "not_existing_file.m",
        ),
        (
            ValueError,
            "The given file {} is encrypted and cannot be parsed.",
            MATLAB_FILES_DIR_PATH / "dummy_test.p",
        ),
        (
            ValueError,
            "The given file {} is not a matlab function.",
            MATLAB_FILES_DIR_PATH / "dummy_script.m",
        ),
        (
            ValueError,
            "The given file {} should either be a matlab function or script.",
            MATLAB_FILES_DIR_PATH / "dummy_file.mat",
        ),
        (
            NameError,
            "Function name dummy_test_no_name does not match with file name .",
            MATLAB_FILES_DIR_PATH / "dummy_test_no_name.m",
        ),
        (
            NameError,
            "Function name dummy_test_wrong_name does not match with file name dummy_test.",
            MATLAB_FILES_DIR_PATH / "dummy_test_wrong_name.m",
        ),
        (
            NameError,
            "Matlab function has no name.",
            MATLAB_FILES_DIR_PATH / "dummy_test_no_output.m",
        ),
        (
            NameError,
            "Matlab function has no name.",
            MATLAB_FILES_DIR_PATH / "dummy_test_no_input.m",
        ),
        (
            ValueError,
            "The given file {} is not a matlab function.",
            MATLAB_FILES_DIR_PATH / "not_a_matlab_function.m",
        ),
    ),
)
def test_errors(error, match_pattern, path):
    """Test that exception is raised if file is a directory."""
    # Path on windows that may have regex keywords.
    match = re.escape(match_pattern.format(path))
    with pytest.raises(error, match=match):
        MatlabParser(path)


def test_check_path_good_from_relative_path():
    """Test that file is found when given relative path."""
    parser = MatlabParser(MATLAB_FILES_DIR_PATH / "dummy_test.m")
    assert parser.function_name == "dummy_test"


# TODO: refactor, bad match --> giving a function with no output
#   return the no name exception


# TODO: refactor, bad match --> giving a function with no input
#   return no name exception


def test_scan_onef_complex():
    """Test that a more complex function is correctly parsed."""
    parser = MatlabParser(MATLAB_FILES_DIR_PATH / "dummy_complex_fct.m")
    assert parser.inputs == ["a", "b", "c", "d", "e", "f"]
    assert parser.outputs == ["x", "y", "z"]
