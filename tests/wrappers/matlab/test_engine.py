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

from pathlib import Path

import numpy as np
import pytest

# skip if matlab API is not found
matlab = pytest.importorskip("matlab")

from gemseo.wrappers.matlab.engine import get_matlab_engine  # noqa: E402

from .matlab_files import MATLAB_FILES_DIR_PATH  # noqa: E402


@pytest.fixture
def matlab_engine():
    """Return a brand new matlab engine with clean cache."""
    get_matlab_engine.cache_clear()
    matlab = get_matlab_engine()
    matlab.add_path(MATLAB_FILES_DIR_PATH, True)
    yield matlab
    get_matlab_engine.cache_clear()


def test_singleton():
    """Test that the engine is effectively a singleton."""
    assert get_matlab_engine() is get_matlab_engine()


def test_start_engine(matlab_engine):
    """Test that matlab engine starts correctly."""
    assert matlab_engine.is_closed is False


def execute_function(matlab_engine):
    matlab_engine.execute_function("assignin", "base", "dummy_var", 2, nargout=0)


@pytest.mark.parametrize(
    "setup, name, is_found, type_",
    (
        (None, "sin", True, "MATLAB-built-in-function"),
        (None, MATLAB_FILES_DIR_PATH, True, "folder"),
        (None, "dummy_test.m", True, "file"),
        (None, "dummy_test.p", True, "P-code-file"),
        (None, "non_existing_file", False, None),
        (execute_function, "dummy_var", True, "variable"),
    ),
)
def test_exist(setup, name, is_found, type_):
    """Test the existence of a built-in functions."""
    matlab_engine = get_matlab_engine()
    matlab_engine.add_path(MATLAB_FILES_DIR_PATH, True)
    if setup is not None:
        setup(matlab_engine)
    assert matlab_engine.exist(name) == (is_found, type_)


# TODO: add test on MEX-file existence
# TODO: add test on Simulink-file existence
# TODO: add test on class existence


def test_add_toolbox(matlab_engine):
    """Test an add of toolbox."""
    matlab_engine.add_toolbox("signal_toolbox")
    assert "signal_toolbox" in matlab_engine.get_toolboxes()


def test_remove_toolbox(matlab_engine):
    """Test a remove of toolbox."""
    matlab_engine.add_toolbox("signal_toolbox")
    matlab_engine.remove_toolbox("signal_toolbox")
    assert "signal_toolbox" not in matlab_engine.get_toolboxes()


def test_add_path_sub_folder(matlab_engine):
    """Test that sub-folder is correctly added in path."""
    assert str(MATLAB_FILES_DIR_PATH) in matlab_engine.paths
    assert str(MATLAB_FILES_DIR_PATH / "matlab_files_bis_test") in matlab_engine.paths


def test_execute_error(matlab_engine):
    """Test that execution raise an error if something wrong."""
    with pytest.raises(matlab.engine.MatlabExecutionError):
        matlab_engine.execute_function("dummy_test")


@pytest.mark.parametrize("x, res", [(2, 4), (3, 9), (4, 16)])
def test_execute(x, res, matlab_engine):
    """Test the execution of a matlab function."""
    assert res == matlab_engine.execute_function("dummy_test", x, nargout=1)


def test_parallel(matlab_engine):
    """If parallel exist, test that it starts and stop correctly."""
    matlab_engine.start_parallel_computing(4)
    assert matlab_engine.is_parallel is matlab_engine.exist("parpool")[0]

    matlab_engine.end_parallel_computing()
    assert matlab_engine.is_parallel is False


def test_launch_script_dir(matlab_engine):
    """Test script launching and make sure that the path of matlab workspace is the
    directory that contains the script."""
    matlab_engine.execute_script("dummy_script_2")
    path = matlab_engine.execute_function("pwd", nargout=1)
    assert (
        Path(path).absolute()
        == (MATLAB_FILES_DIR_PATH / "matlab_files_bis_test").absolute()
    )


def test_get_variable(matlab_engine):
    """Test getting variable from the matlab workspace to Python."""
    matlab_engine.execute_script("dummy_script_2")
    assert matlab_engine.get_variable("a") == pytest.approx(2)
    assert matlab_engine.get_variable("b") == pytest.approx(3)
    assert matlab_engine.get_variable("c") == pytest.approx(np.array([10, 20]))
    assert matlab_engine.get_variable("d") == np.array(["test_string"])


def test_get_variable_error(matlab_engine):
    """Test that an error is raised if we try to get a variable that does not exist in
    matlab workspace."""
    with pytest.raises(
        ValueError,
        match="The variable toto does not exist in the current matlab workspace.",
    ):
        matlab_engine.get_variable("toto")


def test_quit(matlab_engine):
    """Test that engine is correctly quit."""
    matlab_engine.close_session()
    assert matlab_engine.is_closed is True
