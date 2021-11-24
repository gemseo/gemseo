# -*- coding: utf-8 -*-
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

"""Test helpers."""
import contextlib
import os
import sys
import tempfile

import matplotlib.pyplot as plt
import matplotlib.testing.decorators
import pytest

from gemseo.core.factory import Factory
from gemseo.utils.py23_compat import PY2, Path


def __tmp_wd(tmp_path):
    """Generator to move into a temporary directory forth and back.

    Return the path to the temporary directory.
    """
    prev_cwd = Path.cwd()
    os.chdir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        os.chdir(str(prev_cwd))


# Fixture to more into a temporary directory.
tmp_wd = pytest.fixture()(__tmp_wd)


def pytest_sessionstart(session):
    """Bypass console pollution from fortran code."""
    # Prevent fortran code (like lbfgs from scipy) from writing to the console
    # by setting its stdout and stderr units to the standard file descriptors.
    # As a side effect, the files fort.{0,6} are created with the content of
    # the console outputs.
    os.environ["GFORTRAN_STDOUT_UNIT"] = "1"
    os.environ["GFORTRAN_STDERR_UNIT"] = "2"


def pytest_sessionfinish(session):
    """Remove file pollution from fortran code."""
    # take care of pytest_sessionstart side effects
    for file_ in Path(".").glob("fort.*"):
        file_.unlink()


@pytest.fixture(autouse=True)
def skip_under_windows(request):
    """Fixture that add a marker to skip under windows.

    Use it like a usual skip marker.
    """
    if request.node.get_closest_marker("skip_under_windows"):
        if sys.platform.startswith("win"):
            pytest.skip("skipped on windows")


@pytest.fixture
def baseline_images(request):
    """Return the baseline_images contents.

    Used when the compare_images decorator has indirect set.
    """
    return request.param


@pytest.fixture
def pyplot_close_all():
    """Fixture that prevents figures aggregation with matplotlib pyplot."""
    plt.close("all")


@pytest.fixture
def reset_factory():
    """Reset the factory cache."""
    Factory.cache_clear()
    yield
    Factory.cache_clear()


# Backup before we monkey patch.
if PY2:
    # workaround to get image_comparison working
    import matplotlib
    from matplotlib.testing.conftest import (  # noqa: F401
        mpl_image_comparison_parameters,
    )

    matplotlib._called_from_pytest = True

    # monkey patch the _image_directories function that expects the tests directory
    # layout of matplotlib
    from matplotlib import cbook
    from matplotlib.testing import decorators

    # keep an alias to the old function to be overridden
    _old_image_directories = decorators._image_directories

    def _new_image_directories(func):
        dir_paths = _old_image_directories(func)
        # remove the parents directories
        new_dir_paths = []
        for path in dir_paths:
            path_parts = list(Path(path).parts)
            path_parts.pop(-2)
            new_dir_paths += [str(Path(*path_parts))]
        # create the good dir and leave the bad one
        cbook.mkdirs(new_dir_paths[1])
        return new_dir_paths

    # hook in our function override
    decorators._image_directories = _new_image_directories
original_image_directories = matplotlib.testing.decorators._image_directories

# Context manager to change the current working directory to a temporary one.
__ctx_tmp_wd = contextlib.contextmanager(__tmp_wd)


def _image_directories(func):
    """Create the result_images directory in a temporary parent directory."""
    with __ctx_tmp_wd(tempfile.mkdtemp()):
        baseline_dir, result_dir = original_image_directories(func)
    return baseline_dir, result_dir


matplotlib.testing.decorators._image_directories = _image_directories


if PY2:
    import backports.unittest_mock

    backports.unittest_mock.install()
