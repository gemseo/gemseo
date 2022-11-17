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
"""Pytest helpers."""
from __future__ import annotations

import contextlib
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.testing.decorators
import pytest
from packaging import version

from gemseo.core.factory import Factory
from gemseo.utils.python_compatibility import Final

__ABSTRACTMETHODS__: Final[str] = "__abstractmethods__"


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


@pytest.fixture(scope="module")
def module_tmp_wd(tmp_path_factory):
    """Generator to move into a temporary subdirectory forth and back.

    Return the path to the temporary directory.
    """
    prev_cwd = Path.cwd()
    tmp_path = tmp_path_factory.getbasetemp()
    os.chdir(str(tmp_path))
    try:
        yield tmp_path
    finally:
        os.chdir(str(prev_cwd))


# Fixture to move into a temporary directory.
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
    if version.parse(matplotlib.__version__) < version.parse("3.6.0"):
        plt.close("all")


@pytest.fixture
def reset_factory():
    """Reset the factory cache."""
    Factory.cache_clear()
    yield
    Factory.cache_clear()


# Backup before we monkey patch.
original_image_directories = matplotlib.testing.decorators._image_directories


if "GEMSEO_KEEP_IMAGE_COMPARISONS" not in os.environ:
    # Context manager to change the current working directory to a temporary one.
    __ctx_tmp_wd = contextlib.contextmanager(__tmp_wd)

    def _image_directories(func):
        """Create the result_images directory in a temporary parent directory."""
        with __ctx_tmp_wd(tempfile.mkdtemp()):
            baseline_dir, result_dir = original_image_directories(func)
        return baseline_dir, result_dir

    matplotlib.testing.decorators._image_directories = _image_directories


@contextlib.contextmanager
def concretize_classes(*classes: type) -> None:
    """Context manager forcing classes to be concrete.

    Args:
        *classes: The classes.
    """
    classes_to___abstractmethods__ = {}
    for cls in classes:
        if hasattr(cls, __ABSTRACTMETHODS__):
            classes_to___abstractmethods__[cls] = cls.__abstractmethods__
            del cls.__abstractmethods__

    try:
        yield
    finally:
        for cls, __abstractmethods__ in classes_to___abstractmethods__.items():
            cls.__abstractmethods__ = __abstractmethods__
