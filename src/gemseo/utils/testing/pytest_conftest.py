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
import faulthandler
import os
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.testing.decorators
import pytest
from packaging import version

from gemseo.core.base_factory import BaseFactory

if TYPE_CHECKING:
    from collections.abc import Generator


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


def pytest_sessionstart(session) -> None:
    """Bypass console pollution from fortran code."""
    # Prevent fortran code (like lbfgs from scipy) from writing to the console
    # by setting its stdout and stderr units to the standard file descriptors.
    # As a side effect, the files fort.{0,6} are created with the content of
    # the console outputs.
    os.environ["GFORTRAN_STDOUT_UNIT"] = "1"
    os.environ["GFORTRAN_STDERR_UNIT"] = "2"


def pytest_sessionfinish(session) -> None:
    """Remove file pollution from fortran code."""
    # take care of pytest_sessionstart side effects
    for file_ in Path().glob("fort.*"):
        with contextlib.suppress(PermissionError):
            # On windows the file may be opened and not released by another component.
            file_.unlink()


@pytest.fixture(autouse=True)
def skip_under_windows(request) -> None:
    """Fixture that add a marker to skip under windows.

    Use it like a usual skip marker.
    """
    if request.node.get_closest_marker(
        "skip_under_windows"
    ) and sys.platform.startswith("win"):
        pytest.skip("skipped on windows")


@pytest.fixture()
def baseline_images(request):
    """Return the baseline_images contents.

    Used when the compare_images decorator has indirect set.
    """
    return request.param


@pytest.fixture()
def pyplot_close_all() -> None:
    """Fixture that prevents figures aggregation with matplotlib pyplot."""
    if version.parse(matplotlib.__version__) < version.parse("3.6.0"):
        plt.close("all")


@pytest.fixture()
def reset_factory():
    """Reset the factory cache."""
    BaseFactory.clear_cache()
    yield
    BaseFactory.clear_cache()


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


# Fixtures to deal with the Excel disciplines.
# Check the presence of xlwings, and skip accordingly.
@pytest.fixture(scope="module")
def disable_fault_handler() -> Generator[None, None, None]:
    """Generator to temporarily disable the fault handler.

    Return a call to disable the fault handler.
    """
    if faulthandler.is_enabled():
        try:
            faulthandler.disable()
            yield
        finally:
            faulthandler.enable()


@pytest.fixture(scope="module")
def import_or_skip_xlwings() -> Any:
    """Fixture to skip a test when xlwings cannot be imported."""
    return pytest.importorskip("xlwings", reason="xlwings is not available")


@pytest.fixture(scope="module")
def is_xlwings_usable(import_or_skip_xlwings, disable_fault_handler) -> bool:
    """Check if xlwings is usable.

    Args:
        import_or_skip_xlwings: Fixture to import xlwings when available,
            otherwise skip the test.
        disable_fault_handler: Fixture to temporarily disable the fault handler.
    """
    xlwings = import_or_skip_xlwings

    try:
        # Launch xlwings from a context manager to ensure it closes immediately.
        # See https://docs.xlwings.org/en/stable/whatsnew.html#v0-24-3-jul-15-2021
        with xlwings.App(visible=False) as app:  # noqa: F841
            pass
    except:  # noqa: E722,B001
        return False
    else:
        return True


@pytest.fixture(scope="module")
def skip_if_xlwings_is_not_usable(is_xlwings_usable: bool) -> None:
    """Fixture to skip a test when xlwings is not usable."""
    if not is_xlwings_usable:
        pytest.skip("This test requires excel.")


@pytest.fixture(scope="module")
def skip_if_xlwings_is_usable(is_xlwings_usable: bool) -> None:
    """Fixture to skip a test when xlwings is usable."""
    if is_xlwings_usable:
        pytest.skip("This test is only required when excel is not available.")
