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
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from syrupy.matchers import path_type

from gemseo import configuration
from gemseo.core.base_factory import BaseFactory
from gemseo.util.platform import PLATFORM_IS_WINDOWS

# Rewrite asserts in helpers so syrupy's snapshot diff is shown on mismatch.
pytest.register_assert_rewrite("gemseo.util.testing.helper")
pytest.register_assert_rewrite("gemseo.util.testing.package_import")

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
    if request.node.get_closest_marker("skip_under_windows") and PLATFORM_IS_WINDOWS:
        pytest.skip("skipped on windows")


@pytest.fixture
def reset_factory():
    """Reset the factory cache."""
    BaseFactory.clear_cache()
    yield
    BaseFactory.clear_cache()


@pytest.fixture
def snapshot_allclose(snapshot):
    """Return a factory building a ``snapshot`` comparing floats with tolerances.

    Use in place of ``snapshot`` for snapshots containing Python-level floats
    whose last bits drift across OSes or Python versions
    (e.g. plotly figures decoded from ``json.loads(fig.to_json())``).
    Numpy arrays serialized by plotly as ``{"dtype": "f8", "bdata": ...}``
    are not affected: they are still compared byte-for-byte.

    Floats are canonicalized following ``numpy.allclose`` semantics
    (``|x - y| <= atol + rtol * |y|``):
    values with ``|x| <= atol`` collapse to ``0.0``,
    and the remaining values are rounded to a number of significant digits
    derived from ``rtol``.
    Two floats within tolerance therefore canonicalize to the same value
    and compare equal in the snapshot.

    The arguments of the fixture are
    the relative tolerance `rtol`
    and the absolute tolerance `atol`.
    """

    def make(rtol: float = 1e-9, atol: float = 0.0):
        sig_digits = max(1, -math.floor(math.log10(rtol)))

        def quantize(value: float, _) -> float:  # pragma: no cover
            if not math.isfinite(value) or value == 0.0:
                return value
            if abs(value) <= atol:
                return 0.0
            order = math.floor(math.log10(abs(value)))
            return round(value, sig_digits - order - 1)

        return snapshot(matcher=path_type(types=(float,), replacer=quantize))

    return make


@pytest.fixture
def enable_function_statistics() -> Generator[None, None, None]:
    """Enable functions statistics temporary."""
    configuration.enable_function_statistics = True
    yield
    configuration.enable_function_statistics = False


@pytest.fixture
def enable_discipline_status() -> Generator[None, None, None]:
    """Enable discipline status temporary."""
    configuration.enable_discipline_status = True
    yield
    configuration.enable_discipline_status = False


@pytest.fixture
def enable_discipline_statistics() -> Generator[None, None, None]:
    """Enable discipline statistics temporary.."""
    configuration.enable_discipline_statistics = True
    yield
    configuration.enable_discipline_statistics = False
