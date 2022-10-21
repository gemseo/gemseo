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

import pytest

# skip if matlab API is not found
pytest.importorskip("matlab")

from gemseo.wrappers.matlab.engine import get_matlab_engine  # noqa: E402
from gemseo.wrappers.matlab.license_manager import LicenseManager  # noqa: E402


@pytest.fixture
def matlab_engine():
    """Return a brand new matlab engine with clean cache."""
    get_matlab_engine.cache_clear()
    yield get_matlab_engine()
    get_matlab_engine.cache_clear()


@pytest.mark.parametrize(
    "toolbox", (LicenseManager.CURVE_FIT_TOOL, LicenseManager.DISTRIB_COMP_TOOL)
)
def test_check_when_adding_toolbox_curve_fit(matlab_engine, toolbox):
    """Check that the license is checked when curve fit toolbox is added in engine."""
    matlab_engine.add_toolbox(toolbox)
    manager = LicenseManager(matlab_engine)
    manager.check_licenses()
    assert manager.licenses == matlab_engine.get_toolboxes()
