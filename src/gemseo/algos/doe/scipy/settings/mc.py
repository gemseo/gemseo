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
"""Settings for the Monte Carlo DOE from the SciPy library."""

from __future__ import annotations

from gemseo.algos.doe.scipy.settings.base_scipy_doe_settings import BaseSciPyDOESettings


class MC_Settings(BaseSciPyDOESettings):  # noqa: N801
    """The settings for the Monte Carlo DOE from the SciPy library."""

    _TARGET_CLASS_NAME = "MC"
