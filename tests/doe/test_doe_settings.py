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

from gemseo.doe.core.base_doe_settings import BaseDOESettings
from gemseo.util.testing.helper import assert_exception


def test_doe_settings_vectorize_in_parallel(snapshot):
    """Check the error when setting vectorize in parallel mode."""
    with assert_exception(NotImplementedError, snapshot):
        BaseDOESettings(vectorize=True, n_processes=2)


def preprocessor(index: int) -> None: ...


def test_doe_settings_preprocessors_and_vectorization(snapshot):
    """Check the error when combining preprocessors and vectorization."""
    with assert_exception(NotImplementedError, snapshot):
        BaseDOESettings(vectorize=True, preprocessors=[preprocessor])
