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

from gemseo.algos.doe.base_doe_settings import BaseDOESettings


def test_doe_settings_wait_time_between_samples(caplog):
    """Check the log when setting wait_time_between_samples in serial mode."""
    BaseDOESettings(wait_time_between_samples=0.2)
    assert caplog.record_tuples == [
        (
            "gemseo.algos.doe.base_doe_settings",
            30,
            "The option 'wait_time_between_samples' is ignored "
            "when the option 'n_processes' is 1 (serial mode).",
        )
    ]


def test_doe_settings_vectorize_in_parallel():
    """Check the error when setting vectorize in parallel mode."""
    with pytest.raises(
        NotImplementedError,
        match=re.escape("Vectorization in parallel is not supported."),
    ):
        BaseDOESettings(vectorize=True, n_processes=2)


def preprocessor(index: int) -> None: ...


def test_doe_settings_preprocessors_and_vectorization():
    """Check the error when combining preprocessors and vectorization."""
    with pytest.raises(
        NotImplementedError,
        match=re.escape("Combining preprocessors and vectorization is not supported."),
    ):
        BaseDOESettings(vectorize=True, preprocessors=[preprocessor])
