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

from gemseo.utils.configuration import Fast_GEMSEO_Settings
from gemseo.utils.configuration import GEMSEO_Settings


def test_default():
    """Check the GEMSEO_Settings."""
    settings = GEMSEO_Settings()
    assert settings.model_fields.keys() == {
        "check_desvars_bounds",
        "enable_discipline_cache",
        "enable_discipline_statistics",
        "enable_discipline_status",
        "enable_function_statistics",
        "enable_parallel_execution",
        "enable_progress_bar",
        "validate_input_data",
        "validate_output_data",
    }
    assert settings.check_desvars_bounds
    assert settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert settings.validate_input_data
    assert settings.validate_output_data


def test_fast():
    """Check the Fast_GEMSEO_Settings."""
    settings = Fast_GEMSEO_Settings()
    assert settings.model_fields.keys() == {
        "check_desvars_bounds",
        "enable_discipline_cache",
        "enable_discipline_statistics",
        "enable_discipline_status",
        "enable_function_statistics",
        "enable_parallel_execution",
        "enable_progress_bar",
        "validate_input_data",
        "validate_output_data",
    }
    assert not settings.check_desvars_bounds
    assert not settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert not settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert not settings.validate_input_data
    assert not settings.validate_output_data
