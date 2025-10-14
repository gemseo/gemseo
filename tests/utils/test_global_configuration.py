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

from logging import INFO
from logging import NullHandler
from logging import getLogger

from gemseo.utils.global_configuration import GlobalConfiguration


def test_default():
    """Check the GlobalConfiguration."""
    assert GlobalConfiguration.model_fields.keys() == {
        "check_desvars_bounds",
        "enable_discipline_cache",
        "enable_discipline_statistics",
        "enable_discipline_status",
        "enable_function_statistics",
        "enable_parallel_execution",
        "enable_progress_bar",
        "logging",
        "fast",
        "validate_input_data",
        "validate_output_data",
    }
    settings = GlobalConfiguration()
    assert settings.check_desvars_bounds
    assert settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert not settings.fast
    logging = settings.logging
    assert logging.date_format == "%H:%M:%S"
    assert logging.enable
    assert logging.file_path == ""
    assert logging.file_mode == "a"
    assert logging.level == INFO
    assert logging.message_format == "%(levelname)8s - %(asctime)s: %(message)s"
    assert settings.validate_input_data
    assert settings.validate_output_data


def test_fast():
    """Check the GlobalConfiguration."""
    settings = GlobalConfiguration(fast=True)
    assert not settings.check_desvars_bounds
    assert not settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert not settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert settings.fast
    logging = settings.logging
    assert logging.date_format == "%H:%M:%S"
    assert settings.logging.enable
    assert logging.file_path == ""
    assert logging.file_mode == "a"
    assert logging.level == INFO
    assert logging.message_format == "%(levelname)8s - %(asctime)s: %(message)s"
    assert not settings.validate_input_data
    assert not settings.validate_output_data

    settings.fast = False
    assert settings.check_desvars_bounds
    assert settings.enable_discipline_cache
    assert not settings.enable_discipline_statistics
    assert not settings.enable_discipline_status
    assert not settings.enable_function_statistics
    assert settings.enable_parallel_execution
    assert settings.enable_progress_bar
    assert not settings.fast
    logging = settings.logging
    assert logging.date_format == "%H:%M:%S"
    assert settings.logging.enable
    assert logging.file_path == ""
    assert logging.file_mode == "a"
    assert logging.level == INFO
    assert logging.message_format == "%(levelname)8s - %(asctime)s: %(message)s"
    assert settings.validate_input_data
    assert settings.validate_output_data


def test_environment_variable(monkeypatch):
    """Check the use of environment variables."""
    logger = getLogger("NewLogger")
    logger.addHandler(NullHandler())
    assert GlobalConfiguration().enable_progress_bar
    monkeypatch.setenv("GEMSEO_ENABLE_PROGRESS_BAR", "False")
    monkeypatch.setenv("GEMSEO_LOGGING_ENABLE", "False")
    configuration = GlobalConfiguration()
    assert not configuration.enable_progress_bar
    assert not configuration.logging.enable
    assert not getLogger("gemseo").handlers
    assert getLogger("").handlers
    assert getLogger("NewLogger").handlers


def test_environment_variable_env_file(monkeypatch, tmp_wd):
    """Check the use of environment variables from a .env file."""
    assert GlobalConfiguration().enable_progress_bar
    with (tmp_wd / ".env").open("w") as f:
        f.write("GEMSEO_ENABLE_PROGRESS_BAR=False\n")
        f.write("GEMSEO_LOGGING_ENABLE=True")
    configuration = GlobalConfiguration()
    assert not configuration.enable_progress_bar
    assert configuration.logging.enable


def test_extra_config_in_env_file(monkeypatch, tmp_wd):
    """Check that extra config in .env file doesn't raise validation errors."""
    with (tmp_wd / ".env").open("w") as f:
        f.write("SOME_OTHER_APP_CONFIG=value\n")

    # This should not raise a ValidationError about extra_forbidden
    configuration = GlobalConfiguration()
    assert not hasattr(configuration, "some_other_app_config")
