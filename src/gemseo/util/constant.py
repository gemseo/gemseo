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
"""Constants."""

from __future__ import annotations

import logging
import sys
from multiprocessing import cpu_count
from types import MappingProxyType
from typing import Any
from typing import Final

from numpy import iinfo
from numpy import int32

n_cpus: Final[int] = cpu_count()
"""The number of CPUs in the system."""

read_only_empty_dict: Final[MappingProxyType[Any, Any]] = MappingProxyType({})
"""A read-only empty dictionary."""

settings: Final[str] = "settings"
"""The name of the argument to pass a Pydantic model."""

infinite_int: Final[int] = iinfo(int32).max
"""An integer standing for infinity, i.e. the largest 32-bit integer."""

epsilon: Final[float] = sys.float_info.epsilon
"""The machine epsilon."""

# Default settings for GlobalConfiguration
_check_desvars_bounds: Final[bool] = True
_enable_discipline_cache: Final[bool] = True
_enable_discipline_statistics: Final[bool] = False
_enable_discipline_status: Final[bool] = False
_enable_function_statistics: Final[bool] = False
_enable_parallel_execution: Final[bool] = True
_enable_progress_bar: Final[bool] = True
_validate_input_data: Final[bool] = True
_validate_output_data: Final[bool] = True
_logging_date_format: Final[str] = "%H:%M:%S"
_logging_message_format: Final[str] = "%(levelname)8s - %(asctime)s: %(message)s"
_logging_level: Final[int] = logging.INFO
_logging_file_mode: Final[str] = "a"
_logging_file_path: Final[str] = ""
