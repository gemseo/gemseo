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
from multiprocessing import cpu_count
from types import MappingProxyType
from typing import Any
from typing import Final

from numpy import iinfo
from numpy import int32

N_CPUS: Final[int] = cpu_count()
"""The number of CPUs in the system."""

READ_ONLY_EMPTY_DICT: Final[MappingProxyType[Any, Any]] = MappingProxyType({})
"""A read-only empty dictionary."""

SETTINGS: Final[str] = "settings"
"""The name of the argument to pass a Pydantic model."""

C_LONG_MAX: Final[int] = iinfo(int32).max
"""The largest 32-bit integer."""

# Default settings for GlobalConfiguration
_CHECK_DESVARS_BOUNDS: Final[bool] = True
_ENABLE_DISCIPLINE_CACHE: Final[bool] = True
_ENABLE_DISCIPLINE_STATISTICS: Final[bool] = False
_ENABLE_DISCIPLINE_STATUS: Final[bool] = False
_ENABLE_FUNCTION_STATISTICS: Final[bool] = False
_ENABLE_PARALLEL_EXECUTION: Final[bool] = True
_ENABLE_PROGRESS_BAR: Final[bool] = True
_VALIDATE_INPUT_DATA: Final[bool] = True
_VALIDATE_OUTPUT_DATA: Final[bool] = True
_LOGGING_DATE_FORMAT: Final[str] = "%H:%M:%S"
_LOGGING_MESSAGE_FORMAT: Final[str] = "%(levelname)8s - %(asctime)s: %(message)s"
_LOGGING_LEVEL: Final[int] = logging.INFO
_LOGGING_FILE_MODE: Final[str] = "a"
_LOGGING_FILE_PATH: Final[str] = ""
