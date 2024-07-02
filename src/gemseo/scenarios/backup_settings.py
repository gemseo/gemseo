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
"""Backup settings."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclasses.dataclass
class BackupSettings:
    """The settings of the backup file to store the evaluations."""

    file_path: str | Path
    """The backup file path."""

    at_each_iteration: bool = False
    """Whether the backup file is updated at every iteration of the optimization."""

    at_each_function_call: bool = True
    """Whether the backup file is updated at every function call."""

    erase: bool = False
    """Whether the backup file is erased before the run."""

    load: bool = False
    """Whether the backup file is loaded before run.

    A backup file can be useful after a crash.
    """

    plot: bool = False
    """Whether to plot the optimization history view at each iteration.

    The plots will be generated only after the first two iterations.
    """
