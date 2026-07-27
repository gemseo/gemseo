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

from pathlib import Path

from pydantic import BaseModel
from pydantic import Field

from gemseo.utils.pydantic import update_field


class BaseBackupSettings(BaseModel, extra="forbid", validate_default=True):
    """The base settings of the backup file to store the evaluations."""

    file_path: Path = Field(
        default="backup.h5",
        description="""The backup file path. In the context of the DirectoryManager,
        only a file name is necessary as the path is handled by the
        DirectoryManager.""",
    )

    at_each_iteration: bool = Field(
        default=False,
        description="Whether the backup file is updated at every"
        " iteration of the optimization.",
    )

    at_each_function_call: bool = Field(
        default=True,
        description="Whether the backup is updated at every function call.",
    )

    plot: bool = Field(
        default=False,
        description="""Whether to plot the optimization history view at each iteration.

      The plots will be generated only after the first two iterations.
      """,
    )


class BackupSettings(BaseBackupSettings):
    """The full settings of the backup file to store the evaluations."""

    erase: bool = Field(
        default=False, description="Whether the backup file is erased before the run."
    )

    load: bool = Field(
        default=False,
        description="Whether the backup file is loaded before run,"
        " useful after a crash.",
    )


update_field(BackupSettings, "file_path", description="The backup file path.")
