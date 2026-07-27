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
"""Tests for the backup settings."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from gemseo.scenarios.backup_settings import BackupSettings
from gemseo.scenarios.backup_settings import BaseBackupSettings
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize("settings_class", [BaseBackupSettings, BackupSettings])
def test_unknown_setting(settings_class, snapshot):
    """Verify that an unknown setting name raises instead of being ignored."""
    with assert_exception(ValidationError, snapshot):
        settings_class(file_path="backup.h5", at_each_iterations=True)
