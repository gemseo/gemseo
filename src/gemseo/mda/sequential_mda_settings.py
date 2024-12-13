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
"""Settings for MDASequential ."""

from __future__ import annotations

from collections.abc import Sequence  # Noqa: TC003
from typing import ClassVar  # Noqa: TC003

from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mda.composed_mda_settings import ComposedMDASettings


class MDASequential_Settings(BaseMDASettings, ComposedMDASettings):  # noqa: N801
    """The settings for :class:`.MDASequential`."""

    _settings_names_to_be_cascaded: ClassVar[Sequence[str]] = ["log_convergence"]
    """The settings that must be cascaded to the inner MDAs."""
