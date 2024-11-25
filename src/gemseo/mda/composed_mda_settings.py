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
"""Settings for compose MDAs."""

from __future__ import annotations

from collections.abc import Sequence  # Noqa: TC003
from typing import TYPE_CHECKING  # Noqa: TC003
from typing import ClassVar  # Noqa: TC003

from pydantic import model_validator

if TYPE_CHECKING:
    from typing_extensions import Self

    from gemseo.mda.base_mda import BaseMDA


class ComposedMDASettings:
    """The settings for composed MDAs."""

    _sub_mdas: Sequence[BaseMDA] = []
    """The sub-MDAs."""

    _settings_names_to_be_cascaded: ClassVar[Sequence[str]] = ()
    """The settings that must be cascaded to the inner MDAs."""

    @model_validator(mode="after")
    def __cascade_settings(self) -> Self:
        """Cascade settings to the sub-MDAs."""
        for sub_mda in self._sub_mdas:
            for setting in self._settings_names_to_be_cascaded:
                value = self.__getattribute__(setting)
                sub_mda.settings.__setattr__(setting, value)
        return self
