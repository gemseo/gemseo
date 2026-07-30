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
"""Factory for instantiating directory manager processors."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from gemseo.util._directory_manager.processor.base import BaseDMProcessor
from gemseo.util._workflow_observer.base_processor_factory import BaseProcessorFactory


class DMProcessorFactory(BaseProcessorFactory[BaseDMProcessor]):
    """Factory to create a directory manager processor."""

    _CLASS: ClassVar[type[BaseDMProcessor]] = BaseDMProcessor
    _PACKAGE_NAMES: ClassVar[tuple[str, ...]] = ("gemseo.util._directory_manager",)


DM_PROCESSOR_FACTORY: Final[DMProcessorFactory] = DMProcessorFactory()
"""The factory for `BaseDMProcessor` objects."""
