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
"""Settings for the algorithms."""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel
from pydantic import ConfigDict


class BaseAlgorithmSettings(BaseModel):
    """The settings common to all the algorithms."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    _TARGET_CLASS_NAME: ClassVar[str]
    """The name of the algorithm class."""
