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
"""Factory of classes for displaying data in progress bars."""

from __future__ import annotations

from typing import ClassVar
from typing import Final

from strenum import StrEnum

from gemseo.core.algorithm.progress_bar_data.base import BaseProgressBarData
from gemseo.core.base_factory import BaseFactory
from gemseo.util.string import convert_camel_case_to_screaming_snake_case


class ProgressBarDataFactory(BaseFactory):
    """The factory for `BaseProgressBarData` objects."""

    _class: ClassVar[type[BaseProgressBarData]] = BaseProgressBarData
    _package_names: ClassVar[tuple[str, ...]] = (
        "gemseo.core.algorithm.progress_bar_data",
    )


progress_bar_data_factory: Final[ProgressBarDataFactory] = ProgressBarDataFactory()
"""The factory for `BaseProgressBarData` objects."""

ProgressBarDataName = StrEnum(
    "ProgressBarDataName",
    {
        convert_camel_case_to_screaming_snake_case(name): name
        for name in progress_bar_data_factory.class_names
    },
)
"""A name of a [BaseProgressBarData][gemseo.core.algorithm.progress_bar_data.base.BaseProgressBarData] subclass."""  # noqa: E501
