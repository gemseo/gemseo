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
"""The datasets."""

from __future__ import annotations

from strenum import StrEnum

from gemseo.datasets.factory import DatasetFactory
from gemseo.utils.string_tools import convert_camel_case_to_screaming_snake_case

DatasetClassName = StrEnum(
    "DatasetClassName",
    {
        convert_camel_case_to_screaming_snake_case(name): name
        for name in DatasetFactory().class_names
    },
)
"""The enumeration of [Dataset][gemseo.datasets.dataset.Dataset] class names."""
