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
"""Compatibility with the data type values of the past releases.

The value of a [DataType][gemseo.enum.DesignVariableType] is stored in the
files a design space is written to, so a file written by a past release refers to
a data type by the value that release used. Such a value keeps being accepted,
whatever the release that wrote the file, so that these files remain readable.
"""

from __future__ import annotations

from gemseo.enum._data_type import legacy_data_type_values as legacy_data_type_values
from gemseo.enum._data_type import (
    warn_legacy_data_type_value as warn_legacy_data_type_value,
)
