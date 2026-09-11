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

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from strenum import StrEnum

from gemseo.dataset.factory import dataset_factory
from gemseo.util.package_import import install_lazy_reexport
from gemseo.util.string import convert_camel_case_to_screaming_snake_case

if TYPE_CHECKING:
    from collections.abc import Mapping

    # static visibility for mypy / IDEs
    from gemseo.dataset.dataset import Dataset  # noqa: F401
    from gemseo.dataset.io_dataset import IODataset  # noqa: F401
    from gemseo.dataset.optimization_dataset import OptimizationDataset  # noqa: F401

DatasetClassName = StrEnum(
    "DatasetClassName",
    {
        convert_camel_case_to_screaming_snake_case(name): name
        for name in dataset_factory.class_names
    },
)
"""The enumeration of [Dataset][gemseo.dataset.dataset.Dataset] class names."""

# Exported name -> location (lazy-loaded on attribute access).
_name_to_location: Final[Mapping[str, str]] = MappingProxyType({
    "Dataset": "dataset",
    "IODataset": "io_dataset",
    "OptimizationDataset": "optimization_dataset",
})

install_lazy_reexport(
    globals(), _name_to_location, extra_all=("DatasetClassName", "dataset_factory")
)
