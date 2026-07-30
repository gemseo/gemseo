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

from typing import TYPE_CHECKING
from typing import Final

from strenum import StrEnum

from gemseo.dataset.factory import DATASET_FACTORY
from gemseo.util.package_import import install_lazy_reexport
from gemseo.util.string import convert_camel_case_to_screaming_snake_case

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.dataset.dataset import Dataset  # noqa: F401
    from gemseo.dataset.io_dataset import IODataset  # noqa: F401
    from gemseo.dataset.optimization_dataset import OptimizationDataset  # noqa: F401

DatasetClassName = StrEnum(
    "DatasetClassName",
    {
        convert_camel_case_to_screaming_snake_case(name): name
        for name in DATASET_FACTORY.class_names
    },
)
"""The enumeration of [Dataset][gemseo.dataset.dataset.Dataset] class names."""

# Exported name -> location (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "Dataset": "dataset",
    "IODataset": "io_dataset",
    "OptimizationDataset": "optimization_dataset",
}

install_lazy_reexport(
    globals(), _NAME_TO_LOCATION, extra_all=("DatasetClassName", "DATASET_FACTORY")
)
