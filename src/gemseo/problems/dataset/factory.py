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
#
# Contributors:
# - Matthias De Lozzo
# - Jean-Christophe Giret
# - François Gallard
# - Antoine DECHAUME
"""
Dataset factory
===============

This module contains a factory
to instantiate a :class:`.Dataset` from its class name.
The class can be internal to |g| or located in an external module whose path
is provided to the constructor. It also provides a list of available cache
types and allows you to test if a cache type is available.
"""
from __future__ import annotations

import logging
from typing import Any

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory

LOGGER = logging.getLogger(__name__)


class DatasetFactory:
    """A factory for :class:`.Dataset`."""

    def __init__(self) -> None:
        self.factory = Factory(Dataset, ("gemseo.problems.dataset",))

    def create(self, dataset: str, **options: Any) -> Dataset:
        """Create a :class:`.Dataset`.

        Args:
            dataset: The name of the dataset (its classname).
            **options: The options of the dataset.

        Returns:
            A dataset.
        """
        return self.factory.create(dataset, **options)

    @property
    def datasets(self) -> list[str]:
        """The names of the available datasets."""
        return self.factory.classes

    def is_available(self, dataset: str) -> bool:
        """Check the availability of a dataset.

        Args:
            dataset: The name of the dataset (its class name).

        Returns:
            Whether the dataset is available.
        """
        return self.factory.is_available(dataset)
