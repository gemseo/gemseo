# -*- coding: utf-8 -*-
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

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Dataset factory
===============

This module contains a factory
to instantiate a :class:`.Dataset` from its class name.
The class can be internal to |g| or located in an external module whose path
is provided to the constructor. It also provides a list of available cache
types and allows you to test if a cache type is available.
"""
from __future__ import division, unicode_literals

import logging

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory

LOGGER = logging.getLogger(__name__)


class DatasetFactory(object):
    """This factory instantiates a :class:`.Dataset` from its class name.

    The class can be internal to |g| or located in an external module whose path is
    provided to the constructor.
    """

    def __init__(self):
        """Initializes the factory: scans the directories to search for subclasses of
        Dataset.

        Searches in "GEMSEO_PATH" and gemseo.mlearning.p_datasets
        """
        self.factory = Factory(Dataset, ("gemseo.problems.dataset",))

    def create(self, dataset, **options):
        """Create a dataset.

        :param str dataset: name of the dataset (its classname).
        :param options: additional options specific
        :return: dataset
        :rtype: Dataset
        """
        return self.factory.create(dataset, **options)

    @property
    def datasets(self):
        """Lists the available datasets.

        :returns: the list of datasets.
        :rtype: list(str)
        """
        return self.factory.classes

    def is_available(self, dataset):
        """Checks the availability of a dataset.

        :param str dataset:  name of the dataset (its class name).
        :returns: True if the dataset is available.
        :rtype: bool
        """
        return self.factory.is_available(dataset)
