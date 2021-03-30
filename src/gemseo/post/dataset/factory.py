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
Dataset plot factory
====================

The module :mod:`~gemseo.post.dataset.factory` contains
the :class:`DatasetPlotFactory` class which is a factory
to instantiate a :class:`.DatasetPlot` from its class name.
The class can be internal to |g| or located in an external module whose path
is provided to the constructor. It also provides a list of available cache
types and allows you to test if a cache type is available.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.post.dataset.dataset_plot import DatasetPlot

standard_library.install_aliases()


from gemseo import LOGGER


class DatasetPlotFactory(object):
    """This factory instantiates a :class:`.DatasetPlot` from its class name.
    The class can be internal to |g| or located in an external module
    whose path is provided to the constructor.
    """

    def __init__(self):
        """
        Initializes the factory: scans the directories to search for
        subclasses of DatasetPlot.
        Searches in "GEMSEO_PATH" and gemseo.mlearning.p_datasets
        """
        self.factory = Factory(DatasetPlot, ("gemseo.post.dataset",))

    def create(self, plot_name, **options):
        """
        Create a plot for dataset

        :param str plot_name: name of the plot for dataset
            (its classname)
        :param options: additional options specific
        :return: dataset plot
        """
        return self.factory.create(plot_name, **options)

    @property
    def plots(self):
        """
        Lists the available plots for datasets.

        :returns: the list of plots for datasets.
        :rtype: list(str)
        """
        return self.factory.classes

    def is_available(self, plot_name):
        """
        Checks the availability of a plot for dataset.

        :param str plot_name:  name of the plot for dataset
            (its class name).
        :returns: True if the plot for dataset is available.
        :rtype: bool
        """
        return self.factory.is_available(plot_name)
