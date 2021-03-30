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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Distribution factory
====================

This module contains a factory to instantiate a :class:`.Distribution`
from its class name.
The class can be internal to |g|
or located in an external module whose path is provided to the constructor.
It also provides a list of available distributions types
and allows you to test if a distribution type is available.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.uncertainty.distributions.distribution import Distribution

standard_library.install_aliases()


from gemseo import LOGGER


class DistributionFactory(object):
    """This factory instantiates a :class:`.Distribution` from its class name.
    The class can be internal to |g| or located in an external module
    whose path is provided to the constructor.
    """

    def __init__(self):
        """
        Initializes the factory: scans the directories to search for
        subclasses of Distribution.
        Searches in "GEMSEO_PATH" and gemseo.uncertainty.p_dist.
        """
        self.factory = Factory(Distribution, ("gemseo.uncertainty.distributions",))

    def create(self, distribution_name, variable, **parameters):
        """
        Creates a distribution.

        :param str distribution_name: name of the distribution (its classname)
        :param str variable: variable name.
        :param parameters: distribution parameters
        :return: distribution_name distribution
        """
        return self.factory.create(distribution_name, variable=variable, **parameters)

    @property
    def distributions(self):
        """
        Lists the available classes.

        :returns: the list of classes names.
        :rtype: list(str)
        """
        return self.factory.classes

    def is_available(self, distribution_name):
        """
        Checks the availability of a distribution.

        :param str distribution_name:  name of the distribution.
        :returns: True if the distribution is available.
        :rtype: bool
        """
        return self.factory.is_available(distribution_name)
