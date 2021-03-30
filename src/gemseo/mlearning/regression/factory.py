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
Regression model factory
========================

This module contains a factory to instantiate a :class:`.MLRegressionAlgo`
from its class name. The class can be internal to |g| or located in an
external module whose path is provided to the constructor. It also provides a
list of available regression models and allows you to test if a regression
model type is available.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.mlearning.core.factory import MLAlgoFactory
from gemseo.mlearning.regression.regression import MLRegressionAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class RegressionModelFactory(MLAlgoFactory):
    """This factory instantiates a :class:`.MLRegressionAlgo`
    from its class name.
    The class can be internal to |g| or located in an external module
    whose path is provided to the constructor.
    """

    def __init__(self):
        """Initializes the factory: scans the directories to search for
        subclasses of :class:`.MLRegressionAlgo`. Searches in "GEMSEO_PATH"
        and gemseo.mlearning.regression.
        """
        super(RegressionModelFactory, self).__init__()
        self.factory = Factory(MLRegressionAlgo, ("gemseo.mlearning.regression",))
