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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Machine learning algorithm factory
==================================

This module contains a factory to instantiate a :class:`.MLAlgo` from its class
name. The class can be internal to |g| or located in an external module whose
path is provided to the constructor. It also provides a list of available
machine learning algorithm types and allows you to test if a machine learning
algorithm type is available.
"""
from __future__ import absolute_import, division, unicode_literals

import pickle
from os.path import join

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.mlearning.core.ml_algo import MLAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class MLAlgoFactory(object):
    """This factory instantiates a :class:`.MLAlgo`
    from its class name. The class can be internal to |g| or located in an
    external module whose path is provided to the constructor.
    """

    def __init__(self):
        """Initializes the factory: scans the directories to search for
        subclasses of MLAlgo. Searches in "GEMSEO_PATH" and gemseo.mlearning.
        """
        self.factory = Factory(MLAlgo, ("gemseo.mlearning",))

    def create(self, ml_algo, **options):
        """Create machine learning algorithm.

        :param str ml_algo: name of the machine learning algorithm (its
            classname).
        :param options: machine learning algorithm options.
        :return: MLAlgo
        """
        return self.factory.create(ml_algo, **options)

    @property
    def models(self):
        """Lists the available classes.

        :returns: list of class names.
        :rtype: list(str)
        """
        return self.factory.classes

    def is_available(self, ml_algo):
        """Checks the availability of a cache.

        :param str ml_algo:  name of the machine learning algorithm (its class
            name).
        :returns: True if the machine learning algorithm is available.
        :rtype: bool
        """
        return self.factory.is_available(ml_algo)

    def load(self, directory):
        """Load a machine learning algorithm from the disk.

        :param str directory: directory name.
        """
        with open(join(str(directory), MLAlgo.FILENAME), "rb") as handle:
            objects = pickle.load(handle)
        model = self.factory.create(
            objects.pop("algo_name"),
            data=objects.pop("data"),
            **objects.pop("parameters")
        )
        for key, value in objects.items():
            setattr(model, key, value)
        model.load_algo(directory)
        return model
