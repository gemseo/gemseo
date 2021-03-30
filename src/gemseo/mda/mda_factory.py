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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
A factory to instantiate MDA from their class names
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library

from gemseo.core.factory import Factory
from gemseo.mda.mda import MDA

standard_library.install_aliases()

from gemseo import LOGGER


class MDAFactory(object):
    """MDA factory to create the MDA from a name or a class."""

    def __init__(self):
        """
        Initializes the factory: scans the directories to search for
        subclasses of MDA.
        Searches in "GEMSEO_PATH" and gemseo.mda
        """
        self.factory = Factory(MDA, ("gemseo.mda",))

    def create(self, mda_name, disciplines, **options):
        """
        Create a MDA

        :param mda_name: name of the MDA (its classname)
        :param disciplines: list of the disciplines
        :param options: additional options specific
            to the MDA
        """

        return self.factory.create(mda_name, disciplines=disciplines, **options)

    @property
    def mdas(self):
        """
        Lists the available classes

        :returns : the list of classes names
        """
        return self.factory.classes

    def is_available(self, mda_name):
        """
        Checks the availability of a MDA

        :param name :  mda_name of the MDA
        :returns: True if the MDA is available
        """
        return self.factory.is_available(mda_name)
