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
#        :author: Benoit Pauwels
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Factory for the trust updater
*****************************
"""
from __future__ import division, unicode_literals

import logging

from gemseo.algos.opt.core.trust_updater import PenaltyUpdater, RadiusUpdater

LOGGER = logging.getLogger(__name__)


class UpdaterFactory(object):
    """Creates the trust updater."""

    RADIUS = "radius"
    PENALTY = "penalty"
    TRUST_PARAMETERS = [RADIUS, PENALTY]

    def __init__(self):
        """Initializer."""
        self.__update_name_to_updater = {
            UpdaterFactory.RADIUS: RadiusUpdater,
            UpdaterFactory.PENALTY: PenaltyUpdater,
        }

    def create(self, name, thresholds, multipliers, bound):
        """Factory method to create a TrustUpdater subclass from an update name.

        :param name: update name
        :type name: string
        :param thresholds: thresholds for the decreases ratio
        :type thresholds: tuple
        :param multipliers: multipliers for the trust parameter
        :type multipliers: tuple
        :param bound: (lower or upper) bound for the trust parameter
        :returns: trust updater
        """
        if name in self.__update_name_to_updater:
            updater = self.__update_name_to_updater[name](
                thresholds, multipliers, bound
            )
        else:
            raise ValueError(
                "No update method named "
                + str(name)
                + " is available among update methods : "
                + str(list(self.__update_name_to_updater.keys()))
            )
        return updater
