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
"""Factory for the trust updater."""
from __future__ import annotations

from gemseo.algos.opt.core.trust_updater import PenaltyUpdater
from gemseo.algos.opt.core.trust_updater import RadiusUpdater
from gemseo.algos.opt.core.trust_updater import TrustUpdater


class UpdaterFactory:
    """A factory of :class:`.TrustUpdater`."""

    RADIUS = "radius"
    PENALTY = "penalty"
    TRUST_PARAMETERS = [RADIUS, PENALTY]

    def __init__(self) -> None:  # noqa:D107
        self.__update_name_to_updater = {
            UpdaterFactory.RADIUS: RadiusUpdater,
            UpdaterFactory.PENALTY: PenaltyUpdater,
        }

    def create(
        self,
        name: str,
        thresholds: tuple[float, float],
        multipliers: tuple[float, float],
        bound: float,
    ) -> TrustUpdater:
        """Create a :class:`.TrustUpdater`.

        Args:
            name: The name of the updater.
            thresholds: The thresholds for the decrease ratio.
            multipliers: The multipliers for the trust parameter.
            bound: The absolute bound for the trust parameter.

        Raises:
            ValueError: When the updater does not exist.
        """
        if name not in self.__update_name_to_updater:
            raise ValueError(
                f"No update method named {name} is available among update methods: "
                f"{list(self.__update_name_to_updater.keys())}."
            )

        return self.__update_name_to_updater[name](thresholds, multipliers, bound)
