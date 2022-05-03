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
"""Some chains of SSBJ disciplines."""
from __future__ import annotations

import logging

from gemseo.core.chain import MDOChain
from gemseo.problems.sobieski.disciplines import create_disciplines

LOGGER = logging.getLogger(__name__)


class SobieskiChain(MDOChain):
    """Chain the disciplines of the Sobieski's SSBJ use case.

    Order: structure, aerodynamics, propulsion and mission.
    """

    def __init__(self, dtype: str = "float64") -> None:
        """
        Args:
            dtype: The NumPy type for data arrays, either "float64" or "complex128".
        """
        super().__init__(create_disciplines(dtype), "SobieskiChain")
