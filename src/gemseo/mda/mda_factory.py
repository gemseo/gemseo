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
"""A factory to instantiate MDA from their class names."""
from __future__ import annotations

from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Union

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.factory import Factory
from gemseo.mda.mda import MDA

MDAOptionType = Optional[
    Union[float, int, bool, str, Iterable[MDOCouplingStructure], Sequence[MDA]]
]


class MDAFactory:
    """MDA factory to create the MDA from a name or a class."""

    def __init__(self) -> None:
        self.factory = Factory(MDA, ("gemseo.mda",))

    def create(
        self,
        mda_name: str,
        disciplines: Sequence[MDODiscipline],
        **options: MDAOptionType,
    ) -> MDA:
        """Create an MDA.

        Args:
            mda_name: The name of the MDA (its class name).
            disciplines: The disciplines.
            **options: The options of the MDA.
        """

        return self.factory.create(mda_name, disciplines=disciplines, **options)

    @property
    def mdas(self) -> list[str]:
        """The names of the available MDAs."""
        return self.factory.classes

    def is_available(
        self,
        mda_name: str,
    ) -> bool:
        """Check the availability of an MDA.

        Args:
            mda_name: The name of the MDA.

        Returns:
            Whether the MDA is available.
        """
        return self.factory.is_available(mda_name)
