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
"""A factory to instantiate a formulation or check its availability."""
from __future__ import annotations

import logging
from typing import Sequence

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.factory import Factory
from gemseo.core.formulation import MDOFormulation

LOGGER = logging.getLogger(__name__)


class MDOFormulationsFactory:
    """MDO Formulations factory to create the formulation from a name or a class."""

    def __init__(self) -> None:
        """Scan the directories to search for subclasses of :class:`.MDOFormulation`.

        Searches in "GEMSEO_PATH" and gemseo.formulations
        """
        self.factory = Factory(MDOFormulation, ("gemseo.formulations",))

    def create(
        self,
        formulation_name: str,
        disciplines: Sequence[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        **options,
    ) -> MDOFormulation:
        """Create a formulation.

        Args:
            formulation_name: The name of a class implementing a formulation.
            disciplines: The disciplines.
            objective_name: The name of the objective function.
            design_space: The design space.
            **options: The options for the creation of the formulation.
        """
        return self.factory.create(
            formulation_name,
            disciplines=disciplines,
            design_space=design_space,
            objective_name=objective_name,
            **options,
        )

    @property
    def formulations(self) -> list[str]:
        """The available formulations."""
        return self.factory.classes

    def is_available(
        self,
        formulation_name: str,
    ) -> bool:
        """Check the availability of a formulation.

        Args:
            formulation_name: The formulation name to check.

        Returns:
            Whether the formulation is available.
        """
        return self.factory.is_available(formulation_name)
