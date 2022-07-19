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
#
# Contributors:
# - Jean-Christophe Giret
# - François Gallard
# - Matthias De Lozzo
# - Charlie Vanaret
# - Benoit Pauwels
# - Antoine DECHAUME
"""A factory to instantiate a formulation or check its availability."""
from __future__ import annotations

from gemseo.core.base_formulation import BaseFormulationsFactory
from gemseo.core.formulation import MDOFormulation


class MDOFormulationsFactory(BaseFormulationsFactory):
    """A factory of :class:`~gemseo.core.formulation.MDOFormulation`."""

    def __init__(self) -> None:
        super().__init__(MDOFormulation, ("gemseo.formulations",))
