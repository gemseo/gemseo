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
"""Quasi-Monte Carlo sampling algorithm using the Sobol' sequence."""

from __future__ import annotations

from typing import ClassVar

from gemseo.uncertainty.reliability.openturns.mc import OT_MC
from gemseo.uncertainty.reliability.openturns.sobol_settings import OT_Sobol_Settings


class OT_Sobol(OT_MC):  # noqa: N801
    """The quasi-Monte Carlo sampling algorithm using the Sobol' sequence."""

    settings_class: ClassVar[type[OT_Sobol_Settings]] = OT_Sobol_Settings
