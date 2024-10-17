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
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A multidisciplinary scenario to be executed by an optimizer."""

from __future__ import annotations

from typing import ClassVar

from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.scenarios.base_scenario import BaseScenario


class MDOScenario(BaseScenario):
    """A multidisciplinary scenario to be executed by an optimizer."""

    _ALGO_FACTORY_CLASS: ClassVar[type[OptimizationLibraryFactory]] = (
        OptimizationLibraryFactory
    )
