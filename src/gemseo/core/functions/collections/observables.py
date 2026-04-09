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
"""A mutable sequence of observables."""

from __future__ import annotations

import logging
from typing import ClassVar

from gemseo.core.functions.array_function import ArrayFunction
from gemseo.core.functions.collections.functions import Functions

LOGGER = logging.getLogger(__name__)


class Observables(Functions):
    """A mutable sequence of observables."""

    _F_TYPES: ClassVar[tuple[ArrayFunction.FunctionType]] = (
        ArrayFunction.FunctionType.OBS,
    )

    def format(self, function: ArrayFunction) -> ArrayFunction | None:
        """Format an observable.

        Returns:
            A formatted observable ready to be added to the sequence.
        """
        name = function.name
        if name in self.get_names():
            LOGGER.warning('The optimization problem already observes "%s".', name)
            return None

        function.f_type = function.FunctionType.OBS
        return function
