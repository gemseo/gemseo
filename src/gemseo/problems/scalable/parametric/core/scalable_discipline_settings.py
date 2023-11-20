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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Settings of a scalable discipline."""

from __future__ import annotations

from typing import NamedTuple

from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_D_I
from gemseo.problems.scalable.parametric.core.default_settings import (
    DEFAULT_N_DISCIPLINES,
)
from gemseo.problems.scalable.parametric.core.default_settings import DEFAULT_P_I


class ScalableDisciplineSettings(NamedTuple):
    """The configuration of a scalable discipline."""

    d_i: int = DEFAULT_D_I
    r"""The size of local design variable :math:`x_i` specific to this discipline."""

    p_i: int = DEFAULT_P_I
    r"""The size of the coupling variable :math:`y_i` outputted by this discipline."""


DEFAULT_SCALABLE_DISCIPLINE_SETTINGS = tuple(
    ScalableDisciplineSettings() for _ in range(DEFAULT_N_DISCIPLINES)
)
"""The default settings of the scalable disciplines used in a scalable problem."""
