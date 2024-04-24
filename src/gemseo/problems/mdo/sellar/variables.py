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
"""The variable names of the customizable Sellar MDO problem."""

from typing import Final

Y_1: Final[str] = "y_1"
"""The name of the coupling variable computed by :class:`.Sellar1`."""

Y_2: Final[str] = "y_2"
"""The name of the coupling variable computed by :class:`.Sellar2`."""

X_SHARED: Final[str] = "x_shared"
"""The name of the shared design variable."""

X_1: Final[str] = "x_1"
"""The name of the local design variable specific to :class:`.Sellar1`."""

X_2: Final[str] = "x_2"
"""The name of the local design variable specific to :class:`.Sellar2`."""

OBJ: Final[str] = "obj"
"""The name of the objective to minimize."""

C_1: Final[str] = "c_1"
"""The name of the constraint based on ``"y_1"``."""

C_2: Final[str] = "c_2"
"""The name of the constraint based on ``"y_2"``."""

ALPHA: Final[str] = "alpha"
"""The name of the tunable parameter in the constraint ``"c_1"``."""

BETA: Final[str] = "beta"
"""The name of the tunable parameter in the constraint ``"c_2"``."""

GAMMA: Final[str] = "gamma"
"""The name of the tunable parameter in the discipline :class:`.Sellar1`."""
