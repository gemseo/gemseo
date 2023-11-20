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
"""The functions to define the names of the variables used in the scalable problem."""

from __future__ import annotations

from typing import Final

SHARED_DESIGN_VARIABLE_NAME: Final[str] = "x_0"
"""The name of the shared design variables."""

OBJECTIVE_NAME: Final[str] = "f"
"""The name of the objective."""

LOCAL_DESIGN_VARIABLE_BASE_NAME: Final[str] = "x"
"""The base name of a design variable.

To be suffixed by the index of the corresponding scalable discipline, e.g. ``"x_3"``.
"""

UNCERTAIN_VARIABLE_BASE_NAME: Final[str] = "u"
"""The base name of an uncertain variable.

To be suffixed by the index of the corresponding scalable discipline, e.g. ``"u_3"``.
"""

CONSTRAINT_VARIABLE_BASE_NAME: Final[str] = "c"
"""The base name of a constraint.

To be suffixed by the index of the corresponding scalable discipline, e.g. ``"c_3"``.
"""

COUPLING_VARIABLE_BASE_NAME: Final[str] = "y"
"""The base name of a coupling variable.

To be suffixed by the index of the corresponding scalable discipline, e.g. ``"y_3"``.
"""


def get_u_local_name(index: int) -> str:
    """Return the name of an uncertain variable specific to a scalable discipline.

    Args:
        index: The index of the scalable discipline.

    Returns:
        The name of the uncertain variable specific to the scalable discipline.
    """
    return __compute_name(UNCERTAIN_VARIABLE_BASE_NAME, index)


def get_x_local_name(index: int) -> str:
    """Return the name of the design variable specific to a scalable discipline.

    Args:
        index: The index of the scalable discipline.

    Returns:
        The name of the design variable specific to the scalable discipline.
    """
    return __compute_name(LOCAL_DESIGN_VARIABLE_BASE_NAME, index)


def get_coupling_name(index: int) -> str:
    """Return the name of the coupling variable outputted by a scalable discipline.

    Args:
        index: The index of the scalable discipline.

    Returns:
        The name of the coupling variable outputted by the scalable discipline.
    """
    return __compute_name(COUPLING_VARIABLE_BASE_NAME, index)


def get_constraint_name(index: int) -> str:
    """Return the name of the constraint specific to a scalable discipline.

    Args:
        index: The index of the scalable discipline.

    Returns:
        The name of the constraint specific to the scalable discipline.
    """
    return __compute_name(CONSTRAINT_VARIABLE_BASE_NAME, index)


def __compute_name(base_name: str, index: int) -> str:
    """Define a name from a base name and an index.

    Args:
        base_name: The base name.
        index: The index.

    Returns:
        The name joining the base name and the index.
    """
    return f"{base_name}_{index}"
