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
"""The design space of the Sobieski's SSBJ use case."""
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.problem import SobieskiProblem


def create_design_space(
    dtype: str = "float64", physical_naming: bool = False
) -> DesignSpace:
    """Create the design space for the Sobieski's SSBJ use case.

    Args:
        dtype: The data type for the NumPy arrays, either "float64" or "complex128".
        physical_naming: Whether to use physical names
            for the input and output variables (e.g. `"range"`)
            or mathematical notations (e.g. `"y_4"`).

    Returns:
        The design space for the Sobieski's SSBJ use case.
    """
    if physical_naming:
        return SobieskiProblem(dtype).design_space_with_physical_naming
    else:
        return SobieskiProblem(dtype).design_space
