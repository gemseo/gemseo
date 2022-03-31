# -*- coding: utf-8 -*-
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
from __future__ import division
from __future__ import unicode_literals

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sobieski.core.problem import SobieskiProblem


def create_design_space(
    dtype,  # type: str
):  # type: (...) -> DesignSpace
    """Create the design space for the Sobieski's SSBJ use case.

    Args:
        dtype: The data type for the NumPy arrays, either "float64" or "complex128".

    Returns:
        The design space for the Sobieski's SSBJ use case.
    """
    return SobieskiProblem(dtype).design_space
