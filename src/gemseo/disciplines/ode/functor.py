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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A discipline for solving ordinary differential equations (ODEs)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.disciplines.ode.base_functor import BaseFunctor

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class Functor(BaseFunctor):
    """A function with time and state as arguments to compute RHS."""

    def __call__(self, time: RealArray, state: RealArray) -> RealArray:
        """
        Args:
            time: The time at the evaluation of the function.
            state: The state of the ODE at the evaluation of the function.

        Returns: The value of the function at the given time and state.

        """  # noqa: D205, D212, D415
        return self._adapter.evaluate(self._compute_input_vector(time, state))
