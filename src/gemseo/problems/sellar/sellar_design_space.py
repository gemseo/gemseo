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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The design space for the MDO problem proposed by Sellar et al. in.

Sellar, R., Batill, S., & Renaud, J. (1996). Response surface based, concurrent subspace
optimization for multidisciplinary system design. In 34th aerospace sciences meeting and
exhibit (p. 714).
"""
from __future__ import annotations

from numpy import array
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.sellar.sellar import X_LOCAL
from gemseo.problems.sellar.sellar import X_SHARED
from gemseo.problems.sellar.sellar import Y_1
from gemseo.problems.sellar.sellar import Y_2


class SellarDesignSpace(DesignSpace):
    """The design space for the MDO problem proposed by Sellar et al (1996).

    It is composed of:
    - :math:`x_{local}` belonging to :math:`[0., 10.]`,
    - :math:`x_{shared,1}` belonging to :math:`[-10., 10.]`,
    - :math:`x_{shared,2}` belonging to :math:`[0., 10.]`,
    - :math:`y_1` belonging to :math:`[-100., 100.]`,
    - :math:`y_2` belonging to :math:`[-100., 100.]`.

    This design space is initialized with the initial solution:

    - :math:`x_{local}=1`,
    - :math:`x_{shared,1}=4`,
    - :math:`x_{shared,2}=3`,
    - :math:`y_1=1`,
    - :math:`y_2=1`.
    """

    def __init__(
        self,
        dtype: str = "complex128",
    ) -> None:
        """
        Args:
            dtype: The type of the variables defined in the design space.
        """
        super().__init__()

        x_local, x_shared, y_1, y_2 = self.__get_initial_solution(dtype)
        self.add_variable(X_LOCAL, l_b=0.0, u_b=10.0, value=x_local)
        self.add_variable(X_SHARED, 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=x_shared)
        self.add_variable(Y_1, l_b=-100.0, u_b=100.0, value=y_1)
        self.add_variable(Y_2, l_b=-100.0, u_b=100.0, value=y_2)

    @staticmethod
    def __get_initial_solution(
        dtype: str,
    ) -> tuple[ndarray]:
        """Return an initial solution for the MDO problem.

        Args:
            dtype: The type of the variables defined in the design space.

        Returns:
            An initial solution for both local design variables,
            shared design variables and coupling variables.
        """
        x_local = array([1.0], dtype=dtype)
        x_shared = array([4.0, 3.0], dtype=dtype)
        y_1 = array([1.0], dtype=dtype)
        y_2 = array([1.0], dtype=dtype)
        return x_local, x_shared, y_1, y_2
