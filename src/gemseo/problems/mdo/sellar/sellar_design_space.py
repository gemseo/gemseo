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
"""The design space for the customizable Sellar MDO problem."""

from __future__ import annotations

from numpy import array
from numpy import ones
from strenum import StrEnum

from gemseo.algos.design_space import DesignSpace
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2


class RealOrComplexDType(StrEnum):
    """A real or complex NumPy data type."""

    COMPLEX = "complex128"
    FLOAT = "float64"


class SellarDesignSpace(DesignSpace):
    r"""The design space for the customizable Sellar MDO problem.

    - :math:`x_1\in[0., 10.]^n` (initial: 1),
    - :math:`x_2\in[0., 10.]^n` (initial: 1),
    - :math:`x_{shared,1}\in[-10., 10.]` (initial: 4),
    - :math:`x_{shared,2}\in[0., 10.]` (initial: 3),
    - :math:`y_1\in[-100., 100.]^n` (initial: 1),
    - :math:`y_2\in[-100., 100.]^n` (initial: 1),

    where :math:`n` is the size of the local design variables and coupling variables.
    """

    def __init__(
        self,
        dtype: RealOrComplexDType = RealOrComplexDType.COMPLEX,
        n: int = 1,
        add_couplings: bool = True,
    ) -> None:
        """
        Args:
            dtype: The type of the variables defined in the design space.
            n: The size of the local design variables and coupling variables.
            add_couplings: Whether to add the coupling variables to the design space.
        """  # noqa: D205 D212
        super().__init__()
        self.add_variable(
            X_1, lower_bound=0.0, upper_bound=10.0, value=ones(n, dtype=dtype), size=n
        )
        self.add_variable(
            X_2, lower_bound=0.0, upper_bound=10.0, value=ones(n, dtype=dtype), size=n
        )
        self.add_variable(
            X_SHARED,
            2,
            lower_bound=(-10, 0.0),
            upper_bound=(10.0, 10.0),
            value=array([4.0, 3.0], dtype=dtype),
        )
        if add_couplings:
            self.add_variable(
                Y_1,
                lower_bound=-100.0,
                upper_bound=100.0,
                value=ones(n, dtype=dtype),
                size=n,
            )
            self.add_variable(
                Y_2,
                lower_bound=-100.0,
                upper_bound=100.0,
                value=ones(n, dtype=dtype),
                size=n,
            )
