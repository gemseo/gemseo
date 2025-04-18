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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Simone Coniglio
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A discipline for topology optimization density filter."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import ceil
from numpy import maximum
from numpy import minimum
from numpy import newaxis
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from scipy.sparse import csr_array
from scipy.sparse import diags

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


class DensityFilter(Discipline):
    """Apply density filter to the design variables of a topology optimization problem.

    This helps to avoid checkerboard patterns in density based topology optimization.
    A filter matrix :math:`H` is assembled at instantiation.
    The discipline computes the physical density :math:`xPhys` from the design variables
    :math:`x` with the formula :math:`xPhys=Hx`.
    This physical density is an approximation of a convolution integral.
    """

    def __init__(
        self,
        n_x: int = 100,
        n_y: int = 100,
        min_member_size: float = 1.5,
        name: str = "",
    ) -> None:
        """
        Args:
            n_x: The number of elements in the x-direction.
            n_y: The number of elements in the y-direction.
            min_member_size: The minimum structural member size.
        """  # noqa: D205, D212, D415
        super().__init__(name=name)
        self.n_x = n_x
        self.n_y = n_y
        self.min_member_size = min_member_size
        self.filter_matrix = None
        self._create_filter_matrix()
        self.io.input_grammar.update_from_names(["x"])
        self.io.output_grammar.update_from_names(["xPhys"])
        self.io.input_grammar.defaults = {"x": ones((n_x * n_y,))}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x = input_data["x"][:, newaxis]
        self.io.update_output_data({"xPhys": (self.filter_matrix @ x).flatten()})
        self._has_jacobian = True
        self._init_jacobian()
        self.jac["xPhys"] = {"x": self.filter_matrix}

    def _create_filter_matrix(self) -> None:
        """Create the filter matrix."""
        # Filter: Build (and assemble) the index+data vectors for the coo matrix format
        n_filter = int(
            self.n_x * self.n_y * ((2 * (ceil(self.min_member_size) - 1) + 1) ** 2)
        )
        ih = zeros(n_filter)
        jh = zeros(n_filter)
        sh = zeros(n_filter)
        cc = 0
        for i_x in range(self.n_x):
            for i_y in range(self.n_y):
                row = i_x * self.n_y + i_y
                kk1 = int(maximum(i_x - (ceil(self.min_member_size) - 1), 0))
                kk2 = int(minimum(i_x + ceil(self.min_member_size), self.n_x))
                ll1 = int(maximum(i_y - (ceil(self.min_member_size) - 1), 0))
                ll2 = int(minimum(i_y + ceil(self.min_member_size), self.n_y))
                for k in range(kk1, kk2):
                    for ll in range(ll1, ll2):
                        col = k * self.n_y + ll
                        fac = self.min_member_size - norm([i_x - k, i_y - ll])
                        ih[cc] = row
                        jh[cc] = col
                        sh[cc] = maximum(0.0, fac)
                        cc += 1
        # Finalize assembly and convert to csc format
        h_mat = csr_array(
            (sh, (ih, jh)), shape=(self.n_x * self.n_y, self.n_x * self.n_y)
        )
        self.filter_matrix = diags(1 / h_mat.sum(1)).tocsr() @ h_mat
