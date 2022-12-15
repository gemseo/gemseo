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
"""A discipline for topology optimization volume fraction."""
from __future__ import annotations

from typing import Sequence

from numpy import array
from numpy import atleast_2d
from numpy import mean
from numpy import ones
from numpy import ones_like
from numpy import size

from gemseo.core.discipline import MDODiscipline


class VolumeFraction(MDODiscipline):
    """Compute the volume fraction from the density.

    Volume fraction is computed as the average of the density value (rho) on each finite
    element.
    """

    def __init__(
        self,
        n_x: int = 100,
        n_y: int = 100,
        empty_elements: Sequence[int] | None = None,
        full_elements: Sequence[int] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            n_x: The number of elements in the x-direction.
            n_y: The number of elements in the y-direction.
            empty_elements: The index of the empty element
                ids that are not part of the design space.
            full_elements: The index of the full element
                ids that are not part of the design space.
            name: The name of the discipline.
                If None, use the class name.
        """  # noqa: D205, D212, D415
        super().__init__(name=name)
        self.n_x = n_x
        self.n_y = n_y
        self.input_grammar.update(["rho"])
        self.output_grammar.update(["volume fraction"])
        self.default_inputs = {"rho": ones(n_x * n_y)}

    def _run(self) -> None:
        rho = self.get_inputs_by_name("rho")
        self.local_data["volume fraction"] = array([mean(rho.ravel())])
        self._is_linearized = True
        self._init_jacobian(with_zeros=True)
        self.jac["volume fraction"] = {"rho": atleast_2d(ones_like(rho).T / size(rho))}
