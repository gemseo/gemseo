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
"""A discipline for topology optimization material model interpolation."""
from __future__ import annotations

from typing import Sequence

from numpy import atleast_2d
from numpy import diag
from numpy import ones
from numpy import ones_like

from gemseo.core.discipline import MDODiscipline


class MaterialModelInterpolation(MDODiscipline):
    """Material Model Interpolation class for topology optimization problems.

    Compute the Young's modulus (E) and the material local density (rho) from filtered
    design variables xPhys with the SIMP (Solid Isotropic Material with Penalization)
    exponential method.
    """

    def __init__(
        self,
        e0: float,
        penalty: float,
        n_x: int,
        n_y: int,
        empty_elements: Sequence[int],
        full_elements: Sequence[int],
        contrast: float = 1e9,
    ) -> None:
        """
        Args:
            e0: The full material Young modulus.
            penalty: The SIMP penalty coefficient.
            n_x: The number of elements in the x-direction.
            n_y: The number of elements in the y-direction.
            empty_elements: The index of an empty element
                ids that are not part of the design space.
            full_elements: The index of full element
                ids that are not part of the design space.
            contrast: The ratio between the full material Young's modulus
                and void material Young's modulus.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.E0 = e0
        self.penalty = penalty
        self.Emin = e0 / contrast
        self.empty_elements = empty_elements
        self.full_elements = full_elements
        self.N_elements = n_x * n_y
        self.input_grammar.update(["xPhys"])
        self.output_grammar.update(["rho", "E"])
        self.default_inputs = {"xPhys": ones(n_x * n_y)}

    def _run(self) -> None:
        xphys = self.get_inputs_by_name("xPhys")
        xphys[self.empty_elements] = 0
        xphys[self.full_elements] = 1
        xphys = xphys.flatten()
        rho = xphys[:]
        young_modulus = self.Emin + (self.E0 - self.Emin) * xphys**self.penalty
        self.local_data["E"] = young_modulus
        self.local_data["rho"] = rho
        self._is_linearized = True
        self._init_jacobian(with_zeros=True)
        dyoung_modulus_dxphys = (
            self.penalty * xphys.ravel() ** (self.penalty - 1) * (self.E0 - self.Emin)
        )
        dyoung_modulus_dxphys[self.empty_elements] = 0
        dyoung_modulus_dxphys[self.full_elements] = 0
        self.jac["E"] = {"xPhys": atleast_2d(diag(dyoung_modulus_dxphys))}
        drho_dxphys = ones_like(xphys)
        drho_dxphys[self.empty_elements] = 0
        drho_dxphys[self.full_elements] = 0
        self.jac["rho"] = {"xPhys": atleast_2d(diag(drho_dxphys))}
