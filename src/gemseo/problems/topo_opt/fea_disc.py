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
"""Finite element analysis (FEA) for 2D topology optimization problems."""
from __future__ import annotations

from typing import Sequence

import scipy
from numpy import arange
from numpy import array
from numpy import atleast_2d
from numpy import dot
from numpy import kron
from numpy import newaxis
from numpy import ones
from numpy import setdiff1d
from numpy import tile
from numpy import zeros

from gemseo.core.discipline import MDODiscipline


class FininiteElementAnalysis(MDODiscipline):
    """Finite Element Analysis for 2D topology optimization problems.

    Take in input the Young Modulus vector E and computes in output the compliance, i.e.
    twice the work of external forces.
    """

    def __init__(
        self,
        nu: float = 0.3,
        n_x: int = 100,
        n_y: int = 100,
        f_node: int | Sequence[int] = 101 * 101 - 1,
        f_direction: int | Sequence[int] = 1,
        f_amplitude: int | Sequence[int] = -1,
        fixed_nodes: int | Sequence[int] | None = None,
        fixed_dir: int | Sequence[int] | None = None,
        name: str | None = None,
    ) -> None:
        """
        Args:
            nu: The material Poisson's ratio.
            n_x: The number of elements in the x-direction.
            n_y: The number of elements in the y-direction.
            f_node: The indices of the nodes where the forces are applied.
            f_direction: The force direction for each ``f_node``, either 0 for x or 1 for y.
            f_amplitude: The force amplitude for each pair ``(f_node, f_direction)``.
            fixed_nodes: The indices of the nodes where the structure is clamped.
                If None, a default value is used.
            fixed_dir: The clamped direction for each node, encode 0 for x and 1 for y.
                If None, a default value is used.
            name: The name of the discipline.
                If None, use the class name.
        """

        super().__init__(name=name)
        if fixed_nodes is None:
            fixed_nodes = tile(arange(101), 2)
        if fixed_dir is None:
            fixed_dir = array([0] * 101 + [1] * 101)
        self.N_elements = n_x * n_y
        self.N_nodes = (n_x + 1) * (n_y + 1)
        self.N_DOFs = 2 * self.N_nodes
        self.n_x = n_x
        self.n_y = n_y
        self.nu = nu
        self.E = None
        self.KE = None
        self.iK = None
        self.jK = None
        self.edofMat = None
        self.freedofs = None
        self.f_node = f_node
        self.f_direction = f_direction
        self.f_amplitude = f_amplitude
        self.fixednodes = fixed_nodes
        self.fixed_dir = fixed_dir
        self.prepare_fea()
        self.input_grammar.update(["E"])
        self.output_grammar.update(["compliance"])
        self.default_inputs = {"E": ones(self.N_elements)}

    def _run(self) -> None:
        em = self.get_inputs_by_name("E")
        sk = ((self.KE.flatten()[newaxis]).T * em).flatten(order="F")
        k_mat = scipy.sparse.coo_matrix(
            (sk, (self.iK, self.jK)), shape=(self.N_DOFs, self.N_DOFs)
        ).tocsc()
        k_mat = k_mat[self.freedofs, :][:, self.freedofs]
        u_vec = zeros((self.N_DOFs, 1))
        f = zeros((self.N_DOFs, 1))
        f[2 * self.f_node + self.f_direction, 0] = self.f_amplitude
        u_vec[self.freedofs, 0] = scipy.sparse.linalg.spsolve(
            k_mat, f[self.freedofs, 0]
        )
        # Objective function and sensitivity
        ce = ones(self.N_elements)
        ce[:] = (
            dot(u_vec[self.edofMat].reshape(self.N_elements, 8), self.KE)
            * u_vec[self.edofMat].reshape(self.N_elements, 8)
        ).sum(1)
        self.local_data["compliance"] = array([(em * ce).sum()])
        self._is_linearized = True
        self._init_jacobian(with_zeros=True)
        self.jac["compliance"] = {}
        self.jac["compliance"]["E"] = atleast_2d(-ce)

    def prepare_fea(self) -> None:
        """Prepare the Finite Element Analysis."""
        self.KE = self.compute_elementary_stiffeness_matrix()

        # FE: Build the index vectors for the for coo matrix format.
        edof_mat = zeros((self.N_elements, 8), dtype=int)
        for elx in range(self.n_x):
            for ely in range(self.n_y):
                el = ely + elx * self.n_y
                n1 = (self.n_y + 1) * elx + ely
                n2 = (self.n_y + 1) * (elx + 1) + ely
                edof_mat[el, :] = array(
                    [
                        2 * n1 + 2,
                        2 * n1 + 3,
                        2 * n2 + 2,
                        2 * n2 + 3,
                        2 * n2,
                        2 * n2 + 1,
                        2 * n1,
                        2 * n1 + 1,
                    ]
                )
        self.edofMat = edof_mat
        # Construct the index pointers for the coo format
        self.iK = kron(edof_mat, ones((8, 1))).flatten()
        self.jK = kron(edof_mat, ones((1, 8))).flatten()

        # Free DOFs
        alldofs = array(range(0, 2 * self.N_nodes))
        fixeddofs = 2 * self.fixednodes + self.fixed_dir
        self.freedofs = setdiff1d(alldofs, fixeddofs)

    def compute_elementary_stiffeness_matrix(
        self,
    ) -> None:  # noqa: D205,D212,D415
        """Compute the elementary stiffness matrix of 1x1 quadrilateral elements."""
        em = 1.0
        k = array(
            [
                1 / 2 - self.nu / 6,
                1 / 8 + self.nu / 8,
                -1 / 4 - self.nu / 12,
                -1 / 8 + 3 * self.nu / 8,
                -1 / 4 + self.nu / 12,
                -1 / 8 - self.nu / 8,
                self.nu / 6,
                1 / 8 - 3 * self.nu / 8,
            ]
        )
        return (
            em
            / (1 - self.nu**2)
            * array(
                [
                    [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                ]
            )
        )
