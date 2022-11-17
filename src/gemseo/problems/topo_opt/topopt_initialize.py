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
"""Generation of the design space and disciplines of the topology optimization
problems."""
from __future__ import annotations

from numpy import arange
from numpy import concatenate
from numpy import dot
from numpy import fix
from numpy import full
from numpy import kron
from numpy import logical_and
from numpy import ones
from numpy import where
from numpy import zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.topo_opt.density_filter_disc import DensityFilter
from gemseo.problems.topo_opt.fea_disc import FininiteElementAnalysis
from gemseo.problems.topo_opt.material_model_interpolation_disc import (
    MaterialModelInterpolation,
)
from gemseo.problems.topo_opt.volume_fraction_disc import VolumeFraction


def initialize_design_space_and_discipline_to(
    problem: str,
    n_x: int,
    n_y: int,
    e0: float,
    nu: float,
    penalty: float,
    min_member_size: float,
    vf0: float,
) -> tuple[DesignSpace, list[MDODiscipline]]:
    """Initialize design space and disciplines for 2D topology optimization problems.

    Args:
        problem: The problem name, one of "MBB", "L-Shape", "Short_Cantilever".
        n_x: The number of elements in the x-direction.
        n_y: The number of elements in the y-direction.
        e0: The full material Young's modulus.
        nu: The material Poisson's ratio.
        penalty: The SIMP penalty coefficient.
        min_member_size: The minimum structural member size.
        vf0: The minimum structural element dimension
            imposed in the topology optimization solution.

    Returns:

        - The design space.
        - The disciplines.
    """
    # Define the nodal coordinates
    x = 0
    yy = zeros((n_x + 1) * (n_y + 1))
    xx = zeros((n_x + 1) * (n_y + 1))
    for i in range(1, n_x + 2):
        for j in range(1, n_y + 2):
            yy[x] = j
            xx[x] = i
            x += 1
    yy = n_y + 1 - yy
    xx -= 1
    # Compute the centroid coordinates
    xc = zeros(n_x * n_y)
    yc = zeros(n_x * n_y)
    for xi in range(n_x):
        xc[xi * n_y : (xi + 1) * n_y] = xi + 0.5
    for yi in range(n_y):
        yc[arange(yi, n_x * n_y, n_y)] = yi + 0.5
    yc = n_y - yc
    # DEFINE LOADS AND SUPPORTS
    if "MBB" == problem:
        excitation_node = 0  # Node where the force is applied
        excitation_direction = 1  # 0 for x and 1 for y
        amplitude = -1  # Amplitude of the force
        fixednodes = concatenate(
            ([[where(xx == min(xx))], [(n_x + 1) * (n_y + 1) - 1]]), axis=None
        )  # Fixed nodes
        fixed_dir = concatenate(([[ones(n_y + 1)], [2]]), axis=None) - 1
        emptyelts = []  # Mandatory empty elements
        fullelts = []  # Mandatory full elements
    elif "Short_Cantilever" == problem:
        excitation_node = where(
            logical_and(
                (xx == max(xx)), (yy == fix(dot(0.5, min(yy)) + dot(0.5, max(yy))))
            )
        )[0][
            0
        ]  # Nodes where the force is applied
        excitation_direction = 1  # 0 for x and 1 for y
        amplitude = -1  # Amplitude of the force
        fixednodes = kron([1, 1], where(xx == min(xx))[0])  # Fixed nodes
        fixed_dir = (
            concatenate([[ones(n_y + 1)], [dot(2, ones(n_y + 1))]]).flatten() - 1
        )
        emptyelts = []  # Mandatory empty elements
        fullelts = []  # Mandatory full elements
    elif "L-Shape" == problem:
        excitation_node = where(
            logical_and(
                (xx == max(xx)), (yy == fix(dot(0.5, min(yy)) + dot(0.5, max(yy))))
            )
        )[0][
            0
        ]  # Nodes where the force is applied
        excitation_direction = 1  # 0 for x and 1 for y
        amplitude = -1  # Amplitude of the force
        fixednodes = kron([1, 1], where(yy == max(yy))[0])  # Fixed nodes
        fixed_dir = concatenate([[ones(n_x + 1)], [dot(2, ones(n_x + 1))]]).flatten()
        emptyelts = where(
            logical_and(xc >= (max(xx) + min(xx)) / 2, yc >= ((max(yy) + min(yy)) / 2))
        )[
            0
        ]  # Mandatory empty elements
        fullelts = []  # Mandatory full element
    else:
        msg = "The examples covered by this function are MBB, L-Shape and Short_Cantilever."
        raise NotImplementedError(msg)
    initial_point = full((n_x * n_y,), vf0)
    initial_point[emptyelts] = 0
    initial_point[fullelts] = 1
    ds = DesignSpace()
    ds.add_variable(
        "x",
        size=n_x * n_y,
        l_b=zeros((n_x * n_y,)),
        u_b=ones((n_x * n_y,)),
        value=initial_point,
    )
    df = DensityFilter(n_x=n_x, n_y=n_y, min_member_size=min_member_size)
    mmi = MaterialModelInterpolation(
        e0=e0,
        penalty=penalty,
        n_x=n_x,
        n_y=n_y,
        empty_elements=emptyelts,
        full_elements=fullelts,
    )
    fea = FininiteElementAnalysis(
        nu=nu,
        n_x=n_x,
        n_y=n_y,
        f_node=excitation_node,
        f_direction=excitation_direction,
        f_amplitude=amplitude,
        fixed_nodes=fixednodes,
        fixed_dir=fixed_dir,
    )
    vf = VolumeFraction(
        n_x=n_x, n_y=n_y, empty_elements=emptyelts, full_elements=fullelts
    )
    disciplines = [df, mmi, fea, vf]
    return ds, disciplines
