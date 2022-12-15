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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Aerostructure MDO problem's Design Space
****************************************
"""
from __future__ import annotations

import numpy as np

from gemseo.algos.design_space import DesignSpace


class AerostructureDesignSpace(DesignSpace):
    """**AerostructureDesignSpace** creates the :class:`.DesignSpace`
    of the Aerostructure problem
    whose :class:`.MDODiscipline` are :class:`.Aerodynamics`,
    :class:`.Structure` and :class:`.Mission`.

    - thick_airfoils belongs to [5., 25.], with initial value equal to 15.
    - thick_panels belongs to [1., 20.], with initial value equal to 3.
    - sweep belongs to [10., 35.], with initial value equal to 25.
    - drag belongs to [100., 1000.], with initial value equal to 340.
    - forces belongs to [-1000., 1000.], with initial value equal to 400.
    - lift belongs to [0.1, 1.0], with initial value equal to 0.5
    - mass belongs to [100000., 500000.], with initial value equal to 100000.
    - reserve_fact belongs to [1000., 1000.], with initial value equal to 0.
    """

    def __init__(self):
        """Constructor."""
        super().__init__()

        # construct a dictionary with initial solution
        drag = np.array([340.0], dtype=np.complex128)
        forces = np.array([400.0], dtype=np.complex128)
        lift = np.array([0.5], dtype=np.complex128)
        mass = np.array([100000.0], dtype=np.complex128)
        displ = np.array([-700.0], dtype=np.complex128)
        sweep = np.array([25.0], dtype=np.complex128)
        thick_airfoils = np.array([15.0], dtype=np.complex128)
        thick_panels = np.array([3.0], dtype=np.complex128)
        reserve_fact = np.array([0.0], dtype=np.complex128)

        # design variables
        self.add_variable("thick_airfoils", l_b=5.0, u_b=25.0, value=thick_airfoils)
        self.add_variable("thick_panels", l_b=1.0, u_b=20.0, value=thick_panels)

        # shared design variables
        self.add_variable("sweep", l_b=10.0, u_b=35.0, value=sweep)

        # target coupling variables
        self.add_variable("drag", l_b=100.0, u_b=1000.0, value=drag)
        self.add_variable("forces", l_b=-1000.0, u_b=1000.0, value=forces)
        self.add_variable("lift", l_b=0.1, u_b=1.0, value=lift)
        self.add_variable("mass", l_b=100000.0, u_b=500000.0, value=mass)
        self.add_variable("displ", l_b=-1000.0, u_b=1000.0, value=displ)
        self.add_variable("reserve_fact", l_b=-1000.0, u_b=1000.0, value=reserve_fact)
