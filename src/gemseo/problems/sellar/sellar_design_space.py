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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Sellar MDO problem's Design Space
*********************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import super

import numpy as np
from future import standard_library

from gemseo.algos.design_space import DesignSpace

standard_library.install_aliases()


class SellarDesignSpace(DesignSpace):

    """**SellarDesignSpace** creates the :class:`.DesignSpace` of the
    Sellar problem
    whose :class:`.MDODiscipline` are :class:`.Sellar1`, :class:`.Sellar2`
    and :class:`.SellarSystem`.

    - x_local belongs to [0., 10.]
    - x_shared_0 belongs to [-10., 10.]
    - x_shared_1 belongs to [0., 10.]
    - y_0 belongs to [-100., 100.]
    - y_1 belongs to [-100., 100.]
    """

    def __init__(self, dtype="complex128"):
        """The constructor creates a blank :class:`.DesignSpace`
        to which it adds all design variables."""
        super(SellarDesignSpace, self).__init__()

        # construct a dictionary with initial solution
        x_local = np.array([1.0], dtype=dtype)
        x_shared = np.array([4.0, 3.0], dtype=dtype)
        y_0 = np.array([1.0], dtype=dtype)
        y_1 = np.array([1.0], dtype=dtype)

        # design variables
        self.add_variable("x_local", 1, l_b=0.0, u_b=10.0, value=x_local)  # x
        self.add_variable(
            "x_shared", 2, l_b=(-10, 0.0), u_b=(10.0, 10.0), value=x_shared
        )  # z

        # target coupling variables
        self.add_variable("y_0", 1, l_b=-100.0, u_b=100.0, value=y_0)
        self.add_variable("y_1", 1, l_b=-100.0, u_b=100.0, value=y_1)
