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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Some chains of SSBJ disciplines
*******************************
"""
from __future__ import division, unicode_literals

import logging

from gemseo.core.chain import MDOChain
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.jacobi import MDAJacobi
from gemseo.problems.sobieski.wrappers import (
    SobieskiAerodynamics,
    SobieskiMission,
    SobieskiPropulsion,
    SobieskiStructure,
)

LOGGER = logging.getLogger(__name__)


class SobieskiChain(MDOChain):
    """Chains Sobieski disciplines : weight, aero, power and range"""

    def __init__(self, dtype="float64"):
        """Constructor.

        :param dtype: data array type, either "float64" or "complex128".
        :type dtype: str
        """
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiPropulsion(dtype),
            SobieskiMission(dtype),
        ]
        super(SobieskiChain, self).__init__(disciplines, "SobieskiChain")


class SobieskiMDAGaussSeidel(MDAGaussSeidel):
    """Chains Sobieski disciplines to perform and MDA by Gauss Seidel algorithm Loops
    over Sobieski wrappers."""

    def __init__(self, dtype="float64", **mda_options):
        """Constructor of a MDA using Gauss-Seidel.

        :param dtype: data array type, either "float64" or "complex128".
        :type dtype: str
        :param mda_options: MDA options
        """
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiPropulsion(dtype),
            SobieskiMission(dtype),
        ]
        super(SobieskiMDAGaussSeidel, self).__init__(disciplines, **mda_options)


class SobieskiMDAJacobi(MDAJacobi):
    """Chains Sobieski disciplines to perform and MDA by Jacobi algorithm Loops over
    Sobieski wrappers."""

    def __init__(self, n_processes=1, dtype="float64", **mda_options):
        """Constructor of a MDA using Jacobi.

        :param n_processes: maximum number of processors on which to run
        :type n_processes: integer
        :param dtype: data array type, either "float64" or "complex128".
        :type dtype: str
        :param mda_options: MDA options
        """
        disciplines = [
            SobieskiStructure(dtype),
            SobieskiAerodynamics(dtype),
            SobieskiPropulsion(dtype),
            SobieskiMission(dtype),
        ]
        super(SobieskiMDAJacobi, self).__init__(
            disciplines, n_processes=n_processes, **mda_options
        )
