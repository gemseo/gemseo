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
"""The local minimization methods of the SciPy basin-hopping algorithm."""

from __future__ import annotations

from enum import StrEnum


# The methods of scipy.optimize.minimize handling bounds,
# as the bounds of the design space are always passed to the local method.
class BasinHoppingLocalMethod(StrEnum):
    """The local minimization methods of basin-hopping."""

    COBYLA = "COBYLA"
    """The COBYLA method."""

    COBYQA = "COBYQA"
    """The COBYQA method."""

    L_BFGS_B = "L-BFGS-B"
    """The L-BFGS-B method."""

    NELDER_MEAD = "Nelder-Mead"
    """The Nelder-Mead method."""

    POWELL = "Powell"
    """The Powell method."""

    SLSQP = "SLSQP"
    """The SLSQP method."""

    TNC = "TNC"
    """The TNC method."""

    TRUST_CONSTR = "trust-constr"
    """The trust-constr method."""
