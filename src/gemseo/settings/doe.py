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
"""Settings of the DOE algorithms."""

from __future__ import annotations

from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.algos.doe.diagonal_doe.diagonal_doe import DiagonalDOE_Settings
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings
from gemseo.algos.doe.oat_doe.settings.oat_doe_settings import OATDOE_Settings
from gemseo.algos.doe.openturns.settings.ot_axial import OT_AXIAL_Settings
from gemseo.algos.doe.openturns.settings.ot_composite import OT_COMPOSITE_Settings
from gemseo.algos.doe.openturns.settings.ot_factorial import OT_FACTORIAL_Settings
from gemseo.algos.doe.openturns.settings.ot_faure import OT_FAURE_Settings
from gemseo.algos.doe.openturns.settings.ot_fullfact import OT_FULLFACT_Settings
from gemseo.algos.doe.openturns.settings.ot_halton import OT_HALTON_Settings
from gemseo.algos.doe.openturns.settings.ot_haselgrove import OT_HASELGROVE_Settings
from gemseo.algos.doe.openturns.settings.ot_lhs import OT_LHS_Settings
from gemseo.algos.doe.openturns.settings.ot_lhsc import OT_LHSC_Settings
from gemseo.algos.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings
from gemseo.algos.doe.openturns.settings.ot_random import OT_RANDOM_Settings
from gemseo.algos.doe.openturns.settings.ot_reverse_halton import (
    OT_REVERSE_HALTON_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_sobol import OT_SOBOL_Settings
from gemseo.algos.doe.openturns.settings.ot_sobol_indices import (
    OT_SOBOL_INDICES_Settings,
)
from gemseo.algos.doe.pydoe.settings.pydoe_bbdesign import PYDOE_BBDESIGN_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_ccdesign import PYDOE_CCDESIGN_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_ff2n import PYDOE_FF2N_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings
from gemseo.algos.doe.pydoe.settings.pydoe_pbdesign import PYDOE_PBDESIGN_Settings
from gemseo.algos.doe.scipy.settings.halton import Halton_Settings
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings
from gemseo.algos.doe.scipy.settings.mc import MC_Settings
from gemseo.algos.doe.scipy.settings.poisson_disk import PoissonDisk_Settings
from gemseo.algos.doe.scipy.settings.sobol import Sobol_Settings

__all__ = [
    "CustomDOE_Settings",
    "DiagonalDOE_Settings",
    "Halton_Settings",
    "LHS_Settings",
    "MC_Settings",
    "MorrisDOE_Settings",
    "OATDOE_Settings",
    "OT_AXIAL_Settings",
    "OT_COMPOSITE_Settings",
    "OT_FACTORIAL_Settings",
    "OT_FAURE_Settings",
    "OT_FULLFACT_Settings",
    "OT_HALTON_Settings",
    "OT_HASELGROVE_Settings",
    "OT_LHSC_Settings",
    "OT_LHS_Settings",
    "OT_MONTE_CARLO_Settings",
    "OT_OPT_LHS_Settings",
    "OT_RANDOM_Settings",
    "OT_REVERSE_HALTON_Settings",
    "OT_SOBOL_INDICES_Settings",
    "OT_SOBOL_Settings",
    "PYDOE_BBDESIGN_Settings",
    "PYDOE_CCDESIGN_Settings",
    "PYDOE_FF2N_Settings",
    "PYDOE_FULLFACT_Settings",
    "PYDOE_LHS_Settings",
    "PYDOE_PBDESIGN_Settings",
    "PoissonDisk_Settings",
    "Sobol_Settings",
]
