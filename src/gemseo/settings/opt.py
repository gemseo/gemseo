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
"""Settings of the optimization algorithms."""

from __future__ import annotations

from gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_0_settings import (  # noqa: E501 F401
    Augmented_Lagrangian_order_0_Settings,
)
from gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_1_settings import (  # noqa:  E501 F401
    Augmented_Lagrangian_order_1_Settings,
)
from gemseo.algos.opt.augmented_lagrangian.settings.penalty_heuristic_settings import (
    PenaltyHeuristicSettings,
)
from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBI_Settings
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (
    MultiStart_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_bfgs_settings import (  # noqa:F401
    NLOPT_BFGS_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_bobyqa_settings import (  # noqa:F401
    NLOPT_BOBYQA_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import (  # noqa:F401
    NLOPT_COBYLA_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_mma_settings import (  # noqa:F401
    NLOPT_MMA_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_newuoa_settings import (  # noqa:F401
    NLOPT_NEWUOA_Settings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_slsqp_settings import (  # noqa:F401
    NLOPT_SLSQP_Settings,
)
from gemseo.algos.opt.scipy_global.settings.differential_evolution import (
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.algos.opt.scipy_global.settings.dual_annealing import (
    DUAL_ANNEALING_Settings,
)
from gemseo.algos.opt.scipy_global.settings.shgo import SHGO_Settings
from gemseo.algos.opt.scipy_linprog.settings.highs_dual_simplex import (
    DUAL_SIMPLEX_Settings,
)
from gemseo.algos.opt.scipy_linprog.settings.highs_interior_point import (
    INTERIOR_POINT_Settings,
)
from gemseo.algos.opt.scipy_local.settings.cobyqa import COBYQA_Settings
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings
from gemseo.algos.opt.scipy_local.settings.nelder_mead import NELDER_MEAD_Settings
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.algos.opt.scipy_local.settings.tnc import TNC_Settings
from gemseo.algos.opt.scipy_milp.settings.scipy_milp_settings import SciPyMILP_Settings

__all__ = [
    "Augmented_Lagrangian_order_0_Settings",
    "Augmented_Lagrangian_order_1_Settings",
    "COBYQA_Settings",
    "DIFFERENTIAL_EVOLUTION_Settings",
    "DUAL_ANNEALING_Settings",
    "DUAL_SIMPLEX_Settings",
    "INTERIOR_POINT_Settings",
    "L_BFGS_B_Settings",
    "MNBI_Settings",
    "MultiStart_Settings",
    "NELDER_MEAD_Settings",
    "NLOPT_BFGS_Settings",
    "NLOPT_BOBYQA_Settings",
    "NLOPT_COBYLA_Settings",
    "NLOPT_MMA_Settings",
    "NLOPT_NEWUOA_Settings",
    "NLOPT_SLSQP_Settings",
    "PenaltyHeuristicSettings",
    "SHGO_Settings",
    "SLSQP_Settings",
    "SciPyMILP_Settings",
    "TNC_Settings",
]
