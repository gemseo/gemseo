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
from gemseo.algos.opt.augmented_lagrangian.settings.penalty_heuristic_settings import (  # noqa: F401
    PenaltyHeuristicSettings,
)
from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBI_Settings  # noqa: F401
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (  # noqa: F401
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
from gemseo.algos.opt.scipy_global.settings.differential_evolution import (  # noqa: F401
    DIFFERENTIAL_EVOLUTION_Settings,
)
from gemseo.algos.opt.scipy_global.settings.dual_annealing import (  # noqa: F401
    DUAL_ANNEALING_Settings,
)
from gemseo.algos.opt.scipy_global.settings.shgo import SHGO_Settings  # noqa: F401
from gemseo.algos.opt.scipy_linprog.settings.highs_dual_simplex import (  # noqa: F401
    DUAL_SIMPLEX_Settings,
)
from gemseo.algos.opt.scipy_linprog.settings.highs_interior_point import (  # noqa: F401
    INTERIOR_POINT_Settings,
)
from gemseo.algos.opt.scipy_local.settings.cobyqa import COBYQA_Settings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.lbfgsb import L_BFGS_B_Settings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.nelder_mead import (  # noqa: F401
    NELDER_MEAD_Settings,
)
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.tnc import TNC_Settings  # noqa: F401
from gemseo.algos.opt.scipy_milp.settings.scipy_milp_settings import (  # noqa: F401
    SciPyMILP_Settings,
)
