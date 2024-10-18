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
    AugmentedLagrangianOrder0Settings,
)
from gemseo.algos.opt.augmented_lagrangian.settings.augmented_lagrangian_order_1_settings import (  # noqa: E501 F401
    AugmentedLagrangianOrder1Settings,
)
from gemseo.algos.opt.augmented_lagrangian.settings.penalty_heuristic_settings import (  # noqa: F401
    PenaltyHeuristicSettings,
)
from gemseo.algos.opt.mnbi.settings.mnbi_settings import MNBISettings  # noqa: F401
from gemseo.algos.opt.multi_start.settings.multi_start_settings import (  # noqa: F401
    MultiStartSettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_bfgs_settings import (  # noqa:F401
    NLOPTBFGSSettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_bobyqa_settings import (  # noqa:F401
    NLOPTBOBYQASettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_cobyla_settings import (  # noqa:F401
    NLOPTCOBYLASettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_mma_settings import (  # noqa:F401
    NLOPTMMASettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_newuoa_settings import (  # noqa:F401
    NLOPTNEWUOASettings,
)
from gemseo.algos.opt.nlopt.settings.nlopt_slsqp_settings import (  # noqa:F401
    NLOPTSLSQPSettings,
)
from gemseo.algos.opt.scipy_global.settings.differential_evolution import (  # noqa: F401
    DifferentialEvolutionSettings,
)
from gemseo.algos.opt.scipy_global.settings.dual_annealing import (  # noqa: F401
    DualAnnealingSettings,
)
from gemseo.algos.opt.scipy_global.settings.shgo import SHGOSettings  # noqa: F401
from gemseo.algos.opt.scipy_linprog.settings.highs_dual_simplex import (  # noqa: F401
    HiGHSDualSimplexSettings,
)
from gemseo.algos.opt.scipy_linprog.settings.highs_interior_point import (  # noqa: F401
    HiGHSInteriorPointSettings,
)
from gemseo.algos.opt.scipy_local.settings.cobyqa import COBYQASettings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.lbfgsb import LBFGSBSettings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.nelder_mead import (  # noqa: F401
    NelderMeadSettings,
)
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQPSettings  # noqa: F401
from gemseo.algos.opt.scipy_local.settings.tnc import TNCSettings  # noqa: F401
from gemseo.algos.opt.scipy_milp.settings.scipy_milp_settings import (  # noqa: F401
    SciPyMILPSettings,  # noqa: F401
)
