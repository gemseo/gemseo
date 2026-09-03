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
"""GEMSEO optimization algorithm package.

Contains wrappers to algorithm libraries,
together with the problem they solve
([OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.optimization.augmented_lagrangian.settings.order_0 import (  # noqa: F401
        Augmented_Lagrangian_Order_0_Settings,
    )
    from gemseo.optimization.augmented_lagrangian.settings.order_1 import (  # noqa: F401
        Augmented_Lagrangian_Order_1_Settings,
    )
    from gemseo.optimization.mnbi.settings.mnbi_settings import MNBI_Settings  # noqa: F401
    from gemseo.optimization.multi_start.settings.multi_start_settings import (  # noqa: F401
        MultiStart_Settings,
    )
    from gemseo.optimization.nlopt.settings.nlopt_bfgs_settings import (  # noqa: F401
        NLOPT_BFGS_Settings,
    )
    from gemseo.optimization.nlopt.settings.nlopt_bobyqa_settings import (  # noqa: F401
        NLOPT_BOBYQA_Settings,
    )
    from gemseo.optimization.nlopt.settings.nlopt_cobyla_settings import (  # noqa: F401
        NLOPT_COBYLA_Settings,
    )
    from gemseo.optimization.nlopt.settings.nlopt_mma_settings import NLOPT_MMA_Settings  # noqa: F401
    from gemseo.optimization.nlopt.settings.nlopt_newuoa_settings import (  # noqa: F401
        NLOPT_NEWUOA_Settings,
    )
    from gemseo.optimization.nlopt.settings.nlopt_slsqp_settings import (  # noqa: F401
        NLOPT_SLSQP_Settings,
    )
    from gemseo.optimization.problem import OptimizationProblem  # noqa: F401
    from gemseo.optimization.scipy_global.settings.differential_evolution import (  # noqa: F401
        DIFFERENTIAL_EVOLUTION_Settings,
    )
    from gemseo.optimization.scipy_global.settings.dual_annealing import (  # noqa: F401
        DUAL_ANNEALING_Settings,
    )
    from gemseo.optimization.scipy_global.settings.shgo import SHGO_Settings  # noqa: F401
    from gemseo.optimization.scipy_linprog.settings.highs_dual_simplex import (  # noqa: F401
        DUAL_SIMPLEX_Settings,
    )
    from gemseo.optimization.scipy_linprog.settings.highs_interior_point import (  # noqa: F401
        INTERIOR_POINT_Settings,
    )
    from gemseo.optimization.scipy_local.settings.cobyla import COBYLA_Settings  # noqa: F401
    from gemseo.optimization.scipy_local.settings.cobyqa import COBYQA_Settings  # noqa: F401
    from gemseo.optimization.scipy_local.settings.lbfgsb import L_BFGS_B_Settings  # noqa: F401
    from gemseo.optimization.scipy_local.settings.nelder_mead import (  # noqa: F401
        NELDER_MEAD_Settings,
    )
    from gemseo.optimization.scipy_local.settings.slsqp import SLSQP_Settings  # noqa: F401
    from gemseo.optimization.scipy_local.settings.tnc import TNC_Settings  # noqa: F401
    from gemseo.optimization.scipy_milp.settings.scipy_milp_settings import (  # noqa: F401
        MILP_Settings,
    )

# Exported name -> "module.path:Attr" (lazy-loaded on attribute access).
# The module path is relative to this package.
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "Augmented_Lagrangian_Order_0_Settings": (
        "augmented_lagrangian.settings.order_0:Augmented_Lagrangian_Order_0_Settings"
    ),
    "Augmented_Lagrangian_Order_1_Settings": (
        "augmented_lagrangian.settings.order_1:Augmented_Lagrangian_Order_1_Settings"
    ),
    "COBYLA_Settings": "scipy_local.settings.cobyla:COBYLA_Settings",
    "COBYQA_Settings": "scipy_local.settings.cobyqa:COBYQA_Settings",
    "DIFFERENTIAL_EVOLUTION_Settings": (
        "scipy_global.settings.differential_evolution:DIFFERENTIAL_EVOLUTION_Settings"
    ),
    "DUAL_ANNEALING_Settings": (
        "scipy_global.settings.dual_annealing:DUAL_ANNEALING_Settings"
    ),
    "DUAL_SIMPLEX_Settings": (
        "scipy_linprog.settings.highs_dual_simplex:DUAL_SIMPLEX_Settings"
    ),
    "INTERIOR_POINT_Settings": (
        "scipy_linprog.settings.highs_interior_point:INTERIOR_POINT_Settings"
    ),
    "L_BFGS_B_Settings": "scipy_local.settings.lbfgsb:L_BFGS_B_Settings",
    "MILP_Settings": "scipy_milp.settings.scipy_milp_settings:MILP_Settings",
    "MNBI_Settings": "mnbi.settings.mnbi_settings:MNBI_Settings",
    "MultiStart_Settings": (
        "multi_start.settings.multi_start_settings:MultiStart_Settings"
    ),
    "NELDER_MEAD_Settings": "scipy_local.settings.nelder_mead:NELDER_MEAD_Settings",
    "NLOPT_BFGS_Settings": "nlopt.settings.nlopt_bfgs_settings:NLOPT_BFGS_Settings",
    "NLOPT_BOBYQA_Settings": (
        "nlopt.settings.nlopt_bobyqa_settings:NLOPT_BOBYQA_Settings"
    ),
    "NLOPT_COBYLA_Settings": (
        "nlopt.settings.nlopt_cobyla_settings:NLOPT_COBYLA_Settings"
    ),
    "NLOPT_MMA_Settings": "nlopt.settings.nlopt_mma_settings:NLOPT_MMA_Settings",
    "NLOPT_NEWUOA_Settings": (
        "nlopt.settings.nlopt_newuoa_settings:NLOPT_NEWUOA_Settings"
    ),
    "NLOPT_SLSQP_Settings": (
        "nlopt.settings.nlopt_slsqp_settings:NLOPT_SLSQP_Settings"
    ),
    "OptimizationProblem": "problem:OptimizationProblem",
    "SHGO_Settings": "scipy_global.settings.shgo:SHGO_Settings",
    "SLSQP_Settings": "scipy_local.settings.slsqp:SLSQP_Settings",
    "TNC_Settings": "scipy_local.settings.tnc:TNC_Settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
