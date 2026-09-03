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
"""GEMSEO Design of Experiments package.

Contains wrappers to DOE algorithm libraries,
solving an [EvaluationProblem][gemseo.core.problem.evaluation.EvaluationProblem].
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.core.problem.evaluation import EvaluationProblem  # noqa: F401
    from gemseo.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings  # noqa: F401
    from gemseo.doe.diagonal_doe.diagonal_doe import DiagonalDOE_Settings  # noqa: F401
    from gemseo.doe.morris_doe.settings.morris_doe_settings import MorrisDOE_Settings  # noqa: F401
    from gemseo.doe.oat_doe.settings.oat_doe_settings import OATDOE_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_axial import OT_AXIAL_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_composite import OT_COMPOSITE_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_factorial import OT_FACTORIAL_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_faure import OT_FAURE_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_fullfact import OT_FULLFACT_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_halton import OT_HALTON_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_haselgrove import OT_HASELGROVE_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_lhs import OT_LHS_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_lhsc import OT_LHSC_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_monte_carlo import OT_MONTE_CARLO_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_opt_lhs import OT_OPT_LHS_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_random import OT_RANDOM_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_reverse_halton import (
        OT_REVERSE_HALTON_Settings,  # noqa: F401
    )
    from gemseo.doe.openturns.settings.ot_sobol import OT_SOBOL_Settings  # noqa: F401
    from gemseo.doe.openturns.settings.ot_sobol_indices import OT_SOBOL_INDICES_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_bbdesign import PYDOE_BBDESIGN_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_ccdesign import PYDOE_CCDESIGN_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_ff2n import PYDOE_FF2N_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_fullfact import PYDOE_FULLFACT_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings  # noqa: F401
    from gemseo.doe.pydoe.settings.pydoe_pbdesign import PYDOE_PBDESIGN_Settings  # noqa: F401
    from gemseo.doe.scipy.settings.halton import Halton_Settings  # noqa: F401
    from gemseo.doe.scipy.settings.lhs import LHS_Settings  # noqa: F401
    from gemseo.doe.scipy.settings.mc import MC_Settings  # noqa: F401
    from gemseo.doe.scipy.settings.poisson_disk import PoissonDisk_Settings  # noqa: F401
    from gemseo.doe.scipy.settings.sobol import Sobol_Settings  # noqa: F401

# Exported name -> "module.path:Attr" (lazy-loaded on attribute access).
# The module path is relative to ``gemseo.doe``.
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "CustomDOE_Settings": "custom_doe.settings.custom_doe_settings:CustomDOE_Settings",
    "DiagonalDOE_Settings": "diagonal_doe.diagonal_doe:DiagonalDOE_Settings",
    "EvaluationProblem": "gemseo.core.problem.evaluation:EvaluationProblem",
    "Halton_Settings": "scipy.settings.halton:Halton_Settings",
    "LHS_Settings": "scipy.settings.lhs:LHS_Settings",
    "MC_Settings": "scipy.settings.mc:MC_Settings",
    "MorrisDOE_Settings": "morris_doe.settings.morris_doe_settings:MorrisDOE_Settings",
    "OATDOE_Settings": "oat_doe.settings.oat_doe_settings:OATDOE_Settings",
    "OT_AXIAL_Settings": "openturns.settings.ot_axial:OT_AXIAL_Settings",
    "OT_COMPOSITE_Settings": "openturns.settings.ot_composite:OT_COMPOSITE_Settings",
    "OT_FACTORIAL_Settings": "openturns.settings.ot_factorial:OT_FACTORIAL_Settings",
    "OT_FAURE_Settings": "openturns.settings.ot_faure:OT_FAURE_Settings",
    "OT_FULLFACT_Settings": "openturns.settings.ot_fullfact:OT_FULLFACT_Settings",
    "OT_HALTON_Settings": "openturns.settings.ot_halton:OT_HALTON_Settings",
    "OT_HASELGROVE_Settings": "openturns.settings.ot_haselgrove:OT_HASELGROVE_Settings",
    "OT_LHSC_Settings": "openturns.settings.ot_lhsc:OT_LHSC_Settings",
    "OT_LHS_Settings": "openturns.settings.ot_lhs:OT_LHS_Settings",
    "OT_MONTE_CARLO_Settings": (
        "openturns.settings.ot_monte_carlo:OT_MONTE_CARLO_Settings"
    ),
    "OT_OPT_LHS_Settings": "openturns.settings.ot_opt_lhs:OT_OPT_LHS_Settings",
    "OT_RANDOM_Settings": "openturns.settings.ot_random:OT_RANDOM_Settings",
    "OT_REVERSE_HALTON_Settings": (
        "openturns.settings.ot_reverse_halton:OT_REVERSE_HALTON_Settings"
    ),
    "OT_SOBOL_INDICES_Settings": (
        "openturns.settings.ot_sobol_indices:OT_SOBOL_INDICES_Settings"
    ),
    "OT_SOBOL_Settings": "openturns.settings.ot_sobol:OT_SOBOL_Settings",
    "PYDOE_BBDESIGN_Settings": "pydoe.settings.pydoe_bbdesign:PYDOE_BBDESIGN_Settings",
    "PYDOE_CCDESIGN_Settings": "pydoe.settings.pydoe_ccdesign:PYDOE_CCDESIGN_Settings",
    "PYDOE_FF2N_Settings": "pydoe.settings.pydoe_ff2n:PYDOE_FF2N_Settings",
    "PYDOE_FULLFACT_Settings": "pydoe.settings.pydoe_fullfact:PYDOE_FULLFACT_Settings",
    "PYDOE_LHS_Settings": "pydoe.settings.pydoe_lhs:PYDOE_LHS_Settings",
    "PYDOE_PBDESIGN_Settings": "pydoe.settings.pydoe_pbdesign:PYDOE_PBDESIGN_Settings",
    "PoissonDisk_Settings": "scipy.settings.poisson_disk:PoissonDisk_Settings",
    "Sobol_Settings": "scipy.settings.sobol:Sobol_Settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
