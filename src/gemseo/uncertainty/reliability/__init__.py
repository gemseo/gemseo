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
"""Reliability analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.uncertainty.reliability.factory import (
        RELIABILITY_ALGORITHM_FACTORY,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.directional_sampling_settings import (
        OT_DirectionalSampling_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.faure_settings import (
        OT_Faure_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.form_settings import (
        OT_FORM_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.halton_settings import (
        OT_Halton_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.haselgrove_settings import (
        OT_Haselgrove_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.is_form_settings import (
        OT_IS_FORM_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.is_na_settings import (
        OT_IS_NA_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.is_spce_settings import (
        OT_IS_SPCE_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.lhs_settings import (
        OT_LHS_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.mc_settings import (
        OT_MC_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.multi_form_settings import (
        OT_MultiFORM_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.reverse_halton_settings import (
        OT_Reverse_Halton_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.sobol_settings import (
        OT_Sobol_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.sorm_settings import (
        OT_SORM_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.subset_sampling_settings import (
        OT_SubsetSampling_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.openturns.system_form_settings import (
        OT_SystemFORM_Settings,  # noqa: F401
    )
    from gemseo.uncertainty.reliability.problem import ReliabilityProblem  # noqa: F401
    from gemseo.uncertainty.reliability.scenario import (
        ReliabilityScenario,  # noqa: F401
    )


# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "OT_DirectionalSampling_Settings": "openturns.directional_sampling_settings",
    "OT_FORM_Settings": "openturns.form_settings",
    "OT_Faure_Settings": "openturns.faure_settings",
    "OT_Halton_Settings": "openturns.halton_settings",
    "OT_Haselgrove_Settings": "openturns.haselgrove_settings",
    "OT_IS_FORM_Settings": "openturns.is_form_settings",
    "OT_IS_NA_Settings": "openturns.is_na_settings",
    "OT_IS_SPCE_Settings": "openturns.is_spce_settings",
    "OT_LHS_Settings": "openturns.lhs_settings",
    "OT_MC_Settings": "openturns.mc_settings",
    "OT_MultiFORM_Settings": "openturns.multi_form_settings",
    "OT_Reverse_Halton_Settings": "openturns.reverse_halton_settings",
    "OT_SORM_Settings": "openturns.sorm_settings",
    "OT_Sobol_Settings": "openturns.sobol_settings",
    "OT_SubsetSampling_Settings": "openturns.subset_sampling_settings",
    "OT_SystemFORM_Settings": "openturns.system_form_settings",
    "RELIABILITY_ALGORITHM_FACTORY": "factory",
    "ReliabilityProblem": "problem",
    "ReliabilityScenario": "scenario",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
