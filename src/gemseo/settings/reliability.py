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
"""Settings for the reliability algorithms."""

from __future__ import annotations

from gemseo.uncertainty.reliability.openturns.directional_sampling_settings import (
    OT_DirectionalSampling_Settings,
)
from gemseo.uncertainty.reliability.openturns.faure_settings import OT_Faure_Settings
from gemseo.uncertainty.reliability.openturns.form_settings import OT_FORM_Settings
from gemseo.uncertainty.reliability.openturns.halton_settings import OT_Halton_Settings
from gemseo.uncertainty.reliability.openturns.haselgrove_settings import (
    OT_Haselgrove_Settings,
)
from gemseo.uncertainty.reliability.openturns.is_form_settings import (
    OT_IS_FORM_Settings,
)
from gemseo.uncertainty.reliability.openturns.is_na_settings import OT_IS_NA_Settings
from gemseo.uncertainty.reliability.openturns.is_spce_settings import (
    OT_IS_SPCE_Settings,
)
from gemseo.uncertainty.reliability.openturns.lhs_settings import OT_LHS_Settings
from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings
from gemseo.uncertainty.reliability.openturns.multi_form_settings import (
    OT_MultiFORM_Settings,
)
from gemseo.uncertainty.reliability.openturns.reverse_halton_settings import (
    OT_Reverse_Halton_Settings,
)
from gemseo.uncertainty.reliability.openturns.sobol_settings import OT_Sobol_Settings
from gemseo.uncertainty.reliability.openturns.sorm_settings import OT_SORM_Settings
from gemseo.uncertainty.reliability.openturns.subset_sampling_settings import (
    OT_SubsetSampling_Settings,
)
from gemseo.uncertainty.reliability.openturns.system_form_settings import (
    OT_SystemFORM_Settings,
)

__all__ = [
    "OT_DirectionalSampling_Settings",
    "OT_FORM_Settings",
    "OT_Faure_Settings",
    "OT_Halton_Settings",
    "OT_Haselgrove_Settings",
    "OT_IS_FORM_Settings",
    "OT_IS_NA_Settings",
    "OT_IS_SPCE_Settings",
    "OT_LHS_Settings",
    "OT_MC_Settings",
    "OT_MultiFORM_Settings",
    "OT_Reverse_Halton_Settings",
    "OT_SORM_Settings",
    "OT_Sobol_Settings",
    "OT_SubsetSampling_Settings",
    "OT_SystemFORM_Settings",
]
