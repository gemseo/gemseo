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

from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import (  # noqa: F401
    CustomDOE_Settings,
)
from gemseo.algos.doe.diagonal_doe.diagonal_doe import (  # noqa: F401
    DiagonalDOE_Settings,
)
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import (  # noqa: F401
    MorrisDOE_Settings,
)
from gemseo.algos.doe.oat_doe.settings.oat_doe_settings import (  # noqa: F401
    OATDOE_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_axial import OT_AXIAL_Settings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_composite import (  # noqa: F401
    OT_COMPOSITE_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_factorial import (  # noqa: F401
    OT_FACTORIAL_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_faure import OT_FAURE_Settings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_fullfact import (  # noqa: F401
    OT_FULLFACT_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_halton import (  # noqa: F401
    OT_HALTON_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_haselgrove import (  # noqa: F401
    OT_HASELGROVE_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_lhs import OT_LHS_Settings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_lhsc import OT_LHSC_Settings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_monte_carlo import (  # noqa: F401
    OT_MONTE_CARLO_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import (  # noqa: F401
    OT_OPT_LHS_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_random import (  # noqa: F401
    OT_RANDOM_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_reverse_halton import (  # noqa: F401
    OT_REVERSE_HALTON_Settings,
)
from gemseo.algos.doe.openturns.settings.ot_sobol import OT_SOBOL_Settings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_sobol_indices import (  # noqa: F401
    OT_SOBOL_INDICES_Settings,
)
from gemseo.algos.doe.pydoe.settings.pydoe_bbdesign import (  # noqa: F401
    PYDOE_BBDESIGN_Settings,
)
from gemseo.algos.doe.pydoe.settings.pydoe_ccdesign import (  # noqa: F401
    PYDOE_CCDESIGN_Settings,
)
from gemseo.algos.doe.pydoe.settings.pydoe_ff2n import PYDOE_FF2N_Settings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.pydoe_fullfact import (  # noqa: F401
    PYDOE_FULLFACT_Settings,
)
from gemseo.algos.doe.pydoe.settings.pydoe_lhs import PYDOE_LHS_Settings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.pydoe_pbdesign import (  # noqa: F401
    PYDOE_PBDESIGN_Settings,
)
from gemseo.algos.doe.scipy.settings.halton import Halton_Settings  # noqa: F401
from gemseo.algos.doe.scipy.settings.lhs import LHS_Settings  # noqa: F401
from gemseo.algos.doe.scipy.settings.mc import MC_Settings  # noqa: F401
from gemseo.algos.doe.scipy.settings.poisson_disk import (  # noqa: F401
    PoissonDisk_Settings,
)
from gemseo.algos.doe.scipy.settings.sobol import Sobol_Settings  # noqa: F401
