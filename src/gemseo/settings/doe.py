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
    CustomDOESettings,
)
from gemseo.algos.doe.diagonal_doe.diagonal_doe import DiagonalDOESettings  # noqa: F401
from gemseo.algos.doe.morris_doe.settings.morris_doe_settings import (  # noqa: F401
    MorrisDOESettings,
)
from gemseo.algos.doe.oat_doe.settings.oat_doe_settings import (  # noqa: F401
    OATDOESettings,
)
from gemseo.algos.doe.openturns.settings.ot_axial import OTAxialSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_composite import (  # noqa: F401
    OTCompositeSettings,
)
from gemseo.algos.doe.openturns.settings.ot_factorial import (  # noqa: F401
    OTFactorialSettings,
)
from gemseo.algos.doe.openturns.settings.ot_faure import OTFaureSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_full_fact import (  # noqa: F401
    OTFullFactSettings,
)
from gemseo.algos.doe.openturns.settings.ot_halton import OTHaltonSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_haselgrove import (  # noqa: F401
    OTHaselgroveSettings,
)
from gemseo.algos.doe.openturns.settings.ot_lhs import OTLHSSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_lhsc import OTLHSCSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_monte_carlo import (  # noqa: F401
    OTMonteCarloSettings,
)
from gemseo.algos.doe.openturns.settings.ot_opt_lhs import (  # noqa: F401
    OTOptLHSSettings,
)
from gemseo.algos.doe.openturns.settings.ot_random import OTRandomSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_reverse_halton import (  # noqa: F401
    OTReverseHaltonSettings,
)
from gemseo.algos.doe.openturns.settings.ot_sobol import OTSobolSettings  # noqa: F401
from gemseo.algos.doe.openturns.settings.ot_sobol_indices import (  # noqa: F401
    OTSobolIndicesSettings,
)
from gemseo.algos.doe.pydoe.settings.bbdesign import BBDesignSettings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.ccdesign import CCDesignSettings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.ff2n import FF2NSettings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.fullfact import FullFactSettings  # noqa: F401
from gemseo.algos.doe.pydoe.settings.pbdesign import PBDesignSettings  # noqa: F401
from gemseo.algos.doe.scipy.settings.halton import HaltonSettings  # noqa: F401
from gemseo.algos.doe.scipy.settings.lhs import LHSSettings  # noqa: F401
from gemseo.algos.doe.scipy.settings.mc import MCSettings  # noqa: F401
from gemseo.algos.doe.scipy.settings.poisson_disk import (  # noqa: F401
    PoissonDiskSettings,
)
from gemseo.algos.doe.scipy.settings.sobol import SobolSettings  # noqa: F401
