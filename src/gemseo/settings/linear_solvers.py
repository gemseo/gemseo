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
"""Settings of the linear solvers."""

from __future__ import annotations

from gemseo.algos.linear_solvers.scipy_linalg.settings.bicg import (  # noqa: F401
    BICGSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.bicgstab import (  # noqa: F401
    BICGStabSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.cg import (  # noqa: F401
    CGSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.cgs import (  # noqa: F401
    CGSSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.gcrot import (  # noqa: F401
    GCROTSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.gmres import (  # noqa: F401
    GMRESSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.lgmres import (  # noqa: F401
    LGMRESSettings,
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.tfqmr import (  # noqa: F401
    TFQMRSettings,
)
