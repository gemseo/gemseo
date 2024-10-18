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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Wrappers for SciPy's linear solvers."""

from __future__ import annotations

from gemseo.algos.linear_solvers.scipy_linalg.settings.bicg import (
    BICGSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.bicgstab import (
    BICGStabSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.cg import (
    CGSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.cgs import (
    CGSSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.gcrot import (
    GCROTSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.gmres import (
    GMRESSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.lgmres import (
    LGMRESSettings,  # noqa: F401
)
from gemseo.algos.linear_solvers.scipy_linalg.settings.tfqmr import (
    TFQMRSettings,  # noqa: F401
)
