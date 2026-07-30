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
"""The linear solvers.

Together with the problem they solve
([LinearProblem][gemseo.linear.problem.LinearProblem]).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.linear.problem import LinearProblem  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.bicg import BICG_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.bicgstab import (  # noqa: F401
        BICGSTAB_Settings,
    )
    from gemseo.linear.scipy_linalg.settings.cg import CG_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.cgs import CGS_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.gcrot import GCROT_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.gmres import GMRES_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.lgmres import LGMRES_Settings  # noqa: F401
    from gemseo.linear.scipy_linalg.settings.tfqmr import TFQMR_Settings  # noqa: F401

# Exported name -> "module.path:Attr" (lazy-loaded on attribute access).
# The module path is relative to this package.
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "BICGSTAB_Settings": "scipy_linalg.settings.bicgstab:BICGSTAB_Settings",
    "BICG_Settings": "scipy_linalg.settings.bicg:BICG_Settings",
    "CGS_Settings": "scipy_linalg.settings.cgs:CGS_Settings",
    "CG_Settings": "scipy_linalg.settings.cg:CG_Settings",
    "GCROT_Settings": "scipy_linalg.settings.gcrot:GCROT_Settings",
    "GMRES_Settings": "scipy_linalg.settings.gmres:GMRES_Settings",
    "LGMRES_Settings": "scipy_linalg.settings.lgmres:LGMRES_Settings",
    "LinearProblem": "problem:LinearProblem",
    "TFQMR_Settings": "scipy_linalg.settings.tfqmr:TFQMR_Settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
