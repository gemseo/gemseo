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
"""Sobieski's Supersonic Business Jet MDO benchmark."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.problem.mdo.sobieski.discipline import (
        SobieskiAerodynamics,  # noqa: F401
    )
    from gemseo.problem.mdo.sobieski.discipline import SobieskiMission  # noqa: F401
    from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion  # noqa: F401
    from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure  # noqa: F401
    from gemseo.problem.mdo.sobieski.standalone.design_space import (
        SobieskiDesignSpace,  # noqa: F401
    )
    from gemseo.problem.mdo.sobieski.standalone.problem import (
        SobieskiProblem,  # noqa: F401
    )

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "SobieskiAerodynamics": "discipline",
    "SobieskiDesignSpace": "standalone.design_space",
    "SobieskiMission": "discipline",
    "SobieskiProblem": "standalone.problem",
    "SobieskiPropulsion": "discipline",
    "SobieskiStructure": "discipline",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
