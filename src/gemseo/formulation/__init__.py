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
"""MDO formulations or architectures."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.formulation.bilevel_bcd_settings import BiLevelBCD_Settings  # noqa: F401
    from gemseo.formulation.bilevel_settings import BiLevel_Settings  # noqa: F401
    from gemseo.formulation.disciplinary_opt_settings import DisciplinaryOpt_Settings  # noqa: F401
    from gemseo.formulation.idf_settings import IDF_Settings  # noqa: F401
    from gemseo.formulation.mdf_settings import MDF_Settings  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "BiLevelBCD_Settings": "bilevel_bcd_settings",
    "BiLevel_Settings": "bilevel_settings",
    "DisciplinaryOpt_Settings": "disciplinary_opt_settings",
    "IDF_Settings": "idf_settings",
    "MDF_Settings": "mdf_settings",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
