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
"""Quality assessment for regressors."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.machine_learning.regression.quality.factory import (
        REGRESSOR_QUALITY_FACTORY,  # noqa: F401
    )
    from gemseo.machine_learning.regression.quality.mae_measure import (
        MAEMeasure,  # noqa: F401
    )
    from gemseo.machine_learning.regression.quality.me_measure import (
        MEMeasure,  # noqa: F401
    )
    from gemseo.machine_learning.regression.quality.mse_measure import (
        MSEMeasure,  # noqa: F401
    )
    from gemseo.machine_learning.regression.quality.r2_measure import (
        R2Measure,  # noqa: F401
    )
    from gemseo.machine_learning.regression.quality.rmse_measure import (
        RMSEMeasure,  # noqa: F401
    )

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "MAEMeasure": "mae_measure",
    "MEMeasure": "me_measure",
    "MSEMeasure": "mse_measure",
    "R2Measure": "r2_measure",
    "RMSEMeasure": "rmse_measure",
    "REGRESSOR_QUALITY_FACTORY": "factory",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
