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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A chain of MDAs to build hybrids of MDA algorithms sequentially."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.mda.gs_newton_settings import MDAGSNewton_Settings
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.mda.newton_raphson_settings import MDANewtonRaphson_Settings
from gemseo.mda.sequential_mda import MDASequential
from gemseo.utils.pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline


class MDAGSNewton(MDASequential):
    """Perform some Gauss-Seidel iterations and then Newton-Raphson iterations."""

    Settings: ClassVar[type[MDAGSNewton_Settings]] = MDAGSNewton_Settings
    """The pydantic model for the settings."""

    settings: MDAGSNewton_Settings
    """The settings of the MDA"""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDAGSNewton_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            mda_sequence=[],
            settings_model=settings_model,
            **settings,
        )

        cs = {"coupling_structure": self.coupling_structure}

        gs_settings = dict(self.settings.gauss_seidel_settings) | cs
        gs_settings = create_model(MDAGaussSeidel_Settings, **gs_settings)

        nr_settings = dict(self.settings.newton_settings) | cs
        nr_settings = create_model(MDANewtonRaphson_Settings, **nr_settings)

        self.mda_sequence = [
            MDAGaussSeidel(disciplines, settings_model=gs_settings),
            MDANewtonRaphson(disciplines, settings_model=nr_settings),
        ]
        self.settings._sub_mdas = self.mda_sequence
