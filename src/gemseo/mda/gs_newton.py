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

from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mda.gauss_seidel import MDAGaussSeidel
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidelSettings
from gemseo.mda.gs_newton_settings import MDAGSNewtonSettings
from gemseo.mda.newton_raphson import MDANewtonRaphson
from gemseo.mda.newton_raphson_settings import MDANewtonRaphsonSettings
from gemseo.mda.sequential_mda import MDASequential
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.mda.sequential_mda_settings import MDASequentialSettings
    from gemseo.typing import StrKeyMapping


class MDAGSNewton(MDASequential):
    """Perform some Gauss-Seidel iterations and then Newton-Raphson iterations."""

    Settings: ClassVar[type[MDAGSNewtonSettings]] = MDAGSNewtonSettings
    """The pydantic model for the settings."""

    settings: MDAGSNewtonSettings
    """The settings of the MDA"""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        gauss_seidel_settings: MDAGaussSeidelSettings
        | StrKeyMapping = READ_ONLY_EMPTY_DICT,
        newton_settings: MDANewtonRaphsonSettings
        | StrKeyMapping = READ_ONLY_EMPTY_DICT,
        settings_model: MDASequentialSettings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            mda_sequence=[],
            settings_model=settings_model,
            **settings,
        )

        gauss_seidel_settings = create_model(
            MDAGaussSeidelSettings,
            **self.__update_inner_mda_settings(gauss_seidel_settings),
        )
        newton_settings = create_model(
            MDANewtonRaphsonSettings,
            **self.__update_inner_mda_settings(newton_settings),
        )

        mda_sequence = [
            MDAGaussSeidel(disciplines, settings_model=gauss_seidel_settings),
            MDANewtonRaphson(disciplines, settings_model=newton_settings),
        ]

        self._init_mda_sequence(mda_sequence)

    def __update_inner_mda_settings(
        self, settings_model: BaseMDASettings | StrKeyMapping
    ) -> StrKeyMapping:
        """Update the inner MDA settings model."""
        return dict(settings_model) | {
            name: setting
            for name, setting in self.settings
            if name in BaseMDASettings.model_fields
        }
