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
"""First-order reliability method (FORM) with multiple design points."""

from __future__ import annotations

from typing import ClassVar

from openturns import MultiFORM

from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.multi_form_settings import (
    OT_MultiFORM_Settings,
)


class OT_MultiFORM(OT_FORM):  # noqa: N801
    """The first-order reliability method (FORM) with multiple design points."""

    settings_class: ClassVar[type[OT_MultiFORM_Settings]] = OT_MultiFORM_Settings

    _ALGO_CLASS: ClassVar[type[MultiFORM]] = MultiFORM

    _USE_MULTIFORM_RESULT: ClassVar[bool] = True

    @staticmethod
    def _set_algo_options(algo: MultiFORM, settings: OT_MultiFORM_Settings) -> None:
        algo.setMaximumDesignPointsNumber(settings.max_design_points)
