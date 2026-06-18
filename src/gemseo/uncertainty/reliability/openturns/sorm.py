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
"""Second-order reliability method (SORM)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import SORM

from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.sorm_settings import OT_SORM_Settings

if TYPE_CHECKING:
    from openturns import SORMResult


class OT_SORM(OT_FORM):  # noqa: N801
    """The second-order reliability method (SORM)."""

    settings_class: ClassVar[type[OT_SORM_Settings]] = OT_SORM_Settings

    _ALGO_CLASS: ClassVar[type[SORM]] = SORM

    @staticmethod
    def _extract_probability(result: SORMResult, settings: OT_SORM_Settings) -> float:
        return getattr(result, f"getEventProbability{settings.approximation}")()
