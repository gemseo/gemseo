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
"""Settings for the augmented Lagrangian of order 0 algorithm."""

from __future__ import annotations

from gemseo.algos.opt.augmented_lagrangian.settings.penalty_heuristic_settings import (
    PenaltyHeuristicSettings,
)


class Augmented_Lagrangian_order_0_Settings(PenaltyHeuristicSettings):  # noqa: N801
    """The augmented Lagrangian of order 0 settings."""

    _TARGET_CLASS_NAME = "Augmented_Lagrangian_order_0"
