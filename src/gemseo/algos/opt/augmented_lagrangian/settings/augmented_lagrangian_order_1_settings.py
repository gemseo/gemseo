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
"""Settings for the augmented Lagrangian of order 1 algorithm."""

from __future__ import annotations

from gemseo.algos.opt.augmented_lagrangian.settings.penalty_heuristic_settings import (
    PenaltyHeuristicSettings,
)
from gemseo.algos.opt.base_gradient_based_algorithm_settings import (
    BaseGradientBasedAlgorithmSettings,
)


class Augmented_Lagrangian_order_1_Settings(  # noqa: N801
    PenaltyHeuristicSettings, BaseGradientBasedAlgorithmSettings
):
    """The augmented Lagrangian of order 1 settings."""

    _TARGET_CLASS_NAME = "Augmented_Lagrangian_order_1"
