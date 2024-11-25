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
"""Settings for MDAGSNewton."""

from __future__ import annotations

from collections.abc import Sequence  # Noqa: TC003
from typing import ClassVar  # Noqa: TC003

from gemseo.mda.sequential_mda_settings import MDASequential_Settings


class MDAGSNewton_Settings(MDASequential_Settings):  # noqa: N801
    """The settings for :class:`.MDAGSNewton`."""

    _settings_names_to_be_cascaded: ClassVar[Sequence[str]] = [
        "tolerance",
        "max_mda_iter",
        "log_convergence",
        "linear_solver_tolerance",
    ]
    """The settings that must be cascaded to the inner MDAs."""
