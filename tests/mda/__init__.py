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
from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.discipline.base_discipline import CacheType

if TYPE_CHECKING:
    from gemseo.mda.base_mda_solver import BaseMDASolver


def check_iteration_callbacks_execution(mda_solver: BaseMDASolver) -> None:
    """Check the iteration callbacks of an MDA solver.

    Args:
        mda_solver: The MDA solver.
    """
    residuals = []

    def iteration_callback(mda: BaseMDASolver) -> None:
        """Store the current residual of an MDA.

        Args:
            mda: The MDA.
        """
        residuals.append(mda.normed_residual)

    mda_solver.add_iteration_callback(iteration_callback)
    mda_solver.execute()
    residual_history = mda_solver.residual_history
    assert residuals == residual_history


def check_iteration_callbacks_clearing(mda_solver: BaseMDASolver) -> None:
    """Check the clearing of the iteration callbacks of an MDA solver.

    Args:
        mda_solver: The MDA solver.
    """
    mda_solver.set_cache(CacheType.NONE)
    residuals = []
    mda_solver.add_iteration_callback(lambda mda: residuals.append(mda.normed_residual))
    mda_solver.execute()
    assert residuals

    mda_solver.clear_iteration_callbacks()
    residuals = []
    mda_solver.execute()
    assert not residuals
