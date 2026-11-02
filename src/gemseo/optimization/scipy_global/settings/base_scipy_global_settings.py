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
"""Settings for the global optimization algorithms from SciPy."""

from __future__ import annotations

from sys import maxsize
from typing import ClassVar

from gemseo.optimization.core.base_optimizer_settings import BaseOptimizerSettings
from gemseo.util.typing import StrKeyMapping


class BaseSciPyGlobalSettings(BaseOptimizerSettings):
    """The SciPy global optimization library setting."""

    _inherited_field_defaults: ClassVar[StrKeyMapping] = {
        "eq_tolerance": 1e-6,
        "ftol_rel": 1e-9,
        "ftol_abs": 1e-9,
        "max_iter": maxsize,
        "xtol_rel": 1e-9,
        "xtol_abs": 1e-9,
    }


def _check_database_with_processes(use_database: bool, n_processes: int) -> None:
    """Check that the database is used when there are several processes.

    The worker processes send their evaluations back through the database;
    without it, each point would be evaluated twice,
    once in a worker and once in the main process.

    Args:
        use_database: Whether the database is used.
        n_processes: The number of processes, -1 for all the CPUs.

    Raises:
        ValueError: When `use_database` is `False` and `n_processes` is not 1.
    """
    if not use_database and n_processes != 1:
        names = "all the CPUs" if n_processes == -1 else f"{n_processes} workers"
        msg = (
            f"The workers cannot be used with {names} when use_database is False; "
            "use 1 worker or set use_database to True."
        )
        raise ValueError(msg)
