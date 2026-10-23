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
"""Utils for the customizable Sellar MDO problem."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import atleast_2d
from numpy import float64
from numpy import ones
from numpy import zeros

from gemseo.problem.mdo.sellar import with_2d_array
from gemseo.problem.mdo.sellar.variable import alpha
from gemseo.problem.mdo.sellar.variable import beta
from gemseo.problem.mdo.sellar.variable import gamma
from gemseo.problem.mdo.sellar.variable import x_1
from gemseo.problem.mdo.sellar.variable import x_2
from gemseo.problem.mdo.sellar.variable import x_shared
from gemseo.problem.mdo.sellar.variable import y_1
from gemseo.problem.mdo.sellar.variable import y_2

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.mda.core.base import BaseMDA
    from gemseo.util.typing import RealArray


def get_initial_data(names: Iterable[str] = (), n: int = 1) -> dict[str, RealArray]:
    """Generate an initial solution for the MDO problem.

    Args:
        names: The names of the discipline inputs.
        n: The size of the local design variables and coupling variables

    Returns:
        The default values of the discipline inputs.
    """
    inputs = {
        x_1: zeros(n),
        x_2: zeros(n),
        x_shared: array([1.0, 0.0], dtype=float64),
        y_1: ones(n, dtype=float64),
        y_2: ones(n, dtype=float64),
        alpha: array([3.16]),
        beta: array([24.0]),
        gamma: array([0.2]),
    }
    if with_2d_array:  # pragma: no cover
        inputs[x_shared] = atleast_2d(inputs[x_shared])
    if not names:
        return inputs
    return {name: inputs[name] for name in names if name in inputs}


def get_y_opt(mda: BaseMDA) -> ndarray:
    """Return the optimal `y` array.

    Args:
        mda: The mda.

    Returns:
        The optimal `y` array.
    """
    return array([
        mda.io.output_data[y_1][0].real,
        mda.io.output_data[y_2][0].real,
    ])
