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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A quadratic programming (QP) problem."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

if TYPE_CHECKING:
    from numpy.typing import NDArray


class QuadraticProgrammingProblem(NamedTuple):
    r"""A quadratic programming (QP) problem.

    Minimize :math:`0.5x^TQx + c^Tx + d` with respect to :math:`x` under the linear
    constraints :math:`Ax-b <= 0`.
    """

    Q: NDArray[float]
    r"""The matrix :math:`Q` of the objective function :math:`0.5x^TQx + c^Tx + d`.

    This matrix must be symmetric.
    """

    c: NDArray[float]
    r"""The matrix :math:`c` of the objective function :math:`0.5x^TQx + c^Tx + d`."""

    d: NDArray[float]
    r"""The matrix :math:`d` of the objective function :math:`0.5x^TQx + c^Tx + d`."""

    A: NDArray[float]
    r"""The matrix :math:`A` of the constraint function :math:`Ax-b`."""

    b: NDArray[float]
    r"""The matrix :math:`b` of the constraint function :math:`Ax-b`."""
