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
r"""A problem connecting the Ishigami function with its uncertain space.

The Isighami function
:math:`f(x_1,_2,x_3) = \sin(x_1)+ 7\sin(x_2)^2 + 0.1x_3^4\sin(X_1)`
is commonly studied through the random variable :math:`Y=f(X_1,X_2,X_3)`
where :math:`X_1`, :math:`X_2` and :math:`X_3` are independent random variables
uniformly distributed over :math:`[-\pi,\pi]`.

See :cite:`ishigami1990`.

See Also:
    :class:`.IshigamiFunction`
    :class:`.IshigamiSpace`
"""
from __future__ import annotations

from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.uncertainty.use_cases.ishigami.ishigami_function import IshigamiFunction
from gemseo.uncertainty.use_cases.ishigami.ishigami_space import IshigamiSpace


class IshigamiProblem(OptimizationProblem):
    """A problem connecting the Ishigami function with its uncertain space."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__(IshigamiSpace())
        self.objective = IshigamiFunction()
