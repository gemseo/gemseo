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
"""The Ishigami function."""
from __future__ import annotations

from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.uncertainty.use_cases.ishigami.functions import compute_gradient
from gemseo.uncertainty.use_cases.ishigami.functions import compute_output


class IshigamiFunction(MDOFunction):
    r"""The Ishigami function.

    .. math::
       f(x_1,_2,x_3) = \sin(x_1)+ 7\sin(x_2)^2 + 0.1x_3^4\sin(X_1)

    See :cite:`ishigami1990`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__(compute_output, "Ishigami", jac=compute_gradient)
