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
"""Statistics associated with the Ishigami use case."""
from __future__ import annotations

from numpy import pi

from gemseo.utils.python_compatibility import Final

__A: Final[float] = 7
__B: Final[float] = 0.1

MEAN: Final[float] = __A / 2
r"""The expectation of the output.

.. math::
   \mathbb{E}[Y] = \frac{a}{2}
"""

VARIANCE: Final[float] = (
    0.5 + __A**2 / 8 + __B**2 * pi**8 / 18 + __B * pi**4 / 5
)
r"""The variance of the output.

.. math::
   \mathbb{V}[Y] = \frac{1}{2} + \frac{a^2}{8} + \frac{b^2\pi^8}{18} + \frac{b\pi^4}{5}
"""

SOBOL_1: Final[float] = 0.5 * (1 + __B * pi**4 / 5) ** 2 / VARIANCE
r"""The first-order Sobol' index of :math:`X_1`.

.. math::
   S_1 = \frac{(1+b\frac{pi^4}{5})^2}{2\mathbb{V}[Y]}
"""

SOBOL_2: Final[float] = __A**2 / 8 / VARIANCE
r"""The first-order Sobol' index of :math:`X_2`.

.. math::
   S_2 = \frac{a^2}{8\mathbb{V}[Y]}
"""

SOBOL_3: Final[float] = 0.0
r"""The first-order Sobol' index of :math:`X_3`.

.. math::
   S_3 = 0
"""

SOBOL_12: Final[float] = 0.0
r"""The second-order Sobol' index of :math:`X_1` and :math:`X_2`.

.. math::
   S_{1,2} = 0
"""

SOBOL_23: Final[float] = 0.0
r"""The second-order Sobol' index of :math:`X_2` and :math:`X_3`.

.. math::
   S_{2,3} = 0
"""

SOBOL_13: Final[float] = __B**2 * pi**8 * 8 / 225 / VARIANCE
r"""The second-order Sobol' index of :math:`X_1` and :math:`X_3`.

.. math::
   S_{1,3} = \frac{8b^2\pi^8}{225\mathbb{V}[Y]}
"""

SOBOL_123: Final[float] = 0.0
r"""The second-order Sobol' index of :math:`X_1`, :math:`X_2` and :math:`X_3`.

.. math::
   S_{1,2,3} = 0
"""

TOTAL_SOBOL_1: Final[float] = SOBOL_1 + SOBOL_13
r"""The total Sobol' index of :math:`X_1`.

.. math::
   S_1^T = S_1 + S_{1,3}
"""

TOTAL_SOBOL_2: Final[float] = SOBOL_2
r"""The total Sobol' index of :math:`X_2`.

.. math::
   S_2^T = S_2
"""

TOTAL_SOBOL_3: Final[float] = SOBOL_13
r"""The total Sobol' index of :math:`X_3`.

.. math::
   S_3^T = S_{1,3}
"""
