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

from typing import Final

from numpy import pi

__A: Final[float] = 7
__B: Final[float] = 0.1

mean: Final[float] = __A / 2
r"""The expectation of the output.

$$\mathbb{E}[Y] = \frac{a}{2}$$
"""

variance: Final[float] = 0.5 + __A**2 / 8 + __B**2 * pi**8 / 18 + __B * pi**4 / 5
r"""The variance of the output.

$$\mathbb{V}[Y] = \frac{1}{2} + \frac{a^2}{8} + \frac{b^2\pi^8}{18} + \frac{b\pi^4}{5}$$
"""

sobol_1: Final[float] = 0.5 * (1 + __B * pi**4 / 5) ** 2 / variance
r"""The first-order Sobol' index of $X_1$.

$$S_1 = \frac{(1+b\frac{pi^4}{5})^2}{2\mathbb{V}[Y]}$$
"""

sobol_2: Final[float] = __A**2 / 8 / variance
r"""The first-order Sobol' index of $X_2$.

$$S_2 = \frac{a^2}{8\mathbb{V}[Y]}$$
"""

sobol_3: Final[float] = 0.0
r"""The first-order Sobol' index of $X_3$.

$$S_3 = 0$$
"""

sobol_12: Final[float] = 0.0
r"""The second-order Sobol' index of $X_1$ and $X_2$.

$$S_{1,2} = 0$$
"""

sobol_23: Final[float] = 0.0
r"""The second-order Sobol' index of $X_2$ and $X_3$.

$$S_{2,3} = 0$$
"""

sobol_13: Final[float] = __B**2 * pi**8 * 8 / 225 / variance
r"""The second-order Sobol' index of $X_1$ and $X_3$.

$$S_{1,3} = \frac{8b^2\pi^8}{225\mathbb{V}[Y]}$$
"""

sobol_123: Final[float] = 0.0
r"""The second-order Sobol' index of $X_1$, $X_2$ and $X_3$.

$$S_{1,2,3} = 0$$
"""

total_sobol_1: Final[float] = sobol_1 + sobol_13
r"""The total Sobol' index of $X_1$.

$$S_1^T = S_1 + S_{1,3}$$
"""

total_sobol_2: Final[float] = sobol_2
r"""The total Sobol' index of $X_2$.

$$S_2^T = S_2$$
"""

total_sobol_3: Final[float] = sobol_13
r"""The total Sobol' index of $X_3$.

$$S_3^T = S_{1,3}$$
"""
