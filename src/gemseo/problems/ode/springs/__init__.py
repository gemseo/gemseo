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

r"""A problem illustrating how to couple ODE disciplines.

Consider a system of $n$ point masses with masses $m_1$, $m_2$,...
$m_n$ connected in series by springs. The displacement of the point masses
relative to the position at rest are denoted by $x_1$, $x_2$,...
$x_n$. Each spring has stiffness $k_1$, $k_2$,... $k_{n+1}$.

Motion is assumed to only take place in one dimension, along the axes of the springs.

The extremities of the first and last spring are fixed. This means that by convention,
$x_0 = x_{n+1} = 0$.


For $n=2$, the system is as follows:

```

    |                                                                                 |
    |        k1           ________          k2           ________          k3         |
    |  /\      /\        |        |   /\      /\        |        |   /\      /\       |
    |_/  \    /  \     __|   m1   |__/  \    /  \     __|   m2   |__/  \    /  \     _|
    |     \  /    \  /   |        |      \  /    \  /   |        |      \  /    \  /  |
    |      \/      \/    |________|       \/      \/    |________|       \/      \/   |
    |                         |                              |                        |
                           ---|--->                       ---|--->
                              |   x1                         |   x2
```

The force of a spring with stiffness $k$ is

$$\vec{F} = -kx$$

where $x$ is the displacement of the extremity of the spring.

Newton's second law applied to any point mass $m_i$ can be written as

$$
    m_i \ddot{x_i} = \sum \vec{F} = k_i (x_{i-1} - x_i) + k_{i+1} (x_{i+1} - x_i)
                 = k_i x_{i-1} + k_{i+1} x_{i+1} - (k_i + k_{i+1}) x_i
$$

This can be re-written as a system of first-order ordinary differential equations:

$$
    \left\{ \begin{cases}
        \dot{x_i} &= y_i \\
        \dot{y_i} &=
            - \frac{k_i + k_{i+1}}{m_i}x_i
            + \frac{k_i}{m_i}x_{i-1} + \frac{k_{i+1}}{m_i}x_{i+1}
    \end{cases} \right.
$$
"""

from __future__ import annotations
