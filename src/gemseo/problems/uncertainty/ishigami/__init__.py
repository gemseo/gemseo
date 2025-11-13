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
r"""The Ishigami use case to benchmark and illustrate UQ algorithms.

The Isighami function
$f(x_1,_2,x_3) = \sin(x_1)+ 7\sin(x_2)^2 + 0.1x_3^4\sin(X_1)$
is commonly studied through the random variable $Y=f(X_1,X_2,X_3)$
where $X_1$, $X_2$ and $X_3$ are independent random variables
uniformly distributed over $[-\pi,\pi]$.

!!! quote "References"

    T. Ishigami and T. Homma.
    An importance quantification technique
    in uncertainty analysis for computer models.
    In First International Symposium on Uncertainty Modeling and Analysis, 1990.
"""

from __future__ import annotations
