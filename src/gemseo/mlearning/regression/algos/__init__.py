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
r"""Regressors.

This package includes regression algorithms, a.k.a. regressors.

A regressor aims to find relationships between input and output variables.
After being fitted to a learning set,
the regression algorithms can predict output values of new input data.

A regression algorithm consists of identifying a function
:math:`f: \\mathbb{R}^{n_{\\textrm{inputs}}} \\to
\\mathbb{R}^{n_{\\textrm{outputs}}}`.
Given an input point
:math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`,
the predict method of the regression algorithm will return
the output point :math:`y = f(x) \\in \\mathbb{R}^{n_{\\textrm{outputs}}}`.
See :mod:`~gemseo.mlearning.core.supervised` for more information.

Wherever possible,
the regression algorithms should also be able
to compute the Jacobian matrix of the function it has learned to represent.
Thus,
given an input point :math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`,
the Jacobian prediction method of the regression algorithm should return the matrix

.. math::

    J_f(x) = \\frac{\\partial f}{\\partial x} =
    \\begin{pmatrix}
    \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}
        {\\partial x_{n_{\\textrm{inputs}}}}\\\\
    \\vdots & \\ddots & \\vdots\\\\
    \\frac{\\partial f_{n_{\\textrm{outputs}}}}{\\partial x_1} & \\cdots &
        \\frac{\\partial f_{n_{\\textrm{outputs}}}}
        {\\partial x_{n_{\\textrm{inputs}}}}
    \\end{pmatrix}
    \\in \\mathbb{R}^{n_{\\textrm{outputs}}\\times n_{\\textrm{inputs}}}.

Use the :class:`.RegressorFactory` to access all the available regressors
or derive either the :class:`.BaseRegressor` class to add a new one.
"""

from __future__ import annotations
