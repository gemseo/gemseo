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

r"""Linear model fitting algorithms.

A linear model assumes that the relationship
between the output scalar variables :math:`y_1,\ldots,y_p`
and the input scalar variables :math:`x_1,\ldots,x_d`
is linear, i.e.

.. math::

   y_j = w_0 + w_1 x_1 + \ldots + w_d x_d.

where :math:`w_1,\ldots,w_d` are the weights.

The input variables
(resp. output variables)
are also called
regressors, explanatory variables, predictors, or independent variables
(resp. responses, targets or dependent variables).

Given :math:`n` observations of these variables,
we obtain:

.. math::

   y_j^{(i)} = w_{0,j} + w_{1,j} x_1^{(i)} + \ldots + w_{d,j} x_d^{(i)},\qquad i=1,\ldots,n, \quad j=1,\ldots,p


This system of :math:`n` equations can be written in matrix notation as

.. math::

   Y = Xw

where

.. math::

   X = \left(\begin{matrix}x_1^{(1)} & \ldots & x_d^{(1)} \\ \vdots & \ddots & \vdots \\ x_1^{(n)} & \ldots & x_d^{(n)}\end{matrix}\right)\in\mathcal{M}_{n,d}(\mathbb{R})

   Y = \left(\begin{matrix}y_1^{(1)} & \ldots & y_p^{(1)} \\ \vdots & \ddots & \vdots \\ y_1^{(n)} & \ldots & y_p^{(n)}\end{matrix}\right)\in\mathcal{M}_{n,p}(\mathbb{R})

   w = \left(\begin{matrix}w_{1,1} & \ldots & w_{1,p} \\ \vdots & \ddots & \vdots \\ w_{d,1} & \ldots & w_{d,p}\end{matrix}\right)\in\mathcal{M}_{d,p}(\mathbb{R})

This package proposes different algorithms to fit such a linear model,
i.e. finding the weights :math:`w` minimizing :math:`\|Y-Xw\|`.
These linear model fitting algorithms derive
from the base class :class:`.BaseLinearModelFitter`
which is equipped with :class:`.BaseLinearModelFitter_Settings`
and can be created from a :class:`.LinearModelFitterFactory`.

The available algorithms are
:class:`.LinearRegression`, :class:`.Lasso`, :class:`.Ridge`, :class:`.LARS`,
:class:`.ElasticNet` and :class:`.OrthogonalMatchingPursuit`.
"""  # noqa: E501
