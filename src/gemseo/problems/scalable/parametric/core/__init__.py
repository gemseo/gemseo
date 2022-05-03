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
r"""
Scalable module from Tedford and Martins (2010)
***********************************************

The modules located in this directory offer a set of
classes relative to the scalable problem introduced in the paper:

    Tedford NP, Martins JRRA (2010), Benchmarking
    multidisciplinary design optimization algorithms,
    Optimization and Engineering, 11(1):159-183.

Overview
~~~~~~~~

This scalable problem aims to minimize an objective function
quadratically depending on shared design parameters and coupling variables,
under inequality constraints linearly depending on these coupling variables.

System discipline
-----------------

A system discipline computes the constraints and the objective
in function of the shared design parameters and coupling variables.

The discipline takes the global design parameters :math:`z` and
the coupling variables :math:`y_1,y_2,\ldots,y_N` as inputs
and returns the objective function value :math:`f(x,y(x,y))` to minimize as
well as the inequality constraints ones
:math:`c_1(y_1),c_2(y_2),\ldots,c_N(y_N)` which are expressed as:

.. math::

   f(z,y) = |z|_2^2 + \sum_{i=1}^N |y_i|_2^2


and:

.. math::

   c_i(y_i) = 1- C_i^{-T}Iy_i

Strongly coupled disciplines
----------------------------

The coupling variables are the outputs of strongly coupled disciplines.

Each strongly coupled discipline computes a set of coupling variables
linearly depending on local design parameters, shared design parameters,
coupling variables from other strongly coupled disciplines,
and belonging to the unit hypercube.

The i-th discipline takes local design parameters :math:`x_i`
and shared design parameters :math:`z` in input as well as coupling
variables :math:`\left(y_i\right)_{1\leq j \leq N\atop j\neq i}`
from :math:`N-1` elementary disciplines,
and returns the coupling variables:

.. math::

    y_i =\frac{\tilde{y}_i+C_{z,i}.1+C_{x_i}.1}{\sum_{j=1 \atop j
    \neq i}^NC_{y_j,i}.1+C_{z,i}.1+C_{x_i}.1} \in [0,1]^{n_{y_i}}

where:

.. math::

    \tilde{y}_i = - C_{z,i}.z - C_{x_i}.x_i +
    \sum_{j=1 \atop j \neq i}^N C_{y_j,i}.y_j

Scalability
-----------

This problem is said "scalable"
because several sizing features can be chosen by the user:

- the number of local design parameters for each discipline,
- the number of shared design parameters,
- the number of coupling variables for each discipline,
- the number of disciplines.

A given sizing configuration is called "scaling strategy"
and this scalable module is particularly useful to compare different MDO
formulations with respect to the scaling strategy.
"""
from __future__ import annotations
