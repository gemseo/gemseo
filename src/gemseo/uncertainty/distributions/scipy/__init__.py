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
"""Scipy-based capabilities for probability distributions.

This package interfaces capabilities
from the `SciPy <https://www.scipy.org/>`_ library.

Interfaced distributions
------------------------

This package implements the abstract classes :class:`.Distribution`
and :class:`.ComposedDistribution`.

Classical distributions
-----------------------

This module also implements a deliberately limited selection
of classical probability distributions
in a user-friendly way: :class:`.SPExponentialDistribution`,
:class:`.SPNormalDistribution`, :class:`.SPTriangularDistribution`,
and :class:`.SPUniformDistribution`. More precisely,
the argument whose nature is a dictionary of keyword parameters
is replaced with several user-defined keyword arguments.
In this way, the use writes :code:`SPUniformDistribution('x', -1., 3.)`
or :code:`SPUniformDistribution('x', minimum=-1., maximum=3.)`
instead of  :code:`SPDistribution('x', 'Uniform', {"loc": -1, "scale": 4})`.
Furthermore, these classes inheriting from :class:`.SPDistribution`
are documented in such a way that a newbie could easily apprehend them.

Composed distribution
---------------------

A :code:`SPDistribution` has a :attr:`.SPDistribution._COMPOSED_DISTRIBUTION`
attribute referencing :class:`.SPComposedDistribution`
which is a class to build a composed distribution
related to given random variables from a list of :class:`.SPDistribution` objects
implementing the probability distributions of these variables
based on the SciPy library and from a copula name.

.. note::

   A copula is a mathematical function used to define the dependence
   between random variables from their cumulative density functions.
   `See more <https://en.wikipedia.org/wiki/Copula_(probability_theory)>`__.
"""
from __future__ import annotations
