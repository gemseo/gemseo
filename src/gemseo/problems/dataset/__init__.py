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
"""Examples of dataset.

|g| proposes several datasets containing academic data
to illustrate its capabilities:

- :class:`.IrisDataset` is a collection of iris plants,
  mainly used to benchmark clustering and classification algorithms,
- :class:`.RosenbrockDataset` is a set of evaluations of
  the `Rosenbrock function <https://en.wikipedia.org/wiki/Rosenbrock_function>`__
  over a regular grid,
  initially introduced to illustrate visualization tools dedicated to surfaces
  such as :class:`.ZvsXY`,
- :class:`.BurgersDataset` is a set of solutions of
  the `Burgers' equation <https://en.wikipedia.org/wiki/Burgers%27_equation>`__
  at given times,
  initially introduced to illustrate dimension reduction methods,
  e.g. :class:`.PCA` or :class:`.KLSVD`.
"""
from __future__ import annotations
