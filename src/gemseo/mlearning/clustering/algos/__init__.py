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
"""Clusterers.

This package includes clustering algorithms, a.k.a. clusterers.

Given an input data,
a clusterer is used to group data into classes, a.k.a. clusters.

Wherever possible,
these algorithms should be able to predict the class of a new data,
as well as the probability of belonging to each class.

Use the :class:`.ClustererFactory` to access all the available clusterers
or derive either the :class:`.BaseClusterer` or :class:`.BasePredictiveClusterer`  class
to add a new one.
"""

from __future__ import annotations
