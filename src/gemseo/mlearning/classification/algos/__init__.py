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
"""Classifiers.

This package includes classification algorithms, a.k.a. classifiers.

Given an input data,
a classifier is used to predict
either the class associated with this input data
or the probability of belonging to each class.

Use the :class:`.ClassifierFactory` to access all the available classifiers
or derive the :class:`.BaseClassifier` class to add a new one.
"""

from __future__ import annotations
