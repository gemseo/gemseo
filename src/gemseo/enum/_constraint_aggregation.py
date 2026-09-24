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
"""The functions to aggregate constraints."""

from __future__ import annotations

from enum import StrEnum


# Defined here rather than nested in ConstraintAggregation so that the modules using
# it, e.g. gemseo.optimization.core.constraints, do not import that discipline and
# its dependencies just for the enumeration.
class EvaluationFunction(StrEnum):
    """A function to compute an aggregation of constraints."""

    IKS = "IKS"
    """The induces exponential function."""

    LOWER_BOUND_KS = "lower_bound_KS"
    """The lower bound Kreisselmeier-Steinhauser function."""

    UPPER_BOUND_KS = "upper_bound_KS"
    """The upper bound Kreisselmeier-Steinhauser function."""

    POS_SUM = "POS_SUM"
    """The positive sum squared function."""

    MAX = "MAX"
    """The maximum function."""

    SUM = "SUM"
    """The sum squared function."""
