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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Sebastien Bocquet, Alexandre Scotto Di Perrotolo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The sub-package for acceleration methods."""

from __future__ import annotations

from enum import auto

from strenum import PascalCaseStrEnum


# TODO: Link the enum entrees to the available SequequenceTransformer classes.
class AccelerationMethod(PascalCaseStrEnum):
    """The acceleration method to be used to improve convergence rate.

    More details on each acceleration methods can be found in the dedicated module
    :mod:`gemseo.algos.sequence_transformer.acceleration_methods`.
    """

    AITKEN = auto()
    """The Aitken method."""

    ALTERNATE_2_DELTA = "Alternate2Delta"
    """The alternate 2-ẟ method."""

    ALTERNATE_DELTA_SQUARED = auto()
    """The alternate ẟ² method."""

    MINIMUM_POLYNOMIAL = auto()
    """The minimum polynomial method."""

    NONE = "NoTransformation"
    """No acceleration method."""

    SECANT = auto()
    """The secant method."""
