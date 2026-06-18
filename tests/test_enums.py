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
"""Tests for the gemseo.enums package."""

from __future__ import annotations

from enum import Enum

import pytest

from gemseo import enums


@pytest.mark.parametrize("name", enums.__all__)
def test_all_exports_are_enums(name) -> None:
    """Every name advertised in ``__all__`` resolves to an enumeration."""
    obj = getattr(enums, name)
    assert isinstance(obj, type)
    assert issubclass(obj, Enum)


def test_merged_enums_are_the_owner_class_enums() -> None:
    """The merged enumerations are re-exported from their owner class."""
    from gemseo.algos.evaluation_problem import EvaluationProblem
    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.functions.array_function import ArrayFunction

    assert enums.LinearizationMode is Discipline.LinearizationMode
    assert enums.FunctionType is ArrayFunction.FunctionType
    assert enums.DifferentiationMethod is EvaluationProblem.DifferentiationMethod


def test_sub_enum_values_match_merged_enum() -> None:
    """The sub-enum members share the values of the merged enumeration."""
    assert enums.ApproximationMode.COMPLEX_STEP == enums.LinearizationMode.COMPLEX_STEP
    assert (
        enums.HybridApproximationMode.HYBRID_COMPLEX_STEP
        == enums.LinearizationMode.HYBRID_COMPLEX_STEP
    )
    assert enums.ConstraintType.EQ == enums.FunctionType.EQ
