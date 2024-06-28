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
"""Tests for NormFunction."""

from __future__ import annotations

import numpy
import pytest

from gemseo.algos.preprocessed_functions.norm_function import NormFunction


def test_special_repr(problem_with_identity) -> None:
    """Check NormFunction.special_repr."""
    assert (
        NormFunction(
            problem_with_identity.objective,
            problem_with_identity.design_space,
        ).special_repr
        == "Identity"
    )


def test_function_without_jacobian(problem_with_identity) -> None:
    """Check the handling of a function without Jacobian."""
    with pytest.raises(
        ValueError,
        match="Selected user gradient but function f has no Jacobian function.",
    ):
        NormFunction(
            problem_with_identity.objective,
            problem_with_identity.design_space,
        ).jac(numpy.zeros(3))
