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

import itertools

import numpy
import pytest

from gemseo.core.mdofunctions.norm_function import NormFunction

parametrize_normalize_round_ints = pytest.mark.parametrize(
    ("normalize", "round_ints"), tuple(itertools.product([False, True], [False, True]))
)


@parametrize_normalize_round_ints
def test_special_repr(problem_with_identity, normalize, round_ints) -> None:
    """Check NormFunction.special_repr."""
    assert (
        NormFunction(
            problem_with_identity.objective,
            normalize,
            round_ints,
            problem_with_identity,
        ).special_repr
        == "Identity"
    )


@parametrize_normalize_round_ints
def test_function_without_jacobian(
    problem_with_identity, normalize, round_ints
) -> None:
    """Check the handling of a function without Jacobian."""
    with pytest.raises(
        ValueError,
        match="Selected user gradient but function Identity has no Jacobian matrix !",
    ):
        NormFunction(
            problem_with_identity.objective,
            normalize,
            round_ints,
            problem_with_identity,
        ).jac(numpy.zeros(3))
