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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Tests for the conservative support of a transformed distribution."""

from __future__ import annotations

import pytest
from numpy import inf
from sympy import FiniteSet
from sympy import I
from sympy import Interval
from sympy import Symbol

from gemseo.uncertainty.distribution.openturns import _transformed_support
from gemseo.uncertainty.distribution.openturns._transformed_support import (
    _find_singularities,
)
from gemseo.uncertainty.distribution.openturns._transformed_support import (
    _TransformedSupport,
)
from gemseo.uncertainty.distribution.openturns._transformed_support import (
    compute_transformed_support,
)


@pytest.mark.parametrize(
    ("transformation", "lower_bound", "upper_bound", "expected_support"),
    [
        # The image is never computed;
        # on an unbounded side, both ends are unknown,
        # the ends that are not infinite being those of OpenTURNS.
        ("x^2", -inf, inf, _TransformedSupport(None, inf, False, False)),
        ("exp(x)", -inf, inf, _TransformedSupport(None, inf, False, False)),
        ("exp(x)", 0.0, inf, _TransformedSupport(None, inf, False, False)),
        ("exp(-x)", 0.0, inf, _TransformedSupport(None, None, False, False)),
        # The range of OpenTURNS is exact on a support bounded on both sides
        # for a continuous transformation.
        ("x^2", -1.0, 2.0, _TransformedSupport(None, None, True, True)),
        # sympy cannot parse the transformation;
        # the image of a support bounded on both sides is compact,
        # so the range of OpenTURNS is kept and known,
        # while on an unbounded side it is kept but unknown,
        # so that it never rejects a truncation bound.
        ("(", 0.0, 1.0, _TransformedSupport(None, None, True, True)),
        ("(", 0.0, inf, _TransformedSupport(None, None, False, False)),
        # sympy parses a comparison as a relation instead of an expression.
        ("(x>0)", 0.0, 1.0, _TransformedSupport(None, None, True, True)),
        ("(x>0)", -inf, inf, _TransformedSupport(None, None, False, False)),
        # sympy leaves a function that it does not know unevaluated.
        ("besselJ0(x)", 0.0, 1.0, _TransformedSupport(None, None, True, True)),
        ("besselJ0(x)", -inf, inf, _TransformedSupport(None, None, False, False)),
        # sympy parses a symbol that is not the variable as another variable,
        # so that the expression is not a function of the variable alone.
        ("x*y", 0.0, 1.0, _TransformedSupport(None, None, True, True)),
        ("x*y", 0.0, inf, _TransformedSupport(None, None, False, False)),
        # The limits tell that the transformation diverges on one side only,
        # so the other end of the range of OpenTURNS is kept but unknown.
        ("abs(x)", -inf, inf, _TransformedSupport(None, inf, False, False)),
        # The transformation has no limit at the infinity;
        # the range of OpenTURNS is kept, flagged as unknown.
        ("x*sin(x)", -inf, inf, _TransformedSupport(None, None, False, False)),
        # sympy cannot tell where the transformation goes at the singular point 0.
        ("sin(1/x)", 0.0, 1.0, _TransformedSupport(-inf, inf, False, False)),
        # The transformation converges at both ends of the support
        # but sympy cannot tell where it goes at the singular point 2.
        ("sin(1/(x-2))", 0.0, 3.0, _TransformedSupport(-inf, inf, False, False)),
        # sympy fails to compute the limit of the transformation at its pole,
        # a root of a quintic polynomial without a closed-form expression.
        ("1/(x^5-x-1)", 0.0, 2.0, _TransformedSupport(-inf, inf, False, False)),
        # The transformation is bounded around 0 but is not defined at it,
        # where OpenTURNS evaluates it as not-a-number,
        # so no end of its range can be trusted.
        ("sin(x)/x", -inf, inf, _TransformedSupport(-inf, inf, False, False)),
        # This transformation diverges at the upper end 1 of the support,
        # which is a singular point that has a single side in the support.
        # The upper end of the range of OpenTURNS is kept.
        ("1/(x-1)", 0.0, 1.0, _TransformedSupport(-inf, None, False, True)),
        # This transformation diverges at the lower end 0 of the support,
        # so only the upper end of its range cannot be trusted.
        ("sin(x)/x^3", 0.0, 1.0, _TransformedSupport(None, inf, True, False)),
        # sympy cannot enumerate the points where the transformation
        # is not continuous, as they form an interval,
        # so no end of the range of OpenTURNS can be trusted.
        (
            "sqrt(x*(x-1)*(x-2))",
            0.0,
            3.0,
            _TransformedSupport(-inf, inf, False, False),
        ),
        # sympy cannot determine the domain of continuity of these transformations;
        # the limits at the ends of the support tell that they diverge at 0,
        # which the poles that sympy can still find also tell for the second one.
        ("gamma(x)", 0.0, 1.0, _TransformedSupport(None, inf, True, False)),
        ("erf(x)/x^3", 0.0, 1.0, _TransformedSupport(None, inf, True, False)),
        # This transformation is bounded around 0 but is not defined at it,
        # which the poles that sympy finds tell.
        ("erf(x)/x", -inf, inf, _TransformedSupport(-inf, inf, False, False)),
        # sympy struggles with these transformations; the image is never computed.
        ("sin(1/x)", 1.0, 2.0, _TransformedSupport(None, None, True, True)),
        ("sin(x^2)", 0.0, 3.0, _TransformedSupport(None, None, True, True)),
        (
            "sin(x)+cos(x/3)",
            0.0,
            20.0,
            _TransformedSupport(None, None, True, True),
        ),
        ("x+sin(x)", 0.0, inf, _TransformedSupport(None, inf, False, False)),
        # The transformation has no limit at the infinity,
        # so the range of OpenTURNS is kept, flagged as unknown.
        (
            "sin(x)+cos(x/3)",
            0.0,
            inf,
            _TransformedSupport(None, None, False, False),
        ),
        (
            "sin(x)+cos(x/3)",
            -inf,
            inf,
            _TransformedSupport(None, None, False, False),
        ),
    ],
    ids=[
        "square-unbounded",
        "exp-real-line",
        "exp-half-line",
        "decreasing-exp-half-line",
        "bounded-support",
        "unparsable-bounded",
        "unparsable-unbounded",
        "relation-bounded",
        "relation-unbounded",
        "unknown-function-bounded",
        "unknown-function-unbounded",
        "other-variable-bounded",
        "other-variable-unbounded",
        "divergence-on-one-side-only",
        "no-limit-at-infinity",
        "unknown-direction-at-singular-point",
        "unknown-direction-at-inner-singular-point",
        "unsolvable-pole",
        "undefined-at-bounded-point",
        "pole-at-upper-end",
        "pole-at-lower-end",
        "interval-of-discontinuities",
        "unknown-domain-diverging-gamma",
        "unknown-domain-diverging-erf",
        "unknown-domain-undefined-at-point",
        "sin-of-inverse",
        "sin-of-square",
        "sum-of-sines",
        "linear-plus-sin-unbounded",
        "sum-of-sines-half-line",
        "sum-of-sines-real-line",
    ],
)
def test_compute_transformed_support(
    transformation, lower_bound, upper_bound, expected_support
) -> None:
    """Check the support computed for the different capabilities of sympy.

    Some cases are transformations that sympy struggles with,
    e.g. its `solveset` does not terminate for the sum of sines:
    the image is never computed.
    There is no timeout, so a regression hangs the test session.
    The cache is bypassed,
    so that sympy is called even when another test computed the same support.
    """
    assert (
        compute_transformed_support.__wrapped__(
            transformation, lower_bound, upper_bound
        )
        == expected_support
    )


def test_transformation_without_known_poles(monkeypatch) -> None:
    """Check a transformation whose domain of continuity and poles are unknown.

    Args:
        monkeypatch: The fixture to patch the search for the poles.
    """

    def raise_error(*args):
        """Raise the error that sympy raises when it cannot find the poles.

        Args:
            *args: The arguments of the search for the poles.

        Raises:
            NotImplementedError: When called.
        """
        raise NotImplementedError

    monkeypatch.setattr(_transformed_support, "compute_singularities", raise_error)
    # The transformation is assumed to be continuous,
    # so the range that OpenTURNS estimates is kept,
    # instead of being rejected because the transformation is not defined at 0.
    assert compute_transformed_support.__wrapped__(
        "erf(x)/x", -inf, inf
    ) == _TransformedSupport(None, None, False, False)


def test_find_singularities_not_real(monkeypatch) -> None:
    """Check a transformation whose poles are not all real numbers.

    Args:
        monkeypatch: The fixture to patch the search for the poles.
    """

    def raise_error(*args):
        """Raise the error that sympy raises when it cannot find the continuity domain.

        Args:
            *args: The arguments of the search for the continuity domain.

        Raises:
            NotImplementedError: When called.
        """
        raise NotImplementedError

    monkeypatch.setattr(_transformed_support, "continuous_domain", raise_error)
    monkeypatch.setattr(
        _transformed_support, "compute_singularities", lambda *args: FiniteSet(I)
    )
    assert _find_singularities(Symbol("x"), Interval(0, 1)) is None


def test_cache() -> None:
    """Check that a support is only computed once."""
    compute_transformed_support("2*x+1", -1.0, 1.0)
    hits = compute_transformed_support.cache_info().hits
    compute_transformed_support("2*x+1", -1.0, 1.0)
    assert compute_transformed_support.cache_info().hits == hits + 1
