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
"""The conservative support of a transformed OpenTURNS distribution.

The image of the support before transformation under the transformation
is never computed,
as any symbolic computation of an image can run forever,
e.g. that of "sin(x)+cos(x/3)" on [0, inf).
sympy is only used to parse the transformation
and to compute its limits and its singular points.

When the support before transformation is bounded on both sides,
the range that OpenTURNS estimates is exact for a continuous transformation,
so it is kept,
unless sympy finds that the transformation diverges
or is not defined at a point of the support,
in which case the end concerned is infinite.

When the support before transformation is unbounded on at least one side,
an end is infinite when sympy finds that the transformation diverges that way
and is otherwise the end of the range that OpenTURNS estimates,
including when the transformation has no limit at an infinite end, e.g. "sin(x)",
or when sympy cannot handle the transformation, e.g. "besselJ0(x)".
Both ends are flagged as unknown,
as this range is a numerical estimate
that clips the unbounded side
and can miss an extremum of the transformation,
so a truncation bound is never compared with it.
The support is an interval,
so it can include more than the image,
e.g. when the transformation is not continuous,
as the image of the real line under "1/x".

This module is imported lazily,
so that sympy is only imported when a transformation is actually used.
"""

from __future__ import annotations

from functools import lru_cache
from math import isfinite
from tokenize import TokenError
from typing import Any
from typing import Final
from typing import NamedTuple

from numpy import inf
from sympy import Abs
from sympy import E
from sympy import Expr
from sympy import FiniteSet
from sympy import Interval
from sympy import Rational
from sympy import S
from sympy import Symbol
from sympy import acot
from sympy import ceiling
from sympy import cot
from sympy import exp
from sympy import floor
from sympy import limit
from sympy import log
from sympy import loggamma
from sympy import oo
from sympy import pi
from sympy import sign
from sympy import sqrt
from sympy.calculus.singularities import singularities as compute_singularities
from sympy.calculus.util import continuous_domain
from sympy.core.function import AppliedUndef
from sympy.core.sympify import SympifyError
from sympy.parsing.sympy_parser import convert_xor
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import standard_transformations
from sympy.polys.polyerrors import PolynomialError

from gemseo.uncertainty.distribution.core.base import BaseDistribution

_variable: Final[Symbol] = Symbol(BaseDistribution.default_variable_name)
"""The symbol of the variable that a transformation applies to."""

_parsing_transformations: Final[tuple[Any, ...]] = (
    *standard_transformations,
    convert_xor,
)
"""The sympy parsing rules, mapping the power operator `^` of OpenTURNS to `**`."""

_aliases: Final[dict[str, Any]] = {
    "acotan": acot,
    "cbrt": lambda argument: sign(argument) * Abs(argument) ** Rational(1, 3),
    "ceil": ceiling,
    "cotan": cot,
    "e_": E,
    "expm1": lambda argument: exp(argument) - 1,
    "hypot": lambda first, second: sqrt(first**2 + second**2),
    "lngamma": loggamma,
    "log10": lambda argument: log(argument, 10),
    "log1p": lambda argument: log(1 + argument),
    "log2": lambda argument: log(argument, 2),
    "pi_": pi,
    "rint": lambda argument: floor(argument + S.Half),
    "round": lambda argument: floor(argument + S.Half),
    "sgn": sign,
    "trunc": lambda argument: sign(argument) * floor(Abs(argument)),
}
"""The sympy counterparts of the OpenTURNS symbols that sympy does not know.

`cbrt` is the real cube root of OpenTURNS,
whereas the `cbrt` of sympy is the principal complex one.

`round` and `rint` round the ties towards the infinity,
whereas OpenTURNS rounds them away from zero and to the even integer respectively;
this only changes the value at a tie
and so does not change the support of a random variable.
"""

_errors: Final[tuple[type[Exception], ...]] = (
    AttributeError,
    NotImplementedError,
    PolynomialError,
    RecursionError,
    SympifyError,
    SyntaxError,
    TokenError,
    TypeError,
    ValueError,
)
"""The errors raised by sympy when it cannot analyze a transformation."""


class _TransformedSupport(NamedTuple):
    """The support of a transformed random variable."""

    lower_bound: float | None
    """The lower bound, or `None` to keep the lower end of the OpenTURNS range."""

    upper_bound: float | None
    """The upper bound, or `None` to keep the upper end of the OpenTURNS range."""

    lower_bound_is_known: bool
    """Whether a truncation bound can be compared with the lower bound."""

    upper_bound_is_known: bool
    """Whether a truncation bound can be compared with the upper bound."""


def _parse_transformation(transformation: str) -> Expr | None:
    """Parse a transformation written in the language of OpenTURNS.

    Args:
        transformation: The transformation, e.g. `"exp(x)"`.

    Returns:
        The expression of the transformation,
        or `None` when sympy cannot represent it faithfully.
    """
    try:
        expression = parse_expr(
            transformation,
            local_dict=dict(_aliases),
            transformations=_parsing_transformations,
        )
    except _errors:
        return None

    if (
        not isinstance(expression, Expr)
        or expression.atoms(AppliedUndef)
        or expression.free_symbols - {_variable}
    ):
        # sympy parses a function that it does not know, e.g. besselJ0,
        # as a function that it leaves unevaluated instead of raising an error,
        # which would silently yield a wrong support.
        # A comparison, e.g. "(x>0)", is parsed as a relation
        # that no calculus utility of sympy can handle.
        return None

    return expression


def _create_domain(lower_bound: float, upper_bound: float) -> Interval:
    """Create the sympy interval standing for the support before transformation.

    Args:
        lower_bound: The lower bound of the support before transformation.
        upper_bound: The upper bound of the support before transformation.

    Returns:
        The sympy interval.
    """
    return Interval(
        -oo if lower_bound == -inf else lower_bound,
        oo if upper_bound == inf else upper_bound,
    )


def _compute_bound(value: Any) -> float | None:
    """Compute a bound of the support from a value computed by sympy.

    Args:
        value: The value computed by sympy.

    Returns:
        The bound of the support,
        or `None` when the value is not a real number,
        e.g. an expression that sympy left unevaluated,
        not-a-number or the complex infinity.
    """
    if value.is_number and value.is_extended_real:
        return float(value)

    return None


def _compute_limit(expression: Expr, point: Any, direction: str = "+") -> float | None:
    """Compute the limit of a transformation at a point of the support.

    Args:
        expression: The expression of the transformation.
        point: The point of the support.
        direction: The side of the point that the limit is taken from.

    Returns:
        The limit,
        or `None` when sympy cannot tell where the transformation goes,
        e.g. when it oscillates.
    """
    try:
        return _compute_bound(limit(expression, _variable, point, direction))
    except _errors:
        return None


def _find_singularities(expression: Expr, domain: Interval) -> tuple[Any, ...] | None:
    """Find the points of the support where a transformation is not continuous.

    Args:
        expression: The expression of the transformation.
        domain: The support before transformation.

    Returns:
        The points of the support where the transformation is not continuous,
        or `None` when sympy cannot enumerate them.
    """
    try:
        points = domain - continuous_domain(expression, _variable, domain)
    except _errors:
        # sympy cannot determine the domain of continuity,
        # e.g. of "erf(x)/x^3",
        # so its poles are searched for instead.
        try:
            points = compute_singularities(expression, _variable, domain)
        except _errors:
            # sympy cannot find the poles either,
            # e.g. those of "erf(x)" or "floor(x)",
            # so the transformation is assumed to be continuous.
            return ()

    if points is S.EmptySet:
        return ()

    if not isinstance(points, FiniteSet):
        return None

    points = tuple(points)
    if any(_compute_bound(point) is None for point in points):
        return None

    return points


def _find_divergence(expression: Expr, domain: Interval) -> tuple[bool, bool] | None:
    r"""Find where a transformation makes the range of OpenTURNS untrustworthy.

    The range that OpenTURNS estimates is exact
    where the transformation is bounded and evaluated without error,
    which is not the case near a point of the support
    where it diverges or is not defined.

    The transformation is examined at both ends of the support,
    whether they are finite or infinite,
    as it can diverge at a finite end too, e.g. `"gamma(x)"` at 0.

    Args:
        expression: The expression of the transformation.
        domain: The support before transformation.

    Returns:
        Whether the lower end of the range must be reported as $-\infty$
        and whether the upper end must be reported as $+\infty$;
        a transformation without limit at an infinite end, e.g. `"sin(x)"`,
        does not diverge there for this purpose,
        as the support is then unbounded on that side
        and its ends are flagged as unknown by the caller.
        `None` is returned
        when sympy cannot tell where the transformation goes
        at a finite end or at a singular point, e.g. `"sin(1/x)"` at 0.
    """
    diverges_down = diverges_up = False
    for point, direction in ((domain.inf, "+"), (domain.sup, "-")):
        value = _compute_limit(expression, point, direction)
        if value is None:
            if point in {-oo, oo}:
                # The transformation has no limit at the infinity, e.g. "sin(x)";
                # the end of the range of OpenTURNS is kept, flagged as unknown.
                continue

            return None

        if value == inf:
            diverges_up = True
        elif value == -inf:
            diverges_down = True

    singularities = _find_singularities(expression, domain)
    if singularities is None:
        return True, True

    for point in singularities:
        directions = []
        if point > domain.inf:
            directions.append("-")

        if point < domain.sup:
            directions.append("+")

        values = []
        for direction in directions:
            value = _compute_limit(expression, point, direction)
            if value is None:
                return None

            values.append(value)

        if inf in values:
            diverges_up = True

        if -inf in values:
            diverges_down = True

        if all(isfinite(value) for value in values) and (
            _compute_bound(expression.subs(_variable, point)) is None
        ):
            # The transformation is bounded around this point
            # but is not defined at it, e.g. "sin(x)/x" at 0;
            # OpenTURNS evaluates it as not-a-number there
            # and the extremum that its range misses because of it
            # can be anywhere.
            return True, True

    return diverges_down, diverges_up


@lru_cache(maxsize=128)
def compute_transformed_support(
    transformation: str, lower_bound: float, upper_bound: float
) -> _TransformedSupport:
    """Compute the support of a transformed random variable.

    Args:
        transformation: The transformation, e.g. `"exp(x)"`.
        lower_bound: The lower bound of the support before transformation.
        upper_bound: The upper bound of the support before transformation.

    Returns:
        The support of the transformed random variable.
    """
    expression = _parse_transformation(transformation)
    if expression is None:
        if isfinite(lower_bound) and isfinite(upper_bound):
            # The image of a support bounded on both sides
            # under a continuous transformation is compact,
            # so both ends of the range are exact up to a rounding error,
            # whatever the points of the support they come from.
            # The continuity cannot be verified for a transformation
            # that sympy cannot represent,
            # so the range is trusted for such a transformation,
            # e.g. that of "besselY0(x)" over [0, 1] is reported as the support
            # even though this transformation diverges at 0.
            return _TransformedSupport(None, None, True, True)

        # Nothing is known about the ends coming from an unbounded side,
        # so the range of OpenTURNS is kept and flagged as unknown.
        return _TransformedSupport(None, None, False, False)

    domain = _create_domain(lower_bound, upper_bound)
    divergence = _find_divergence(expression, domain)
    if divergence is None:
        # The transformation is beyond the reach of sympy near a singular point,
        # so no end of the range of OpenTURNS can be trusted.
        return _TransformedSupport(-inf, inf, False, False)

    diverges_down, diverges_up = divergence
    if isfinite(lower_bound) and isfinite(upper_bound):
        return _TransformedSupport(
            -inf if diverges_down else None,
            inf if diverges_up else None,
            not diverges_down,
            not diverges_up,
        )

    # The range of OpenTURNS is a numerical estimate on an unbounded side:
    # it clips this side and can miss an interior extremum.
    # No image is computed, so the ends that are not infinite are those of this range,
    # and no truncation bound is compared with them.
    return _TransformedSupport(
        -inf if diverges_down else None,
        inf if diverges_up else None,
        False,
        False,
    )
