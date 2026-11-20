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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""The tilted uniform distributions.

The tilted uniform distribution deserves a word,
as it is the only maximum-entropy family that is not implemented by the libraries.
For a support $[a,b]$ and a mean $\mu$,
maximizing the entropy under these constraints
gives the probability density function $f(x) \propto e^{\lambda x}$ over $[a,b]$,
where the tilt $\lambda$ is the Lagrange multiplier of the constraint on the mean.
This is the theorem 5.1 of the expository notes
[Probability distributions and maximum entropy](https://kconrad.math.uconn.edu/blurbs/analysis/entropypost.pdf)
by Keith Conrad,
which also proves the existence and the uniqueness of this tilt
and gives the expression of the mean used below, as well as its series;
the variance below is the derivative of that mean with respect to the tilt.
A negative tilt gives a decreasing density,
i.e. an exponential distribution truncated to the support,
a positive tilt gives an increasing one,
i.e. a reflected truncated exponential distribution,
and a zero tilt gives the uniform distribution.

This family is an exponential family with the natural parameter $\lambda$,
which is why its moments are expressed below
as functions of the dimensionless tilt $t=\lambda(b-a)$,
namely those of the reduced random variable $U=(X-a)/(b-a)$ supported by $[0,1]$:

$$\mathbb{E}[U] = \frac{1}{1-e^{-t}} - \frac{1}{t}
\quad\text{and}\quad
\mathbb{V}[U] = \frac{1}{t^2} - \frac{e^{-t}}{\left(1-e^{-t}\right)^2}$$

Two properties of this family are used repeatedly:
$\mathbb{V}[U]=\frac{\mathrm{d}\mathbb{E}[U]}{\mathrm{d}t}>0$,
so the mean strictly increases with the tilt,
and $\mathbb{V}[U]$ is an even function of $t$,
decreasing from $1/12$ at $t=0$, i.e. the uniform case, to $0$ as $|t|\to\infty$.
"""

from __future__ import annotations

from functools import lru_cache
from math import exp
from math import expm1
from math import isclose
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy_numerical_settings import (  # noqa: E501
        MaximumEntropyNumericalSettings,
    )


def compute_tilted_uniform_mean(
    tilt: float,
    minimum: float,
    maximum: float,
    *,
    small_tilt: float,
    large_tilt: float,
) -> float:
    r"""Compute the mean of a tilted uniform distribution.

    The probability density function of this distribution is proportional to
    $e^{\lambda x}$ over $[a,b]$
    where $\lambda$ is the tilt, $a$ the minimum and $b$ the maximum.

    The mean is $a+(b-a)\mathbb{E}[U]$
    where $\mathbb{E}[U]=\frac{1}{1-e^{-t}}-\frac{1}{t}$
    is the mean of the reduced random variable
    and $t=\lambda(b-a)$ the dimensionless tilt.
    It increases with the tilt,
    from $a$ when $t\to-\infty$ to $b$ when $t\to+\infty$,
    and is the middle of the support when the tilt is zero.

    Args:
        tilt: The tilt of the distribution.
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable,
            which must be greater than the minimum.
        small_tilt: The absolute dimensionless tilt,
            i.e. the tilt times the length of the support,
            below which the series expansions are used.
        large_tilt: The absolute dimensionless tilt
            above which the asymptotic expansions are used.

    Returns:
        The mean of the tilted uniform distribution.
    """
    support_length = maximum - minimum
    t = tilt * support_length
    if abs(t) < small_tilt:
        # The closed-form expression cancels out for small tilts,
        # as both of its terms diverge as 1/t; use its Taylor series instead.
        ratio = 0.5 + t / 12.0 - t**3 / 720.0 + t**5 / 30240.0
    elif t > large_tilt:
        # Beyond this tilt, e^{-t} is below the double-precision resolution
        # and the closed-form expression would overflow for very large tilts;
        # use its asymptote instead.
        ratio = 1.0 - 1.0 / t
    elif t < -large_tilt:
        # Same asymptotic reasoning for a large negative tilt,
        # for which 1/(1-e^{-t}) is negligible.
        ratio = -1.0 / t
    else:
        # 1/(1-e^{-t}) is computed as -1/expm1(-t) for the accuracy of expm1.
        ratio = -1.0 / expm1(-t) - 1.0 / t

    return minimum + support_length * ratio


def compute_tilted_uniform_standard_deviation(
    tilt: float,
    minimum: float,
    maximum: float,
    *,
    small_tilt: float,
    large_tilt: float,
) -> float:
    r"""Compute the standard deviation of a tilted uniform distribution.

    The standard deviation is $(b-a)\sqrt{\mathbb{V}[U]}$
    where $\mathbb{V}[U]=\frac{1}{t^2}-\frac{e^{-t}}{\left(1-e^{-t}\right)^2}$
    is the variance of the reduced random variable
    and $t=\lambda(b-a)$ the dimensionless tilt.
    It decreases with the absolute tilt,
    from $(b-a)/\sqrt{12}$ for the uniform distribution to $0$,
    as the tilt concentrates the distribution on one end of the support.

    Args:
        tilt: The tilt of the distribution.
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable,
            which must be greater than the minimum.
        small_tilt: The absolute dimensionless tilt,
            i.e. the tilt times the length of the support,
            below which the series expansions are used.
        large_tilt: The absolute dimensionless tilt
            above which the asymptotic expansions are used.

    Returns:
        The standard deviation of the tilted uniform distribution.
    """
    support_length = maximum - minimum
    # The variance is an even function of the tilt,
    # as tilting a support in one direction or the other
    # gives two distributions that are mirror images of each other;
    # the absolute tilt spares the branch for a large negative tilt.
    t = abs(tilt) * support_length
    if t < small_tilt:
        # The closed-form expression cancels out for small tilts,
        # as both of its terms diverge as 1/t^2; use its Taylor series instead.
        squared_ratio = 1.0 / 12.0 - t**2 / 240.0 + t**4 / 6048.0
    elif t > large_tilt:
        # Beyond this tilt, e^{-t} is below the double-precision resolution
        # and the closed-form expression would overflow for very large tilts;
        # use its asymptote instead.
        squared_ratio = 1.0 / t**2
    else:
        # (1-e^{-t})^2 is computed as expm1(-t)^2 for the accuracy of expm1.
        squared_ratio = 1.0 / t**2 - exp(-t) / expm1(-t) ** 2

    return support_length * squared_ratio**0.5


def compute_tilted_uniform_tilt(
    minimum: float,
    maximum: float,
    mu: float,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> float:
    """Compute the tilt of the tilted uniform distribution with a given mean.

    Args:
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable,
            which must be greater than the minimum.
        mu: The mean of the random variable,
            which must lie strictly between the minimum and the maximum.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The tilt of the distribution,
        positive when the mean is greater than the middle of the support,
        negative when it is smaller and zero when the distribution is uniform.

    Raises:
        ValueError: When the tilt cannot be bracketed
            within the maximum number of iterations.
    """
    # The settings are unpacked so that the computation can be cached,
    # as a Pydantic model is mutable, hence not hashable.
    return _compute_tilted_uniform_tilt(
        minimum,
        maximum,
        mu,
        settings.tilt_tolerance,
        settings.solver_tolerance,
        settings.small_tilt,
        settings.large_tilt,
        settings.maximum_solver_iterations,
    )


@lru_cache
def _compute_tilted_uniform_tilt(
    minimum: float,
    maximum: float,
    mu: float,
    tilt_tolerance: float,
    solver_tolerance: float,
    small_tilt: float,
    large_tilt: float,
    maximum_solver_iterations: int,
) -> float:
    r"""Compute the tilt of the tilted uniform distribution with given mean and bounds.

    The tilt is a deterministic function of its arguments and is requested
    several times when a distribution is created,
    e.g. by the validation of the settings and the computation of the parameters,
    so the results are cached using the @lru_cache decorator.

    The mean strictly increases with the tilt,
    as its derivative with respect to the tilt is the variance,
    so the tilt is the unique root of the mean error
    and is computed numerically,
    the mean $a+(b-a)\left(\frac{1}{1-e^{-t}}-\frac{1}{t}\right)$
    having no closed-form inverse.
    Conrad's notes, cited in the module docstring,
    suggest Newton's method but observe that it can diverge from a far start;
    the expanding bracket below guarantees
    the convergence of Brent's method instead.

    Args:
        minimum: The minimum of the random variable.
        maximum: The maximum of the random variable,
            which must be greater than the minimum.
        mu: The mean of the random variable,
            which must lie strictly between the minimum and the maximum.
        tilt_tolerance: The distance from the mean to the middle of the support,
            relative to the length of the support,
            below which a tilted uniform distribution is treated as a uniform one;
            it corresponds to a dimensionless tilt
            of about twelve times this value.
        solver_tolerance: The absolute tolerance of the solvers
            computing the distribution parameters;
            it applies to the dimensionless tilt,
            i.e. the tilt times the length of the support.
        small_tilt: The absolute dimensionless tilt,
            i.e. the tilt times the length of the support,
            below which the series expansions are used.
        large_tilt: The absolute dimensionless tilt
            above which the asymptotic expansions are used.
        maximum_solver_iterations: The maximum number of iterations to bracket a tilt.

    Returns:
        The tilt of the distribution,
        positive when the mean is greater than the middle of the support,
        negative when it is smaller and zero when the distribution is uniform.

    Raises:
        ValueError: When the tilt cannot be bracketed
            within the maximum number of iterations.
    """
    # The mean of a zero tilt is the middle of the support;
    # this case is handled apart, as the solver cannot reach a zero tilt exactly.
    middle = (minimum + maximum) / 2.0
    if isclose(mu, middle, rel_tol=0.0, abs_tol=tilt_tolerance * (maximum - minimum)):
        return 0.0

    def compute_mean_error(tilt: float) -> float:
        """Compute the difference between the mean of a tilt and the expected mean.

        Args:
            tilt: The tilt of the distribution.

        Returns:
            The difference between the resulting mean and the expected one.
        """
        return (
            compute_tilted_uniform_mean(
                tilt,
                minimum,
                maximum,
                small_tilt=small_tilt,
                large_tilt=large_tilt,
            )
            - mu
        )

    # The mean increases with the tilt,
    # so the mean error changes sign once and Brent's method applies;
    # the bracket is expanded instead of being fixed,
    # as the mean of a large tilt cannot be computed for any support.
    # A mean close to a bound requires a large tilt,
    # e.g. a mean at 1% of the support length from its minimum needs a tilt
    # of about -100 divided by this length.
    # The bracket and the tolerance are set on the dimensionless tilt,
    # so the accuracy of the mean and the iteration counts
    # do not degrade with the length of the support.
    support_length = maximum - minimum
    lower_tilt = -1.0 / support_length
    upper_tilt = 1.0 / support_length
    for _ in range(maximum_solver_iterations):
        if compute_mean_error(lower_tilt) <= 0.0 <= compute_mean_error(upper_tilt):
            return brentq(
                compute_mean_error,
                lower_tilt,
                upper_tilt,
                xtol=solver_tolerance / support_length,
            )

        lower_tilt *= 2.0
        upper_tilt *= 2.0

    msg = (
        "The tilted uniform distribution with the minimum "
        f"{minimum}, the maximum {maximum} and the mean {mu} "
        f"cannot be computed within the {maximum_solver_iterations} bracket "
        "expansions set by 'maximum_solver_iterations'; "
        "increase it or move the mean away from the bounds."
    )
    raise ValueError(msg)
