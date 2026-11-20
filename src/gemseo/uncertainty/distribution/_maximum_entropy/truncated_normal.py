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
"""The truncated normal distributions.

These distributions maximize the entropy for a mean, a standard deviation
and a minimum, a maximum or both,
as long as this standard deviation does not exceed that of the distribution
maximizing the entropy for the support and the mean only,
which is computed here as well.

Their parameters are those of the normal distribution before truncation
and have no closed-form expression;
they are fitted by a solver whose moments are computed by adaptive quadrature.

The moments of a truncated normal distribution do have closed-form expressions
in terms of the standard normal cumulative distribution function,
directly or through the truncnorm distribution of SciPy,
so the quadrature could be avoided;
both such fits were measured and rejected,
as they lose the regime where the standard deviation approaches
the largest achievable one,
in which the distribution to be truncated degenerates,
its standard deviation diverges
and the mass retained on the support becomes the difference
of two nearly equal values of that cumulative distribution function.
There, a fit parametrized by the parameters of the distribution to be truncated
does not converge at all on part of that regime
and reaches a relative error of about 1e-2 over the rest of it.
A fit in the natural parameters,
whose moments still come from these closed-form expressions,
brings that error down to about 1e-5,
whereas the quadrature stays below 1e-13
over the whole set of admissible moments.
A closed form that would remain accurate there
requires a joint expansion in the tilt and in the curvature
around the tilted uniform limit,
hence more machinery than the quadrature rather than less.
"""

from __future__ import annotations

import itertools
from math import exp
from math import inf
from math import log
from typing import TYPE_CHECKING

from scipy.integrate import quad
from scipy.optimize import root

from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_standard_deviation,
)
from gemseo.uncertainty.distribution._maximum_entropy.tilted_uniform import (
    compute_tilted_uniform_tilt,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from gemseo.uncertainty.distribution.core.base_univariate_settings.maximum_entropy_numerical_settings import (  # noqa: E501
        MaximumEntropyNumericalSettings,
    )


class _QuadratureError(Exception):
    """The error raised when a quadrature converges on noise."""


def _compute_reduced_moments(
    tilt: float, curvature: float, bounded: bool
) -> tuple[float, float, float, float] | None:
    r"""Compute the moments of a reduced exponential-quadratic density.

    The probability density function is proportional to $e^{tu+qu^2}$
    over $[0,1]$ if bounded, or $[0,+\infty)$ otherwise,
    where $t$ is the tilt and $q$ the curvature.
    A negative curvature gives a truncated normal density,
    a zero curvature a tilted uniform one
    and a positive curvature a log-convex one.

    Every guard below blocks a numerical failure mode
    that silently corrupted the moments in double precision
    and is pinned by a test in `test_maximum_entropy_util.py`.

    Args:
        tilt: The coefficient of the linear term of the log-density.
        curvature: The coefficient of the quadratic term of the log-density;
            the caller guarantees it is negative when the support is unbounded,
            as the density would otherwise not be integrable.
        bounded: Whether the support is $[0,1]$; otherwise it is $[0,+\infty)$.

    Returns:
        The mean and the second, third and fourth central moments,
        or `None` when the quadrature cannot resolve the density.
    """
    upper = 1.0 if bounded else inf

    def compute_exponent(u: float) -> float:
        """Compute the exponent of the density.

        Args:
            u: The value of the reduced random variable.

        Returns:
            The exponent of the density.
        """
        return (tilt + curvature * u) * u

    # The exponent is shifted by its maximum over the support,
    # reached at an end or at the vertex of the parabola,
    # so that the integrand peaks at one and can never overflow;
    # the reduction is oriented by the caller so this maximum stays small
    # (test_truncated_normal_parameters_reflection).
    candidates = [0.0]
    if bounded:
        candidates.append(1.0)
    if curvature < 0.0:
        vertex = -tilt / (2.0 * curvature)
        if 0.0 < vertex < upper:
            candidates.append(vertex)
    shift = max(compute_exponent(u) for u in candidates)
    if abs(shift) > 1e4:
        # The shifted exponent is a difference of two numbers of this magnitude,
        # so it inherits a roundoff of about this magnitude times the machine
        # epsilon, i.e. about 2e-12 at the guard, which is all the accuracy the
        # quadrature tolerance below can use;
        # the density cannot be resolved beyond it,
        # which the caller avoids by dropping the bounds
        # beyond the weak-truncation margin
        # (test_truncated_normal_parameters_far_bound).
        return None

    def compute_weight(u: float) -> float:
        """Compute the shifted density.

        Args:
            u: The value of the reduced random variable.

        Returns:
            The density, up to the constant factor of the shift.
        """
        return exp(compute_exponent(u) - shift)

    # A density much narrower than the support underflows to zero
    # at every node of the initial quadrature grid,
    # which then stops with a zero integral instead of subdividing;
    # graded breakpoints keep the spikes resolved,
    # around every end and vertex of the exponent,
    # as a positive curvature makes both ends local maxima,
    # at the scale of the slope for a spike at an end of the support
    # and at the scale of the curvature for a spike at the vertex
    # (test_truncated_normal_parameters_scale_invariance).
    curvature_width = (2.0 * abs(curvature)) ** -0.5 if curvature != 0.0 else inf
    breakpoints = set(candidates)
    for candidate in candidates:
        slope = abs(tilt + 2.0 * curvature * candidate)
        width = min(curvature_width, 1.0 / slope if slope > 0.0 else inf)
        if width < inf:
            breakpoints.update(
                candidate + sign * grading * width
                for grading in (1.0, 3.0, 10.0, 100.0)
                for sign in (1.0, -1.0)
            )
    breakpoints = sorted(u for u in breakpoints if 0.0 < u < upper)

    # The quadrature must be more accurate than the tolerance of the solver
    # it feeds, whose Jacobian steps would otherwise fall below the noise;
    # full_output silences the roundoff warning that such a tight relative
    # tolerance can trigger, as the accuracy is policed by the solver residuals.
    options = {"limit": 400, "epsabs": 0.0, "epsrel": 1e-13, "full_output": 1}
    if bounded:
        # The breakpoints are passed to the quadrature,
        # which is not possible with an infinite bound.
        options["points"] = breakpoints or None
        edges = [0.0, upper]
    else:
        # The integral is split at the breakpoints instead.
        edges = [0.0, *breakpoints, upper]

    def integrate(integrand: Callable[[float], float]) -> tuple[float, float]:
        """Integrate a function over the support, split at the breakpoints.

        Args:
            integrand: The function to integrate.

        Returns:
            The integral of the function over the support
            and the estimate of its absolute error.
        """
        value = 0.0
        error = 0.0
        for lower_edge, upper_edge in itertools.pairwise(edges):
            results = quad(integrand, lower_edge, upper_edge, **options)
            value += results[0]
            error += results[1]

        return value, error

    normalization, error = integrate(compute_weight)
    if not 0.0 < normalization < inf:
        return None

    # A quadrature that converged on noise, e.g. an integrand corrupted by
    # roundoff, reports a large error estimate; such moments are rejected.
    # The third central moment is exempt, as it vanishes for a centered density,
    # where a purely relative tolerance cannot be met
    # (test_weak_truncation_boundary crosses this regime),
    # so the error of every integral but that one is checked below.
    # These estimates are compared to a fraction of the normalization,
    # so they detect noise rather than bound the relative error
    # of a central moment much smaller than that normalization.
    error_bound = 1e-8 * normalization

    def check_error(error: float) -> None:
        """Check that a quadrature did not converge on noise.

        Args:
            error: The estimate of the absolute error of the quadrature.

        Raises:
            _QuadratureError: When this estimate exceeds the error bound.
        """
        if error > error_bound:
            raise _QuadratureError

    try:
        check_error(error)
        mean, error = integrate(lambda u: u * compute_weight(u))
        check_error(error)
        mean /= normalization
        moments = []
        for order in (2, 3, 4):
            moment, error = integrate(
                lambda u, k=order: (u - mean) ** k * compute_weight(u)
            )
            if order != 3:
                check_error(error)

            moments.append(moment / normalization)
    except _QuadratureError:
        return None

    moment_2, moment_3, moment_4 = moments
    if moment_2 <= 0.0:
        # The second central moment underflowed;
        # the density is too narrow to be resolved for these parameters
        # (test_truncated_normal_parameters_underflowing_scale).
        return None

    return mean, moment_2, moment_3, moment_4


def compute_truncated_normal_parameters(
    minimum: float | None,
    maximum: float | None,
    mu: float,
    sigma: float,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> tuple[float, float]:
    r"""Compute the parameters of a truncated normal distribution.

    The distribution is fitted in the natural parameters
    of the exponential family $f(x) \propto e^{\lambda_1 x + \lambda_2 x^2}$,
    expressed on a reduced support,
    because the fit remains regular up to the boundary $\lambda_2 = 0$,
    i.e. up to the largest standard deviation,
    where the distribution degenerates into a tilted uniform one;
    a parametrization by the standard deviation of the distribution to be truncated
    would degenerate there instead, as this standard deviation diverges.
    The moments are computed by adaptive quadrature
    and the Jacobian is analytic on the feasible set,
    the parameters outside it being handled by a barrier:
    the derivatives of the moments with respect to the natural parameters
    are covariances of the sufficient statistics of the family,
    a classical property of the exponential families,
    namely that the gradient and the Hessian
    of the logarithm of the Laplace transform
    are the mean and the covariance of these statistics,
    e.g. Brown, *Fundamentals of Statistical Exponential Families*,
    IMS Lecture Notes-Monograph Series 9, 1986, chapter 2.

    The solver uses the solver tolerance of the settings
    as a relative tolerance on the steps of its natural parameters,
    so the accuracy of the fit is bounded
    by the acceptance threshold of the moment residuals,
    namely the square root of this tolerance,
    rather than by the tolerance itself.

    Args:
        minimum: The minimum of the random variable.
            If `None`, the random variable is not bounded from below.
        maximum: The maximum of the random variable.
            If `None`, the random variable is not bounded from above.
        mu: The mean of the random variable.
        sigma: The standard deviation of the random variable.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The mean and standard deviation of the normal distribution
        to be truncated in order to get the expected mean and standard deviation.

    Raises:
        ValueError: When the parameters cannot be computed.
    """
    solver_tolerance = settings.solver_tolerance
    # A bound is negligible when the mass that it cuts off moves the moments
    # by less than the solver tolerance,
    # and that mass is governed by the tail of the density that is retained,
    # hence by whether the other bound truncates, so two margins are needed.
    # When no bound truncates, the density is the normal one and a bound
    # z standard deviations away from the mean moves the truncated moments
    # by about sigma times the standard normal density at z for the mean
    # and half of z times as much for the standard deviation;
    # the Gaussian margin below solves exp(-z^2/2) = tolerance,
    # plus one for safety (about 8.4 standard deviations at the default tolerance).
    # When the other bound truncates strongly, however,
    # the fitted normal distribution is centered outside the support
    # and the density retained on it decays like exp(-z) instead,
    # so the Gaussian margin would underestimate the neglected mass
    # by orders of magnitude; the exponential margin solves exp(-z) = tolerance
    # and adds a logarithmic correction for the weight of about z^2
    # with which that mass enters the second central moment
    # (about 34 standard deviations at the default tolerance).
    tolerance = min(solver_tolerance, 0.1)
    gaussian_margin = (-2.0 * log(tolerance)) ** 0.5 + 1.0
    exponential_margin = -log(tolerance) + 2.0 * log(-log(tolerance))
    if (minimum is None or (mu - minimum) / sigma >= gaussian_margin) and (
        maximum is None or (maximum - mu) / sigma >= gaussian_margin
    ):
        # Neither bound truncates, so the distribution is the normal one itself;
        # this also spares the quadrature a density
        # that would be much narrower than the support.
        return mu, sigma

    # One bound truncates the density,
    # so the other is dropped only beyond the exponential margin;
    # both bounds cannot be dropped here, as the shortcut above would have applied.
    if minimum is not None and (mu - minimum) / sigma >= exponential_margin:
        minimum = None
    if maximum is not None and (maximum - mu) / sigma >= exponential_margin:
        maximum = None

    # The support is mapped onto [0,1],
    # or onto [0,+inf) with the distance from the mean to the bound as the unit,
    # so that the reduction is scale-free;
    # the support is reflected when the mean is nearer the maximum,
    # so that the density peaks near the origin,
    # where the exponent tu+qu^2 of the reduced density is small:
    # the shift subtracted from this exponent below is about half of the square
    # of the distance from the peak to the origin in reduced standard deviations,
    # so anchoring the reduction at the nearer bound keeps that shift,
    # and hence the roundoff that it costs, as small as possible.
    if minimum is not None and maximum is not None:
        bounded = True
        length = maximum - minimum
        if mu - minimum <= maximum - mu:
            origin = minimum
            orientation = 1.0
        else:
            origin = maximum
            orientation = -1.0
    elif minimum is not None:
        bounded = False
        origin = minimum
        orientation = 1.0
        length = mu - minimum
    else:
        bounded = False
        origin = maximum
        orientation = -1.0
        length = maximum - mu

    reduced_mu = orientation * (mu - origin) / length
    reduced_sigma = sigma / length

    def compute_errors_and_jacobian(
        parameters: tuple[float, float],
    ) -> tuple[list[float], list[list[float]]]:
        """Compute the moment errors and their Jacobian.

        Args:
            parameters: The tilt and the curvature of the reduced density.

        Returns:
            The differences between the resulting reduced moments
            and the expected ones,
            and their Jacobian with respect to the tilt and the curvature.
        """
        tilt, curvature = parameters
        if not bounded and curvature >= 0.0:
            # The density is not integrable on a semi-infinite support;
            # a linear barrier sends the solver back to negative curvatures.
            barrier = 1e3 * (1.0 + curvature)
            return [barrier, barrier], [[1e3, 1e3], [1e3, 1e3]]

        moments = _compute_reduced_moments(tilt, curvature, bounded)
        if moments is None:
            # The quadrature could not resolve the density for these parameters;
            # a large residual sends the solver elsewhere
            # and the acceptance check below rejects a solver stopping here.
            return [1e3, 1e3], [[1e3, 1e3], [1e3, 1e3]]

        mean, moment_2, moment_3, moment_4 = moments
        std = moment_2**0.5
        jacobian = [
            [moment_2, moment_3 + 2.0 * mean * moment_2],
            [
                moment_3 / (2.0 * std),
                (moment_4 + 2.0 * mean * moment_3 - moment_2**2) / (2.0 * std),
            ],
        ]
        return [mean - reduced_mu, std - reduced_sigma], jacobian

    # The curvature of the distribution to be truncated
    # ranges from that of the expected standard deviation,
    # when the truncation is negligible,
    # to nearly zero, when the truncation is strong.
    for factor in (1.0, 2.0, 10.0, 100.0, 1000.0):
        # The margins above keep the support within about 43 standard deviations,
        # so the reduced standard deviation is at least about 1/43
        # and neither its square nor the initial curvature can overflow.
        curvature_0 = -0.5 / (reduced_sigma * factor) ** 2
        solution = root(
            compute_errors_and_jacobian,
            [-2.0 * curvature_0 * reduced_mu, curvature_0],
            jac=True,
            tol=solver_tolerance,
        )
        # The status reported by the solver is unreliable in both directions:
        # its step criterion does not certify the residuals,
        # so it can report success on a spurious root after stalling,
        # and it can report a failure to progress after landing on the root
        # when the curvature is tiny;
        # the solution is thus accepted on its residuals alone.
        if (
            solution.x[1] < 0.0
            and max(abs(solution.fun[0]), abs(solution.fun[1])) <= solver_tolerance**0.5
        ):
            tilt, curvature = solution.x
            # A negative curvature makes the reduced density that of
            # a truncated normal distribution with these parameters,
            # mapped back to the original support by the affine reduction.
            reduced_normal_mu = -tilt / (2.0 * curvature)
            reduced_normal_sigma = (-1.0 / (2.0 * curvature)) ** 0.5
            return (
                origin + orientation * length * reduced_normal_mu,
                length * reduced_normal_sigma,
            )

    msg = (
        "The truncated normal distribution with the mean "
        f"{mu} and the standard deviation {sigma} cannot be computed: "
        "no fit reached moment residuals below "
        f"{solver_tolerance**0.5}, the square root of 'solver_tolerance' "
        f"({solver_tolerance}); "
        "a larger 'solver_tolerance', at most 1e-6, loosens this threshold."
    )
    raise ValueError(msg)


def compute_truncated_normal_maximum_standard_deviation(
    minimum: float | None,
    maximum: float | None,
    mu: float,
    *,
    settings: MaximumEntropyNumericalSettings,
) -> float:
    """Compute the largest standard deviation with a truncated normal distribution.

    Given a support and a mean,
    the distribution maximizing the entropy for an additional standard deviation
    is a truncated normal one
    as long as this standard deviation does not exceed that of the distribution
    maximizing the entropy for the support and the mean only.
    Beyond it,
    the probability density function maximizing the entropy is log-convex,
    i.e. U-shaped, or monotone and convex,
    which no truncated normal distribution can be, as their densities are log-concave,
    and does not even exist when the support is semi-infinite:
    on such a support, the maximum returned here,
    i.e. the distance from the mean to the bound,
    is the existence criterion of Dowson and Wragg,
    *Maximum-entropy distributions having prescribed first and second moments*,
    IEEE Transactions on Information Theory, 19(5):689-693, 1973,
    doi:10.1109/TIT.1973.1055060.
    On a bounded support, the maximizer exists and is unique
    for any achievable pair of moments,
    which is the Hausdorff case of Mead and Papanicolaou,
    *Maximum entropy in the problem of moments*,
    Journal of Mathematical Physics, 25(8):2404-2417, 1984,
    doi:10.1063/1.526446.

    Args:
        minimum: The minimum of the random variable.
            If `None`, the random variable is not bounded from below.
        maximum: The maximum of the random variable.
            If `None`, the random variable is not bounded from above.
        mu: The mean of the random variable.
        settings: The settings of the numerical methods
            computing the distribution parameters.

    Returns:
        The largest standard deviation of a maximum-entropy distribution.
    """
    # With a lower bound, this distribution is an exponential one,
    # whose standard deviation is the distance from the mean to the lower bound.
    if maximum is None:
        return mu - minimum

    # With an upper bound, this distribution is a reflected exponential one,
    # whose standard deviation is the distance from the mean to the upper bound.
    if minimum is None:
        return maximum - mu

    # With two bounds, this distribution is a tilted uniform one.
    return compute_tilted_uniform_standard_deviation(
        compute_tilted_uniform_tilt(minimum, maximum, mu, settings=settings),
        minimum,
        maximum,
        small_tilt=settings.small_tilt,
        large_tilt=settings.large_tilt,
    )
