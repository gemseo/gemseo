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
from __future__ import annotations

from math import erf
from math import exp
from math import gamma
from math import log
from math import pi
from math import sin
from pathlib import Path
from sys import modules
from unittest.mock import patch

import pytest
from numpy import array
from numpy import inf
from numpy import int32
from numpy import int_
from numpy import ndarray
from numpy.testing import assert_allclose
from numpy.testing import assert_almost_equal
from numpy.testing import assert_array_less
from numpy.testing import assert_equal
from openturns import Bernoulli
from openturns import Beta
from openturns import Dirac
from openturns import Exponential
from openturns import FiniteDiscreteDistribution
from openturns import LogNormal
from openturns import Normal
from openturns import RandomGenerator
from openturns import Triangular
from openturns import Uniform
from openturns import WeibullMax
from openturns import WeibullMin
from scipy.stats import truncnorm

from gemseo.uncertainty.distribution._log_normal_util import compute_mu_l_and_sigma_l
from gemseo.uncertainty.distribution.core import base_univariate
from gemseo.uncertainty.distribution.factory import distribution_factory
from gemseo.uncertainty.distribution.openturns.bernoulli_settings import (
    OTBernoulliDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.beta_settings import (
    OTBetaDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.dirac_settings import (
    OTDiracDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.distribution import OTDistribution
from gemseo.uncertainty.distribution.openturns.distribution_settings import (
    OTDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.exponential_settings import (
    OTExponentialDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.finite_discrete_settings import (
    OTFiniteDiscreteDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distribution.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.log_normal import OTLogNormalDistribution
from gemseo.uncertainty.distribution.openturns.log_normal_settings import (
    OTLogNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.normal import OTNormalDistribution
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.triangular import (
    OTTriangularDistribution,
)
from gemseo.uncertainty.distribution.openturns.triangular_settings import (
    OTTriangularDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.weibull_settings import (
    OTWeibullDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception


@pytest.fixture(scope="module")
def distribution() -> OTDistribution:
    """Normal distribution with mean 0 and std 2."""
    return OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Normal", parameters=(0, 2))
    )


def test_joint_distribution() -> None:
    """Check the joint distribution associated with a OTDistribution."""
    assert OTJointDistribution == OTDistribution.joint_distribution_class


def test_constructor(distribution) -> None:
    assert distribution.transformation == "x"


def test_bad_distribution(snapshot) -> None:
    with assert_exception(ImportError, snapshot):
        OTDistribution(
            OTDistribution_Settings(interfaced_distribution="Dummy", parameters=(0, 1))
        )


def test_bad_distribution_parameters(snapshot) -> None:
    with assert_exception(ValueError, snapshot):
        OTDistribution(
            OTDistribution_Settings(
                interfaced_distribution="Normal", parameters=(0, 1, 2)
            )
        )


def test_str() -> None:
    distribution = OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Normal", parameters=(0, 2))
    )
    assert str(distribution) == "Normal(0, 2)"
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Normal",
            parameters=(0, 2),
            standard_parameters={"mean": 0, "var": 4},
        )
    )
    assert str(distribution) == "Normal(mean=0, var=4)"


@pytest.mark.parametrize("n_samples", [3, int_(3)])
def test_compute_samples(distribution, n_samples) -> None:
    RandomGenerator.SetSeed(0)
    sample = distribution.compute_samples(n_samples)
    assert isinstance(sample, ndarray)
    assert sample.ndim == 1
    assert_almost_equal(sample, array([1.216403, -2.532346, -0.876531]), decimal=3)


def test_get_cdf(distribution) -> None:
    assert distribution.compute_cdf(0.0) == 0.5


def test_get_inverse_cdf(distribution) -> None:
    assert distribution.compute_inverse_cdf(0.5) == 0.0


def test_cdf(distribution) -> None:
    assert distribution._cdf(0.0) == 0.5


def test_pdf(distribution) -> None:
    assert distribution._pdf(0.0) == pytest.approx(0.19947114020071632, abs=1e-3)


def test_mean(distribution) -> None:
    assert distribution.mean == 0.0


def test_std(distribution) -> None:
    assert distribution.standard_deviation == 2.0


def test_support(distribution) -> None:
    assert_equal(distribution.support, array([-inf, inf]))


def test_range(distribution) -> None:
    assert_almost_equal(distribution.range, array([-15.301256, 15.301256]), decimal=3)
    distribution = OTDistribution(
        OTDistribution_Settings(interfaced_distribution="Uniform", parameters=(0, 1))
    )
    assert_equal(distribution.range, array([0.0, 1.0]))


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"lower_bound": 0.5, "upper_bound": 0.6}, array([0.5, 0.6])),
        ({"lower_bound": 0.5}, array([0.5, 2.0])),
        ({"upper_bound": 0.5}, array([0.0, 0.5])),
    ],
)
def test_truncation(kwargs, expected) -> None:
    """Check the support after truncation."""
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Uniform", parameters=(0.0, 2.0), **kwargs
        )
    )
    assert_equal(distribution.support, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"upper_bound": 1.5},
        {"lower_bound": -0.5},
        {"lower_bound": -0.5, "upper_bound": 1.0},
        {"lower_bound": 0.0, "upper_bound": 1.5},
    ],
    ids=["upper-only", "lower-only", "lower-of-both", "upper-of-both"],
)
def test_truncate_exception(kwargs, snapshot) -> None:
    """Check that exceptions are raised when mistruncating a distribution."""
    with assert_exception(ValueError, snapshot):
        OTDistribution(
            OTDistribution_Settings(
                interfaced_distribution="Uniform", parameters=(0, 1), **kwargs
            )
        )


@pytest.mark.parametrize(
    ("transformation", "expected_transformation"),
    [
        ("2*x", "2*(x)"),
        ("x^2+x", "(x)^2+(x)"),
        ("exp(x)", "exp((x))"),
    ],
    ids=["product", "polynomial", "variable-in-function-name"],
)
def test_transformation_expression(transformation, expected_transformation) -> None:
    """Check the expression of the transformation of the random variable."""
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Uniform",
            parameters=(0.0, 1.0),
            transformation=transformation,
        )
    )
    assert distribution.transformation == expected_transformation


def test_reflection_support_without_sympy(monkeypatch) -> None:
    """Check that the support of the reflection does not require sympy."""
    monkeypatch.setitem(modules, "sympy", None)
    monkeypatch.delitem(
        modules,
        "gemseo.uncertainty.distribution.openturns._transformed_support",
        raising=False,
    )
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Exponential",
            parameters=(1.0, 0.0),
            transformation="- x",
        )
    )
    assert distribution.math_lower_bound == -inf
    assert distribution.math_upper_bound == 0.0


@pytest.mark.parametrize(
    ("parameters", "transformation", "expected_support"),
    [
        ((0.0, 2.0), "2*x", [0.0, 4.0]),
        ((0.0, pi), "sin(x)", [0.0, 1.0]),
        ((-1.0, 2.0), "x^2", [0.0, 4.0]),
        # OpenTURNS names the constant pi differently from sympy.
        ((0.0, 1.0), "sin(pi_*x)", [0.0, 1.0]),
    ],
    ids=[
        "linear",
        "monotone-sine",
        "square-with-interior-minimum",
        "constant-named-differently",
    ],
)
def test_support_of_transformation_of_bounded_support(
    parameters, transformation, expected_support
) -> None:
    """Check the support of a transformation of a distribution bounded on both sides.

    The support is the numerical range that OpenTURNS estimates,
    which is exact up to a rounding error
    even when the transformation is not monotone
    and an end is thus attained at an interior point of the support.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Uniform",
            parameters=parameters,
            transformation=transformation,
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-11)


@pytest.mark.parametrize(
    ("interfaced_distribution", "parameters", "transformation", "expected_support"),
    [
        ("Exponential", (1.0, 0.0), "exp(-x)", [0.0, 1.0]),
        ("Normal", (0.0, 1.0), "tanh(x)", [-1.0, 1.0]),
        # These transformations have no limit at the infinity,
        # so nothing is known about the ends of their support,
        # which are the ends of the range that OpenTURNS estimates,
        # flagged as unknown so that they never reject a truncation bound.
        ("Normal", (0.0, 1.0), "sin(x)", [-1.0, 1.0]),
        ("Normal", (0.0, 2.0), "sin(x)", [-1.0, 1.0]),
        ("Student", (3.0,), "sin(x)", [-1.0, 1.0]),
        ("LogNormal", (0.0, 1.0, 0.0), "cos(x)", [-1.0, 1.0]),
        ("Normal", (0.0, 1.0), "sin(x)+cos(x/3)", [-1.8096, 1.8787]),
        ("Exponential", (1.0, 0.0), "sin(x)+cos(x/3)", [-1.8787, 1.8787]),
        ("Normal", (0.0, 1.0), "x^2", [0.0, inf]),
        # OpenTURNS estimates the range of its support,
        # which is [-pi/2, pi/2], as the image of the quantiles of its density.
        ("Normal", (0.0, 1.0), "atan(x)", [-1.4408249, 1.4408249]),
        ("Normal", (0.0, 1.0), "x/(1+x^2)", [-0.5, 0.5]),
        # The image of the real line under this transformation
        # is a union of two intervals,
        # of which the support is the smallest interval including it.
        ("Normal", (0.0, 1.0), "1/x", [-inf, inf]),
        # sympy does not know this transformation,
        # so nothing is known about its support,
        # which is thus the range that OpenTURNS estimates, flagged as unknown.
        ("Normal", (0.0, 1.0), "besselJ0(x)", [-0.4028, 1.0]),
        # The transformation is unbounded both ways but has no limit,
        # so the reported ends are only the estimate of OpenTURNS.
        ("Normal", (0.0, 1.0), "x*sin(x)", [-4.8145, 7.4930]),
        # sympy knows this transformation but cannot compute its image;
        # its limits tell that it converges on both sides of the support.
        ("Normal", (0.0, 1.0), "erf(x)", [-1.0, 1.0]),
        # OpenTURNS names this function differently from sympy.
        ("Exponential", (1.0, 1.0), "log10(x)", [0.0, inf]),
        # sympy does not know this function of OpenTURNS at all.
        ("Normal", (0.0, 1.0), "sgn(x)", [-1.0, 1.0]),
    ],
    ids=[
        "decreasing-exponential",
        "tanh",
        "sine",
        "sine-wide-normal",
        "sine-heavy-tailed",
        "cosine-lognormal",
        "sum-of-sines-normal",
        "sum-of-sines-exponential",
        "square-diverging",
        "arctangent",
        "bounded-rational",
        "two-interval-image",
        "unknown-to-sympy",
        "oscillation-growing-without-limit",
        "image-unknown-limits-converge",
        "log10-named-differently",
        "sign-unknown-to-sympy",
    ],
)
def test_support_of_transformation_of_unbounded_support(
    interfaced_distribution, parameters, transformation, expected_support
) -> None:
    """Check the support of a transformation of a partly unbounded support.

    The image of the support is not computed.
    An end is infinite when the transformation diverges that way, e.g. `"x^2"`,
    and is otherwise the end of the range that OpenTURNS estimates,
    including when the transformation has no limit, e.g. `"sin(x)"`,
    or is unknown to sympy, e.g. `"besselJ0(x)"`.
    This range is only a numerical estimate of the end of the support,
    e.g. for `"atan(x)"`, hence the tolerance.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution=interfaced_distribution,
            parameters=parameters,
            transformation=transformation,
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    ("interfaced_distribution", "parameters", "transformation", "expected_support"),
    [
        ("Exponential", (1.0, 0.0), "exp(x)", [1.0, inf]),
        ("Exponential", (1.0, -1.0), "-x", [-inf, 1.0]),
        ("Exponential", (1.0, 1.0), "log(x)", [0.0, inf]),
        # The values of these two transformations
        # leave the representable reals quickly.
        ("Exponential", (0.1, 0.0), "exp(x)", [1.0, inf]),
        ("Exponential", (0.1, 0.0), "-exp(x)", [-inf, -1.0]),
        # sympy cannot compute the image of the real line
        # under this transformation;
        # its limits tell that it diverges towards the infinity.
        ("Normal", (0.0, 1.0), "abs(x)", [0.0, inf]),
        # sympy does not know these functions of OpenTURNS at all.
        ("Normal", (0.0, 1.0), "expm1(x)", [-1.0, inf]),
        ("Exponential", (1.0, 1.0), "log1p(x)", [log(2.0), inf]),
        ("Normal", (0.0, 1.0), "hypot(x,1)", [1.0, inf]),
        # The cube root of sympy is the principal complex one,
        # whereas the one of OpenTURNS is real.
        ("Normal", (0.0, 1.0), "cbrt(x)", [-inf, inf]),
    ],
    ids=[
        "exponential",
        "negation",
        "logarithm",
        "exponential-leaving-reals",
        "opposite-exponential-leaving-reals",
        "absolute-value",
        "expm1-unknown-to-sympy",
        "log1p-unknown-to-sympy",
        "hypot-unknown-to-sympy",
        "real-cube-root",
    ],
)
def test_support_of_diverging_transformation(
    interfaced_distribution, parameters, transformation, expected_support
) -> None:
    """Check the support of a diverging transformation of an unbounded support.

    OpenTURNS reports the range of a transformed distribution as finite
    on both sides; the end of this range is an arbitrary finite value
    on a side where the transformation diverges,
    and the support is reported as infinite there.
    The support is deduced from the limits of the transformation,
    which tell that it diverges at an end of the support;
    the other end is the numerical estimate of OpenTURNS, hence the tolerance.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution=interfaced_distribution,
            parameters=parameters,
            transformation=transformation,
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-3)
    assert_array_less(abs(distribution.range), inf)


@pytest.mark.parametrize(
    ("interfaced_distribution", "parameters", "transformation", "expected_support"),
    [
        # The transformation is not defined at 0,
        # where OpenTURNS evaluates it as not-a-number,
        # so its range misses the extremum 1 that the transformation attains there.
        ("Normal", (0.0, 1.0), "sin(x)/x", [-inf, inf]),
        ("Uniform", (-8.0, 8.0), "sin(x)/x", [-inf, inf]),
        # The transformation diverges at the lower end of the support,
        # which is a bound of the support and not an unbounded side,
        # so only the upper end of the range is untrustworthy.
        ("Uniform", (0.0, 1.0), "sin(x)/x^3", [sin(1.0), inf]),
        # sympy cannot determine the domain of continuity of these transformations,
        # which diverge at the lower end of the support.
        ("Uniform", (0.0, 1.0), "gamma(x)", [gamma(1.0), inf]),
        ("Uniform", (0.0, 1.0), "erf(x)/x^3", [erf(1.0), inf]),
        # This transformation is bounded around 0 but is not defined at it,
        # where OpenTURNS evaluates it as not-a-number,
        # so its range misses the extremum 2/sqrt(pi) that it attains there.
        ("Normal", (0.0, 1.0), "erf(x)/x", [-inf, inf]),
    ],
    ids=[
        "undefined-at-interior-extremum",
        "undefined-at-interior-extremum-bounded",
        "divergence-at-lower-bound",
        "gamma-divergence-at-lower-bound",
        "erf-over-cube-divergence-at-lower-bound",
        "bounded-but-undefined-at-zero",
    ],
)
def test_support_of_singular_transformation(
    interfaced_distribution, parameters, transformation, expected_support
) -> None:
    """Check the support of a transformation that is singular at a point of the support.

    OpenTURNS estimates the range of a `CompositeDistribution`
    by evaluating the transformation,
    which fails where the transformation is not defined
    and misses the extrema that it attains around such a point;
    the corresponding end of the range must not be reported as an end of the support.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution=interfaced_distribution,
            parameters=parameters,
            transformation=transformation,
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-11)


@pytest.mark.parametrize(
    "lower_bound",
    [0.0, -1e-9, -0.5],
    ids=["at-the-support", "within-the-rounding-error", "beyond-the-estimated-end"],
)
def test_truncation_bound_slack_accepted(lower_bound) -> None:
    """Check that a lower bound at or below the transformed support is accepted.

    The support of a transformed distribution is known up to a rounding error,
    which the comparison must tolerate;
    a bound accepted thanks to the slack does not widen the support.
    On a side unbounded before transformation,
    the lower end of the range is a numerical estimate
    that is not compared with a truncation bound.

    Args:
        lower_bound: The lower truncation bound
            of a normal distribution transformed by `x^2`,
            whose exact support is `[0, inf)` before truncation.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Normal",
            parameters=(0.0, 1.0),
            transformation="x^2",
            lower_bound=lower_bound,
        )
    )
    assert_allclose(distribution.support, array([0.0, inf]), rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    ("transformation", "bound_name", "bound", "expected_support"),
    [
        ("exp(x)", "lower_bound", -50000.0, [1.0, inf]),
        ("-exp(x)", "upper_bound", 10000.0, [-inf, -1.0]),
    ],
    ids=["increasing-lower-bound", "decreasing-upper-bound"],
)
def test_truncation_beyond_diverging_transformation_accepted(
    transformation, bound_name, bound, expected_support
) -> None:
    """Check that no bound is rejected on the side that is not infinite.

    The end of the numerical range that OpenTURNS estimates
    on the side where the transformation does not diverge
    is not compared with a truncation bound.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Exponential",
            parameters=(0.1, 0.0),
            transformation=transformation,
            **{bound_name: bound},
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-3)


@pytest.mark.parametrize(
    "rate", [1.0, 33.33, 100.0], ids=["small-rate", "large-rate", "very-large-rate"]
)
@pytest.mark.parametrize(
    ("transformation", "bound_name", "expected_support", "mean_sign"),
    [
        ("-x", "lower_bound", [0.0, 1.0], 1.0),
        ("x", "upper_bound", [-1.0, 0.0], -1.0),
    ],
    ids=["truncation-from-below", "truncation-from-above"],
)
def test_truncation_after_transformation_of_unbounded_support(
    rate, transformation, bound_name, expected_support, mean_sign
) -> None:
    """Check the truncation of a transformed distribution with unbounded support.

    OpenTURNS estimates the range of a `CompositeDistribution` numerically
    and reports it as finite even when the support is unbounded.
    This range must not be used to check the truncation bounds,
    otherwise a valid bound is rejected
    as soon as the numerical range is narrower than the truncation interval,
    i.e. when `rate` is large here.
    The mathematical bound on the truncated side must be the truncation bound,
    whether the truncation is from below or from above.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Exponential",
            parameters=(rate, -1.0),
            transformation=transformation,
            **{bound_name: 0.0},
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-12)
    # The density is proportional to exp(rate*x) on [0, 1] after the transformation
    # "-x" and to exp(-rate*x) on [-1, 0] after the transformation "x".
    expected_mean = mean_sign * (1.0 / (1.0 - exp(-rate)) - 1.0 / rate)
    assert distribution.mean == pytest.approx(expected_mean, rel=1e-7)


@pytest.mark.parametrize(
    ("bound_name", "bound", "expected_support", "mean_sign"),
    [
        ("lower_bound", -1.0, [-1.0, inf], 1.0),
        ("upper_bound", 1.0, [-inf, 1.0], -1.0),
    ],
    ids=["truncation-from-below", "truncation-from-above"],
)
def test_truncation_after_transformation_of_infinite_support(
    bound_name, bound, expected_support, mean_sign
) -> None:
    """Check the truncation of a transformed distribution with an infinite support.

    The numerical range that OpenTURNS estimates for a `CompositeDistribution`
    is finite on both sides, even when the support is infinite on both sides;
    the side that the truncation does not bound must thus be reported as infinite,
    instead of being read from that range.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution="Normal",
            parameters=(0.0, 1.0),
            transformation="2*x",
            **{bound_name: bound},
        )
    )
    assert_equal(distribution.support, array(expected_support))
    # The random variable is twice a standard normal one truncated to [-0.5, +inf),
    # or its opposite when the truncation is from above.
    expected_mean = mean_sign * 2.0 * truncnorm(-0.5, inf).mean()
    assert distribution.mean == pytest.approx(expected_mean, rel=1e-7)


@pytest.mark.parametrize(
    ("interfaced_distribution", "parameters", "transformation", "expected_support"),
    [
        # The transformation diverges at the finite lower end of the support.
        ("Uniform", (0.0, 1.0), "log(x)", [-50.0, 0.0]),
        # The transformation diverges at a point inside the support.
        ("Normal", (0.0, 1.0), "1/x", [-50.0, inf]),
    ],
    ids=[
        "pole-at-lower-bound",
        "pole-inside-support",
    ],
)
def test_truncation_after_transformation_with_pole(
    interfaced_distribution, parameters, transformation, expected_support
) -> None:
    """Check the truncation of a transformed distribution with a pole.

    The image of a support under a transformation with a pole is unbounded,
    even when the support is bounded,
    whereas the numerical range that OpenTURNS estimates is finite;
    a truncation bound beyond this range is thus valid.
    """
    distribution = OTDistribution(
        OTDistribution_Settings(
            interfaced_distribution=interfaced_distribution,
            parameters=parameters,
            transformation=transformation,
            lower_bound=-50.0,
        )
    )
    assert_allclose(distribution.support, array(expected_support), rtol=0.0, atol=1e-11)


def test_truncate_exception_after_transformation(snapshot) -> None:
    """Check the truncation bound check after transforming a bounded support."""
    with assert_exception(ValueError, snapshot):
        OTDistribution(
            OTDistribution_Settings(
                interfaced_distribution="Uniform",
                parameters=(1.0, 2.0),
                transformation="2*x",
                lower_bound=-1.0,
            )
        )


@pytest.mark.parametrize(
    (
        "dimension",
        "file_path",
        "directory_path",
        "file_name",
        "file_extension",
        "expected",
    ),
    [
        (1, "", "", "", "", "pdf_cdf.png"),
        (2, "", "", "", "", "pdf_cdf.png"),
        (2, "foo/bar.png", "", "", "", Path("foo/bar.png")),
        (2, "", "foo", "", "", Path("foo/pdf_cdf.png")),
        (2, "", "foo", "bar", "svg", Path("foo/bar.svg")),
    ],
)
def test_plot_save(
    dimension,
    file_path,
    directory_path,
    file_name,
    file_extension,
    expected,
    tmp_wd,
) -> None:
    """Check the file path computed by plot()."""
    triangular = OTTriangularDistribution()
    with patch.object(base_univariate, "save_show_figure") as mock_method:
        triangular.plot(
            show=False,
            save=True,
            file_path=file_path,
            directory_path=directory_path,
            file_name=file_name,
            file_extension=file_extension,
        )

        args = mock_method.call_args.args
        if isinstance(expected, Path):
            assert args[2] == expected
        else:
            assert args[2] == Path(tmp_wd / expected)


def test_compute_cdf_int32():
    """Check that openturns-based compute_cdf handles numpy.int32."""
    expected = pytest.approx(0.8, abs=0.1)
    x = int32(1)
    assert OTNormalDistribution().compute_cdf(x) == expected
    assert (
        OTJointDistribution(
            OTJointDistribution_Settings(
                marginal_settings=[OTNormalDistribution_Settings()]
            )
        ).compute_cdf([x])
        == expected
    )


@pytest.mark.parametrize(
    ("ot_cls", "settings", "ot_args", "str_"),
    [
        (
            Beta,
            OTBetaDistribution_Settings(),
            (2.0, 2.0, 0.0, 1.0),
            "Beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)",
        ),
        (
            Beta,
            OTBetaDistribution_Settings(alpha=0.1, beta=0.2, minimum=0.3, maximum=0.4),
            (0.1, 0.2, 0.3, 0.4),
            "Beta(lower=0.3, upper=0.4, alpha=0.1, beta=0.2)",
        ),
        (
            Dirac,
            OTDiracDistribution_Settings(),
            (),
            "Dirac(loc=0.0)",
        ),
        (
            Dirac,
            OTDiracDistribution_Settings(variable_value=1.0),
            (1.0,),
            "Dirac(loc=1.0)",
        ),
        (
            Exponential,
            OTExponentialDistribution_Settings(),
            (),
            "Exponential(rate=1.0, loc=0.0)",
        ),
        (
            Exponential,
            OTExponentialDistribution_Settings(rate=0.1, loc=0.2),
            (0.1, 0.2),
            "Exponential(rate=0.1, loc=0.2)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(),
            (*compute_mu_l_and_sigma_l(1.0, 1.0, 0.0), 0.0),
            "LogNormal(mu=1.0, sigma=1.0, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(mu=1.5, sigma=2.5, set_log=True),
            (1.5, 2.5, 0.0),
            "LogNormal(mu=1.5, sigma=2.5, loc=0.0)",
        ),
        (
            LogNormal,
            OTLogNormalDistribution_Settings(mu=1.2, sigma=1.3, location=0.4),
            (*compute_mu_l_and_sigma_l(1.2, 1.3, 0.4), 0.4),
            "LogNormal(mu=1.2, sigma=1.3, loc=0.4)",
        ),
        (Normal, OTNormalDistribution_Settings(), (), "Normal(mu=0.0, sigma=1.0)"),
        (
            Normal,
            OTNormalDistribution_Settings(mu=0.1, sigma=0.2),
            (0.1, 0.2),
            "Normal(mu=0.1, sigma=0.2)",
        ),
        (
            Triangular,
            OTTriangularDistribution_Settings(),
            (0.0, 0.5, 1.0),
            "Triangular(lower=0.0, mode=0.5, upper=1.0)",
        ),
        (
            Triangular,
            OTTriangularDistribution_Settings(minimum=0.1, mode=0.2, maximum=0.3),
            (0.1, 0.2, 0.3),
            "Triangular(lower=0.1, mode=0.2, upper=0.3)",
        ),
        (
            Uniform,
            OTUniformDistribution_Settings(),
            (0.0, 1.0),
            "Uniform(lower=0.0, upper=1.0)",
        ),
        (
            Uniform,
            OTUniformDistribution_Settings(minimum=0.1, maximum=0.2),
            (0.1, 0.2),
            "Uniform(lower=0.1, upper=0.2)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution_Settings(),
            (1.0, 1.0, 0.0),
            "WeibullMin(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMin,
            OTWeibullDistribution_Settings(location=0.1, scale=0.2, shape=0.3),
            (0.2, 0.3, 0.1),
            "WeibullMin(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution_Settings(use_weibull_min=False),
            (1.0, 1.0, 0.0),
            "WeibullMax(location=0.0, scale=1.0, shape=1.0)",
        ),
        (
            WeibullMax,
            OTWeibullDistribution_Settings(
                use_weibull_min=False, location=0.1, scale=0.2, shape=0.3
            ),
            (0.2, 0.3, 0.1),
            "WeibullMax(location=0.1, scale=0.2, shape=0.3)",
        ),
        (
            FiniteDiscreteDistribution,
            OTFiniteDiscreteDistribution_Settings(value_to_weight={0: 1, 1: 2}),
            ([[0], [1]], [1, 2]),
            "FiniteDiscreteDistribution(value_to_weight={0.0: 1.0, 1.0: 2.0})",
        ),
        (
            FiniteDiscreteDistribution,
            OTFiniteDiscreteDistribution_Settings(
                value_to_weight={(0, 0): 1, (1, 1): 2}
            ),
            ([[0, 0], [1, 1]], [1, 2]),
            (
                "FiniteDiscreteDistribution("
                "value_to_weight={(0.0, 0.0): 1.0, (1.0, 1.0): 2.0}"
                ")"
            ),
        ),
        (Bernoulli, OTBernoulliDistribution_Settings(), (), "Bernoulli(p=0.5)"),
        (
            Bernoulli,
            OTBernoulliDistribution_Settings(p=0.2),
            (0.2,),
            "Bernoulli(p=0.2)",
        ),
    ],
)
def test_specific_ot_distributions(ot_cls, settings, ot_args, str_) -> None:
    """Check the specific OpenTURNS-based distributions.

    Args:
        ot_cls: The OpenTURNS class.
        settings: The settings of the OT distribution.
        ot_args: The positional arguments to instantiate the OpenTURNS class.
        str_: The expected string representation of the distribution.
    """
    ot_distribution = distribution_factory.create_from_settings(settings)
    distribution = ot_distribution.distribution
    expected_distribution = ot_cls(*ot_args)
    assert distribution.getName() == expected_distribution.getName()
    assert distribution.getParameter() == expected_distribution.getParameter()
    assert str(ot_distribution) == str_


@pytest.mark.parametrize(
    ("class_name", "expected"),
    [
        ("OTBetaDistribution", "Beta(lower=0.0, upper=1.0, alpha=2.0, beta=2.0)"),
        ("OTDiracDistribution", "Dirac(loc=0.0)"),
        ("OTDistribution", "Uniform()"),
        ("OTExponentialDistribution", "Exponential(rate=1.0, loc=0.0)"),
        ("OTLogNormalDistribution", "LogNormal(mu=1.0, sigma=1.0, loc=0.0)"),
        ("OTNormalDistribution", "Normal(mu=0.0, sigma=1.0)"),
        ("OTTriangularDistribution", "Triangular(lower=0.0, mode=0.5, upper=1.0)"),
        ("OTUniformDistribution", "Uniform(lower=0.0, upper=1.0)"),
        ("OTWeibullDistribution", "WeibullMin(location=0.0, scale=1.0, shape=1.0)"),
    ],
)
def test_specific_ot_distributions_default(class_name, expected):
    """Check the OpenTURNS-based distributions using default settings."""
    distribution = distribution_factory.create(class_name)
    assert str(distribution) == expected


def test_lognormal_distribution():
    """Check that (mu,sigma)=(1,1) for the default log-normal distribution.

    We do this test because we transform the (mu, sigma) into (mu_l, sigma_l) and
    instantiate the wrapped distribution with it. This is a way to test this
    transformation.
    """
    distribution = OTLogNormalDistribution()
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.standard_deviation == pytest.approx(1.0)


def test_settings_truncation_error(snapshot):
    """Test the message raised in the case of a truncation error."""
    with assert_exception(ValueError, snapshot):
        OTDistribution_Settings(
            interfaced_distribution="Normal", lower_bound=0.0, upper_bound=-1
        )
