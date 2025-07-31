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
from __future__ import annotations

import pytest

from gemseo.uncertainty.distributions.factory import DistributionFactory


@pytest.mark.parametrize("prefix", ["OT", "SP"])
@pytest.mark.parametrize(
    "distribution_name",
    [
        "BetaDistribution",
        "DiracDistribution",
        "Distribution",
        "ExponentialDistribution",
        "LogNormalDistribution",
        "NormalDistribution",
        "TriangularDistribution",
        "UniformDistribution",
        "WeibullDistribution",
    ],
)
def test_default_settings(prefix, distribution_name):
    """Test {prefix}{distribution_name}Distribution_Settings with default settings."""
    if distribution_name == "DiracDistribution" and prefix == "SP":
        pytest.skip("invalid parameter combination")

    cls = DistributionFactory().get_class(f"{prefix}{distribution_name}")
    from_args = cls(settings=cls.Settings())
    from_settings = cls()
    assert from_args.mean == from_settings.mean
    assert from_args.standard_deviation == from_settings.standard_deviation


@pytest.mark.parametrize("prefix", ["OT", "SP"])
@pytest.mark.parametrize(
    ("distribution_name", "kwargs"),
    [
        (
            "BetaDistribution",
            {"alpha": 1.23, "beta": 2.24, "minimum": 1.0, "maximum": 2.0},
        ),
        ("DiracDistribution", {"variable_value": 2.0}),
        ("Distribution", {"interfaced_distribution": "Normal"}),
        ("ExponentialDistribution", {"rate": 1.4, "loc": 2.5}),
        (
            "LogNormalDistribution",
            {"mu": 1.5, "sigma": 2.5, "location": 1.0, "set_log": True},
        ),
        ("NormalDistribution", {"mu": 1.4, "sigma": 3.4}),
        ("TriangularDistribution", {}),
        ("UniformDistribution", {"minimum": 4.0, "maximum": 6.0}),
        (
            "WeibullDistribution",
            {"location": 2.0, "scale": 4.0, "shape": 5.0, "use_weibull_min": False},
        ),
    ],
)
def test_user_settings(prefix, distribution_name, kwargs):
    """Test OTDiracDistribution_Settings with user settings."""
    if distribution_name == "DiracDistribution" and prefix == "SP":
        pytest.skip("invalid parameter combination")

    if distribution_name == "Distribution" and prefix == "SP":
        kwargs["interfaced_distribution"] = "norm"

    cls = DistributionFactory().get_class(f"{prefix}{distribution_name}")
    settings = cls.Settings(**kwargs)
    from_args = cls(settings=settings)
    from_settings = cls(**settings.model_dump())
    assert from_args.mean == from_settings.mean
