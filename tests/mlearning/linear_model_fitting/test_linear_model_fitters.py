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

from gemseo.mlearning.linear_model_fitting.factory import LinearModelFitterFactory
from gemseo.mlearning.linear_model_fitting.null_space import NullSpace

MONO_OUTPUT_ALGORITHMS = {
    "ElasticNetCV",
    "LARSCV",
    "LassoCV",
    "OrthogonalMatchingPursuitCV",
}

CLASS_NAMES = sorted(set(LinearModelFitterFactory().class_names) - {NullSpace.__name__})


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_default_settings(input_data, output_data, class_name, multioutput):
    """Check the linear model fitting algorithms with default settings."""
    if multioutput and class_name in MONO_OUTPUT_ALGORITHMS:
        return
    cls = LinearModelFitterFactory().get_class(class_name)
    coefficients = cls().fit(input_data, output_data)
    assert coefficients.shape == (4 if multioutput else 1, 3)


@pytest.mark.parametrize("class_name", CLASS_NAMES)
@pytest.mark.parametrize("fit_intercept", [False, True])
def test_custom_settings(
    input_data, output_data, class_name, multioutput, fit_intercept
):
    """Check the linear model fitting algorithms with custom settings."""
    if multioutput and class_name in MONO_OUTPUT_ALGORITHMS:
        return
    cls = LinearModelFitterFactory().get_class(class_name)
    settings = cls.Settings(fit_intercept=fit_intercept)
    coefficients = cls(settings=settings).fit(input_data, output_data)
    assert coefficients.shape == (4 if multioutput else 1, 3 if fit_intercept else 2)
