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
from numpy import array
from numpy.testing import assert_almost_equal
from pydantic import ValidationError

from gemseo.doe.oat_doe.oat_doe import OATDOE
from gemseo.doe.oat_doe.settings.oat_doe_settings import OATDOE_Settings
from gemseo.util.testing.helper import assert_exception


def test_oat_doe():
    """Check OAT DOE algo."""
    oat = OATDOE()
    a = oat.sample_unit_hypercube(
        3, OATDOE_Settings(initial_point=array([0.2, 0.8, 0.98]))
    )
    assert_almost_equal(
        a,
        array([
            [0.2, 0.8, 0.98],
            [0.25, 0.8, 0.98],
            [0.25, 0.85, 0.98],
            [0.25, 0.85, 0.93],
        ]),
    )


def test_oat_doe_step():
    """Check the step option of the OAT DOE algo."""
    oat = OATDOE()
    a = oat.sample_unit_hypercube(
        3,
        OATDOE_Settings(initial_point=array([0.2, 0.8, 0.98]), step=0.1),
    )
    assert_almost_equal(
        a,
        array([[0.2, 0.8, 0.98], [0.3, 0.8, 0.98], [0.3, 0.9, 0.98], [0.3, 0.9, 0.88]]),
    )


def test_oat_doe_unit_upper_bound():
    """Check the OAT DOE when adding the step would reach the coordinate 1.

    The perturbed coordinate must stay in the open interval $(0,1)$,
    where the quantile function of an input variable is finite;
    the step is taken downwards as soon as `u+step` reaches 1,
    and not only when it exceeds 1.
    """
    oat = OATDOE()
    a = oat.sample_unit_hypercube(2, OATDOE_Settings(initial_point=array([0.95, 0.9])))
    assert_almost_equal(a, array([[0.95, 0.9], [0.9, 0.9], [0.9, 0.95]]))


@pytest.mark.parametrize("step", [0.5, 0.6])
def test_oat_doe_step_too_large(step, snapshot):
    """Check that a step reaching 0.5 is rejected.

    From a coordinate `u` in `[1-step, step]`,
    both `u+step` and `u-step` leave the open interval $(0,1)$
    where the quantile function of an input variable is finite.
    """
    with assert_exception(ValidationError, snapshot):
        OATDOE_Settings(initial_point=array([0.5]), step=step)
