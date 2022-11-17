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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author : Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiMission
from numpy import array

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


def test_dweightratio_dwt(problem):
    h = 1e-30
    sr = problem.mission
    indata = problem.get_default_inputs_equilibrium(
        names=SobieskiMission().get_input_data_names()
    )
    y_14 = indata["y_14"]
    lin_weightratio = sr._SobieskiMission__compute_dweightratio_dwt(y_14[0], y_14[1])
    y_14[0] += 1j * h
    assert lin_weightratio == pytest.approx(
        sr._SobieskiMission__compute_weight_ratio(y_14[0], y_14[1]).imag / h,
        abs=1e-8,
    )


def test_dlnweightratio_dwt(problem):
    h = 1e-30
    sr = problem.mission
    indata = problem.get_default_inputs_equilibrium(
        names=SobieskiMission().get_input_data_names()
    )
    y_14 = indata["y_14"]
    lin_weightratio = sr._SobieskiMission__compute_dlnweightratio_dwt(y_14[0], y_14[1])
    y_14[0] += 1j * h
    import cmath

    assert lin_weightratio == pytest.approx(
        cmath.log(sr._SobieskiMission__compute_weight_ratio(y_14[0], y_14[1])).imag / h,
        abs=1e-8,
    )


def test_d_range_d_wt(problem):
    h = 1e-30
    sr = problem.mission
    indata = problem.get_default_inputs_equilibrium(
        names=SobieskiMission().get_input_data_names()
    )
    y_14 = indata["y_14"]
    y_24 = indata["y_24"]
    y_34 = indata["y_34"]
    x_shared = indata["x_shared"]
    sqrt_theta = sr._SobieskiMission__compute_sqrt_theta(x_shared[1])
    lin_range = sr._SobieskiMission__compute_drange_dtotalweight(
        x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0], sqrt_theta
    )
    y_14[0] += 1j * h
    assert lin_range == pytest.approx(
        sr._SobieskiMission__compute_range(
            x_shared[1], x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0]
        ).imag
        / h,
        abs=1e-8,
    )


def test_d_range_d_wf(problem):
    h = 1e-30
    sr = problem.mission
    indata = problem.get_default_inputs_equilibrium(
        names=SobieskiMission().get_input_data_names()
    )
    y_14 = indata["y_14"]
    y_24 = indata["y_24"]
    y_34 = indata["y_34"]
    x_shared = indata["x_shared"]
    sqrt_theta = sr._SobieskiMission__compute_sqrt_theta(x_shared[1])
    lin_range = sr._SobieskiMission__compute_drange_dfuelweight(
        x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0], sqrt_theta
    )
    y_14[1] += 1j * h
    assert lin_range == pytest.approx(
        sr._SobieskiMission__compute_range(
            x_shared[1], x_shared[2], y_14[0], y_14[1], y_24[0], y_34[0]
        ).imag
        / h,
        abs=1e-8,
    )


def test_jac_mission(problem):
    sr = SobieskiMission("complex128")
    assert sr.check_jacobian(
        threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )
    inpt_data = {
        "y_24": array([4.16647508]),
        "x_shared": array(
            [
                5.00000000e-02,
                4.50000000e04,
                1.60000000e00,
                5.50000000e00,
                5.50000000e01,
                1.00000000e03,
            ]
        ),
        "y_34": array([1.10754577]),
        "y_14": array([50808.33445658, 7306.20262124]),
    }

    assert sr.check_jacobian(inpt_data, derr_approx="complex_step", step=1e-30)
