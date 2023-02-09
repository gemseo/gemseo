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
#      :author: Damien Guenot - 18 mars 2016
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.core.structure import SobieskiStructure as CoreStructure
from gemseo.problems.sobieski.core.utils import SobieskiBase
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from numpy import array

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


def test_dfuelweightdtoverc(problem):
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(
        names=SobieskiStructure().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    lin_wf = sr._SobieskiStructure__compute_dfuelwing_dtoverc(x_shared[3], x_shared[5])
    x_shared[0] += 1j * h
    assert lin_wf == pytest.approx(
        sr._SobieskiStructure__compute_fuelwing_weight(
            x_shared[0], x_shared[3], x_shared[5]
        ).imag
        / h,
        abs=1e-8,
    )


def test_dfuelweightd_ar(problem):
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(
        names=SobieskiStructure().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    lin_wf = sr._SobieskiStructure__compute_dfuelwing_dar(
        x_shared[0], x_shared[3], x_shared[5]
    )
    x_shared[3] += 1j * h
    assert lin_wf == pytest.approx(
        sr._SobieskiStructure__compute_fuelwing_weight(
            x_shared[0], x_shared[3], x_shared[5]
        ).imag
        / h,
        abs=1e-8,
    )


def test_dfuelweightdsref(problem):
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(
        names=SobieskiStructure().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    lin_wf = sr._SobieskiStructure__compute_dfuelwing_dsref(
        x_shared[0], x_shared[3], x_shared[5]
    )
    x_shared[5] += 1j * h
    assert lin_wf == pytest.approx(
        sr._SobieskiStructure__compute_fuelwing_weight(
            x_shared[0], x_shared[3], x_shared[5]
        ).imag
        / h,
        abs=1e-8,
    )


def test_jac_structure(problem):
    """"""
    sr = SobieskiStructure("complex128")
    indata = problem.get_default_inputs(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    indata = problem.get_default_inputs_feasible(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    indata = problem.get_default_inputs_equilibrium(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    for _ in range(5):
        indata = problem.get_random_input(names=sr.get_input_data_names(), seed=1)
        assert sr.check_jacobian(
            indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
        )

    core_s = CoreStructure(SobieskiBase("complex128"))
    core_s._SobieskiStructure__aero_center = core_s.base.compute_aero_center(
        indata["x_1"][0]
    )
    core_s._SobieskiStructure__half_span = core_s.base.compute_half_span(
        indata["x_shared"][3], indata["x_shared"][5]
    )
    core_s._SobieskiStructure__dadimlift_dlift = (
        core_s._SobieskiStructure__compute_dadimlift_dlift(indata["y_21"])
    )
    core_s._SobieskiStructure__derive_constraints(
        sr.jac,
        indata["x_shared"][0],
        indata["x_shared"][3],
        indata["x_shared"][5],
        indata["x_1"][0],
        indata["x_1"][1],
        indata["y_21"][0],
        true_cstr=True,
    )


def test_jac2_sobieski_struct(problem):
    inpt_data = {
        "y_31": array([6555.68459235 + 0j]),
        "y_21": array([50606.9742 + 0j]),
        "x_shared": array(
            [
                5.00000000e-02 + 0j,
                4.50000000e04 + 0j,
                1.60000000e00 + 0j,
                5.50000000e00 + 0j,
                5.50000000e01 + 0j,
                1.00000000e03 + 0j,
            ]
        ),
        "x_1": array([0.25 + 0j, 1.0 + 0j]),
    }

    st = SobieskiStructure("complex128")
    assert st.check_jacobian(inpt_data, derr_approx="complex_step", step=1e-30)
