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

import numpy
import pytest
from numpy import array

from gemseo.problems.mdo.sobieski.core.problem import SobieskiProblem
from gemseo.problems.mdo.sobieski.core.structure import (
    SobieskiStructure as CoreStructure,
)
from gemseo.problems.mdo.sobieski.core.utils import SobieskiBase
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


def test_dfuelweightdtoverc(problem) -> None:
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(names=SobieskiStructure().io.input_grammar)
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


def test_dfuelweightd_ar(problem) -> None:
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(names=SobieskiStructure().io.input_grammar)
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


def test_dfuelweightdsref(problem) -> None:
    h = 1e-30
    sr = problem.structure
    indata = problem.get_default_inputs(names=SobieskiStructure().io.input_grammar)
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


def test_jac_structure(problem) -> None:
    """"""
    sr = SobieskiStructure("complex128")
    indata = problem.get_default_inputs(names=sr.io.input_grammar)
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    indata = problem.get_default_inputs_feasible(names=sr.io.input_grammar)
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    indata = problem.get_default_inputs_equilibrium(names=sr.io.input_grammar)
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    for _ in range(5):
        indata = problem.get_random_input(names=sr.io.input_grammar, seed=1)
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


def test_jac2_sobieski_struct(problem) -> None:
    inpt_data = {
        "y_31": array([6555.68459235 + 0j]),
        "y_21": array([50606.9742 + 0j]),
        "x_shared": array([
            5.00000000e-02 + 0j,
            4.50000000e04 + 0j,
            1.60000000e00 + 0j,
            5.50000000e00 + 0j,
            5.50000000e01 + 0j,
            1.00000000e03 + 0j,
        ]),
        "x_1": array([0.25 + 0j, 1.0 + 0j]),
    }

    st = SobieskiStructure("complex128")
    assert st.check_jacobian(inpt_data, derr_approx="complex_step", step=1e-30)


def test_logarithm_invalid_domain():
    """Test that the mass term is not a number when the weight ratio is not positive.

    In this test, the arguments of `SobieskiStructure._execute` are chosen so that
    the weight ratio is negative.
    """
    assert numpy.isnan(
        CoreStructure(SobieskiBase(SobieskiBase.DataType.FLOAT))._execute(
            tc_ratio=0.01,
            aspect_ratio=8.5,
            sweep=70.0,
            wing_area=1000.0,
            taper_ratio=0.1,
            wingbox_area=0.7700018565802997,
            lift=124646.13088472793,
            engine_mass=7671.188123402499,
            c_0=2000.0,
            c_1=25000.0,
            c_2=6.0,
        )[1]
    )
