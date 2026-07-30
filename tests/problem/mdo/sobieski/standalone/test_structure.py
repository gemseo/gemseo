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

from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.problem.mdo.sobieski.standalone.problem import SobieskiProblem
from gemseo.problem.mdo.sobieski.standalone.structure import (
    SobieskiStructure as CoreStructure,
)
from gemseo.problem.mdo.sobieski.standalone.util import SobieskiBase
from gemseo.util.derivative.check.discipline import DisciplineJacobianChecker

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


@pytest.fixture(scope="module")
def checker() -> DisciplineJacobianChecker:
    """The discipline Jacobian checker."""
    discipline = SobieskiStructure("complex128")
    return DisciplineJacobianChecker(discipline)


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
    checker = DisciplineJacobianChecker(sr)
    indata = problem.get_default_inputs(names=sr.io.input_grammar)
    assert checker.check(
        indata,
        atol=THRESHOLD,
        rtol=THRESHOLD,
        approximation_mode="complex_step",
        step=1e-30,
    )

    indata = problem.get_default_inputs_feasible(names=sr.io.input_grammar)
    assert checker.check(
        indata,
        atol=THRESHOLD,
        rtol=THRESHOLD,
        approximation_mode="complex_step",
        step=1e-30,
    )

    indata = problem.get_default_inputs_equilibrium(names=sr.io.input_grammar)
    assert checker.check(
        indata,
        atol=THRESHOLD,
        rtol=THRESHOLD,
        approximation_mode="complex_step",
        step=1e-30,
    )

    for _ in range(5):
        indata = problem.get_random_input(names=sr.io.input_grammar, seed=1)
        assert checker.check(
            indata,
            atol=THRESHOLD,
            rtol=THRESHOLD,
            approximation_mode="complex_step",
            step=1e-30,
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


@pytest.mark.parametrize("i", [0, 1, 2])
def test_jac_structure_coefficients(checker, i) -> None:
    """Check the Jacobian when the coefficients are not set to their default values."""
    input_data = {
        f"c_{i}": array([
            checker._discipline.sobieski_problem.structure.constants[i] * 1.2
        ])
    }
    assert checker.check(
        input_data,
        atol=THRESHOLD,
        rtol=THRESHOLD,
        approximation_mode="complex_step",
        step=1e-30,
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
    checker = DisciplineJacobianChecker(st)
    assert checker.check(inpt_data, approximation_mode="complex_step", step=1e-30)


def test_logarithm_invalid_domain():
    """Test that the mass term is not a number when the weight ratio is not positive.

    In this test, the arguments of `SobieskiStructure._execute` are chosen so that
    the weight ratio is negative.
    """
    assert numpy.isnan(
        CoreStructure(SobieskiBase(SobieskiBase.DataType.FLOAT))._execute(
            0.01,
            8.5,
            70.0,
            1000.0,
            0.1,
            0.7700018565802997,
            124646.13088472793,
            7671.188123402499,
            False,
            2000.0,
            25000.0,
            6.0,
        )[1]
    )
