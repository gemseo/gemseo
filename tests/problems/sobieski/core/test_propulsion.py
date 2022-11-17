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
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


def test_d_esf_ddrag(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    drag = indata["y_23"][0]
    throttle = indata["x_3"][0]
    lin_esf = sr._SobieskiPropulsion__compute_desf_ddrag(throttle)
    drag += 1j * h
    assert lin_esf == pytest.approx(
        sr._SobieskiPropulsion__compute_esf(drag, throttle).imag / h, abs=1e-12
    )


def test_d_esf_dthrottle(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    drag = indata["y_23"][0]
    throttle = indata["x_3"][0]
    lin_esf = sr._SobieskiPropulsion__compute_desf_dthrottle(drag, throttle)
    throttle += 1j * h
    assert lin_esf == pytest.approx(
        sr._SobieskiPropulsion__compute_esf(drag, throttle).imag / h, abs=1e-4
    )


def test_blackbox_propulsion(problem):
    indata = problem.get_default_inputs(names=["x_shared", "y_23", "x_3"])
    x_shared = indata["x_shared"]
    y_23 = indata["y_23"]
    x_3 = indata["x_3"]
    _, _, _, _, g_3 = problem.propulsion.execute(x_shared, y_23, x_3, true_cstr=False)

    _, _, _, _, g_3_t = problem.propulsion.execute(x_shared, y_23, x_3, true_cstr=True)

    assert len(g_3) == len(g_3_t) + 1
    jac = problem.propulsion._SobieskiPropulsion__initialize_jacobian(False)

    jac_t = problem.propulsion._SobieskiPropulsion__initialize_jacobian(True)

    for var in ["x_shared", "x_3"]:
        assert jac["g_3"][var].shape[0] == jac_t["g_3"][var].shape[0] + 1

    problem.propulsion.linearize(x_shared, y_23, x_3, true_cstr=True)


def test_d_we_dthrottle(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    drag = indata["y_23"][0]
    throttle = indata["x_3"][0]
    d_esf_dthrottle = sr._SobieskiPropulsion__compute_desf_dthrottle(drag, throttle)
    esf = sr._SobieskiPropulsion__compute_esf(drag, throttle)
    lin_we = sr._SobieskiPropulsion__compute_dengineweight_dvar(esf, d_esf_dthrottle)

    throttle += 1j * h
    esf = sr._SobieskiPropulsion__compute_esf(drag, throttle)
    assert lin_we == pytest.approx(
        sr._SobieskiPropulsion__compute_engine_weight(esf).imag / h, abs=1e-8
    )


def test_d_we_ddrag(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    drag = indata["y_23"][0]
    throttle = indata["x_3"][0]
    d_esf_ddrag = sr._SobieskiPropulsion__compute_desf_ddrag(throttle)
    esf = sr._SobieskiPropulsion__compute_esf(drag, throttle)
    lin_we = sr._SobieskiPropulsion__compute_dengineweight_dvar(esf, d_esf_ddrag)

    drag += 1j * h
    esf = sr._SobieskiPropulsion__compute_esf(drag, throttle)
    assert lin_we == pytest.approx(
        sr._SobieskiPropulsion__compute_engine_weight(esf).imag / h, abs=1e-8
    )


def test_d_sfc_dthrottle(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    throttle = indata["x_3"][0]

    lin_sfc = sr._SobieskiPropulsion__compute_dsfc_dthrottle(
        x_shared[1], x_shared[2], throttle
    )
    throttle += 1j * h

    assert lin_sfc == pytest.approx(
        sr._SobieskiPropulsion__compute_sfc(x_shared[1], x_shared[2], throttle).imag
        / h,
        abs=1e-8,
    )


def test_d_sfc_dh(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    throttle = indata["x_3"][0]

    x_shared = indata["x_shared"]
    lin_sfc = sr._SobieskiPropulsion__compute_dsfc_dh(
        x_shared[1], x_shared[2], throttle
    )

    x_shared[1] += 1j * h
    assert lin_sfc == pytest.approx(
        sr._SobieskiPropulsion__compute_sfc(x_shared[1], x_shared[2], throttle).imag
        / h,
        abs=1e-8,
    )


def test_d_sfc_d_m(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    throttle = indata["x_3"][0] * 16168.6

    x_shared = indata["x_shared"]
    lin_sfc = sr._SobieskiPropulsion__compute_dsfc_dmach(
        x_shared[1], x_shared[2], throttle
    )

    x_shared[2] += 1j * h
    assert lin_sfc == pytest.approx(
        sr._SobieskiPropulsion__compute_sfc(x_shared[1], x_shared[2], throttle).imag
        / h,
        abs=1e-8,
    )


def test_dthrottle_constraint_dthrottle(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    x_3 = indata["x_3"]
    lin_throttle = sr._SobieskiPropulsion__compute_dthrconst_dthrottle(
        x_shared[1], x_shared[2]
    )
    x_3[0] += 1j * h
    assert lin_throttle == pytest.approx(
        sr._SobieskiPropulsion__compute_throttle_constraint(
            x_shared[1], x_shared[2], x_3[0]
        ).imag
        / h,
        abs=1e-8,
    )


def test_dthrottle_constraint_dh(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    x_3 = indata["x_3"]
    lin_throttle = sr._SobieskiPropulsion__compute_dthrcons_dh(
        x_shared[1], x_shared[2], x_3[0]
    )
    x_shared[1] += 1j * h
    assert lin_throttle == pytest.approx(
        sr._SobieskiPropulsion__compute_throttle_constraint(
            x_shared[1], x_shared[2], x_3[0]
        ).imag
        / h,
        abs=1e-8,
    )


def test_dthrottle_constraint_dmach(problem):
    h = 1e-30
    sr = problem.propulsion
    indata = problem.get_default_inputs(
        names=SobieskiPropulsion().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    x_3 = indata["x_3"]
    lin_throttle = sr._SobieskiPropulsion__compute_dthrconst_dmach(
        x_shared[1], x_shared[2], x_3[0]
    )
    x_shared[2] += 1j * h
    assert lin_throttle == pytest.approx(
        sr._SobieskiPropulsion__compute_throttle_constraint(
            x_shared[1], x_shared[2], x_3[0]
        ).imag
        / h,
        abs=1e-8,
    )


def test_jac_prop(problem):
    sr = SobieskiPropulsion("complex128")
    indata = problem.get_default_inputs(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )
    #
    indata = problem.get_default_inputs_feasible(names=sr.get_input_data_names())
    assert sr.check_jacobian(indata, derr_approx="complex_step", step=1e-30)

    indata = problem.get_default_inputs_equilibrium(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )

    for _ in range(5):
        indata = problem.get_random_input(names=sr.get_input_data_names(), seed=1)
        assert sr.check_jacobian(
            indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
        )
