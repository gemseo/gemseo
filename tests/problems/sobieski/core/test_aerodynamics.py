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
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from numpy import array

THRESHOLD = 1e-12


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem("complex128")


def test_dk_d_mach(problem):
    """"""
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    lin_k = sr_aero._SobieskiAerodynamics__compute_dk_aero_dmach(
        x_shared[2], x_shared[4]
    ).real
    x_shared[2] += 1j * h
    assert lin_k == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_k_aero(x_shared[2], x_shared[4]).imag
        / h,
        abs=1e-12,
    )


def test_dk_dsweep(problem):
    """"""
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    x_shared[1] = 35000.0
    x_shared[4] += 1j * h
    assert sr_aero._SobieskiAerodynamics__compute_dk_aero_dsweep(
        x_shared[2], x_shared[4]
    ).real == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_k_aero(x_shared[2], x_shared[4]).imag
        / h,
        abs=1e-12,
    )


def test_d_c_dmin_dsweep(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    x_shared[1] = 35000.0
    lin_cd = sr_aero._SobieskiAerodynamics__compute_dcdmin_dsweep(
        x_shared[0], x_shared[4]
    ).real
    x_shared[4] += 1j * h
    fo1 = 0.95 + 1j * 0
    assert lin_cd == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cd_min(
            x_shared[0], x_shared[4], fo1
        ).imag
        / h,
        abs=1e-12,
    )


def test_d_cd_dsweep(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    x_shared[1] = 35000.0
    fo1 = 0.95 + 1j * 0
    cl = 0.0916697016134 + 0j
    fo2 = 1.00005 + 0j
    lin_cd = sr_aero._SobieskiAerodynamics__compute_dcd_dsweep(
        x_shared[0], x_shared[2], x_shared[4], cl, fo2
    ).real
    x_shared[4] += 1j * h
    cdmin = sr_aero._SobieskiAerodynamics__compute_cd_min(x_shared[0], x_shared[4], fo1)
    sr_aero._SobieskiAerodynamics__compute_k_aero(x_shared[2], x_shared[4])
    assert lin_cd == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cd(cl, fo2, cdmin).imag / h,
        abs=1e-12,
    )


def test_d_cd_d_mach(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    y_12 = indata["y_12"] * 0.1
    fo1 = 0.95 + 1j * 0
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    cl = sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0])
    fo2 = 1.00005 + 0j
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_k_aero(x_shared[2], x_shared[4])
    lin_cd = sr_aero._SobieskiAerodynamics__compute_dcd_dmach(
        x_shared[0], x_shared[2], x_shared[4], x_shared[5], y_12[0], fo2
    ).real
    x_shared[2] += 1j * h
    cdmin = sr_aero._SobieskiAerodynamics__compute_cd_min(x_shared[0], x_shared[4], fo1)
    assert lin_cd == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cd(cl, fo2, cdmin).imag / h,
        abs=1e-4,
    )


def test_d_cd_dsref(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    y_12 = indata["y_12"]
    fo1 = 0.95 + 1j * 0
    fo2 = 1.00005 + 0j
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    sr_aero._SobieskiAerodynamics__compute_k_aero(x_shared[2], x_shared[4])
    sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0])
    lin_cd = sr_aero._SobieskiAerodynamics__compute_dcd_dsref(x_shared[5], fo2)
    x_shared[5] += 1j * h
    cl = sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0])
    cdmin = sr_aero._SobieskiAerodynamics__compute_cd_min(x_shared[0], x_shared[4], fo1)
    assert lin_cd == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cd(cl, fo2, cdmin).imag / h,
        abs=1e-12,
    )


def test_d_cl_dh(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    y_12 = indata["y_12"]
    y_12[0] = 5.06069742e04 + 0.0j
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    lin_cl = sr_aero._SobieskiAerodynamics__compute_dcl_dh(
        x_shared[1], x_shared[2], x_shared[5], y_12[0]
    ).real
    x_shared[1] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_cl == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0]).imag / h,
        abs=1e-12,
    )

    sr_aero = problem.aerodynamics
    x_shared[1] = 35000.0
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    lin_cl = sr_aero._SobieskiAerodynamics__compute_dcl_dh(
        x_shared[1], x_shared[2], x_shared[5], y_12[0]
    ).real
    x_shared[1] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_cl == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0]).imag / h,
        abs=1e-12,
    )


def test_d_cl_dsref(problem):
    """"""
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    y_12 = indata["y_12"]
    y_12[0] = 5.06069742e04 + 0.0j
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0])
    lin_cl = sr_aero._SobieskiAerodynamics__compute_dcl_dsref(x_shared[5])
    x_shared[5] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0])
    assert lin_cl == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0]).imag / h,
        abs=1e-12,
    )


def test_d_cl_d_mach(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"][:]
    y_12 = indata["y_12"]
    y_12[0] = 5.06069742e04 + 0.0j
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    lin_cl = sr_aero._SobieskiAerodynamics__compute_dcl_dmach(
        x_shared[1], x_shared[5], y_12[0]
    ).real
    x_shared[2] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_cl == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0]).imag / h,
        abs=1e-12,
    )

    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    x_shared = indata["x_shared"][:]
    x_shared[1] = 35000.0
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    lin_cl = sr_aero._SobieskiAerodynamics__compute_dcl_dmach(
        x_shared[1], x_shared[5], y_12[0]
    ).real
    x_shared[2] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_cl == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_cl(x_shared[5], y_12[0]).imag / h,
        abs=1e-12,
    )


def test_drho_v2_dh(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    lin_rho_v2 = sr_aero._SobieskiAerodynamics__compute_drhov2_dh(
        x_shared[1], x_shared[2]
    ).real
    x_shared[1] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    assert lin_rho_v2 == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rhov2().imag / h, abs=1e-12
    )

    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    x_shared[1] = 35000.0
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    lin_rho_v2 = sr_aero._SobieskiAerodynamics__compute_drhov2_dh(
        x_shared[1], x_shared[2]
    ).real
    x_shared[1] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_rho_v2 == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rhov2().imag / h, abs=1e-1
    )


def test_drho_v2_d_m(problem):
    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    h = 1e-30
    x_shared = indata["x_shared"]
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    lin_rho_v2 = sr_aero._SobieskiAerodynamics__compute_drhov2_dmach(x_shared[1]).real
    x_shared[2] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()

    assert lin_rho_v2 == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rhov2().imag / h, abs=1e-1
    )

    sr_aero = problem.aerodynamics
    indata = problem.get_default_inputs(
        names=SobieskiAerodynamics().get_input_data_names()
    )
    x_shared = indata["x_shared"]
    x_shared[1] = 35000.0
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    lin_rho_v2 = sr_aero._SobieskiAerodynamics__compute_drhov2_dmach(x_shared[1]).real
    x_shared[2] += 1j * h
    sr_aero._SobieskiAerodynamics__compute_rho_v(x_shared[2], x_shared[1])
    sr_aero._SobieskiAerodynamics__compute_rhov2()
    assert lin_rho_v2 == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rhov2().imag / h, abs=1e-1
    )


def test_dv_d_mach(problem):
    """"""
    sr_aero = problem.aerodynamics
    h = 1e-30
    mach = 1.8
    altitude = 45000.0
    lin_v = sr_aero._SobieskiAerodynamics__compute_dv_dmach(altitude)
    assert lin_v == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rho_v(mach + 1j * h, altitude)[1].imag
        / h,
        abs=1e-12,
    )

    sr_aero = problem.aerodynamics
    h = 1e-30
    mach = 1.4
    altitude = 35000.0
    lin_v = sr_aero._SobieskiAerodynamics__compute_dv_dmach(altitude)
    assert lin_v == pytest.approx(
        sr_aero._SobieskiAerodynamics__compute_rho_v(mach + 1j * h, altitude)[1].imag
        / h,
        abs=1e-12,
    )


def test_d_v_dh_drho_dh(problem):
    sr_aero = problem.aerodynamics
    h = 1e-30
    mach = 1.6
    altitude = 45000.0
    d_vdh_drhodh_ref = (
        array(
            sr_aero._SobieskiAerodynamics__compute_rho_v(mach, altitude + 1j * h)
        ).imag
        / h
    )
    d_vdh_ref = d_vdh_drhodh_ref[0]
    drhodh_ref = d_vdh_drhodh_ref[1]

    d_vdh_drhodh = array(
        sr_aero._SobieskiAerodynamics__compute_drho_dh_dv_dh(mach, altitude)
    )
    d_vdh = d_vdh_drhodh[0].real
    drhodh = d_vdh_drhodh[1].real

    assert d_vdh_ref == pytest.approx(d_vdh, 1e-4)
    assert drhodh_ref == pytest.approx(drhodh, 1e-4)

    altitude = 35000.0
    d_vdh_drhodh_ref = (
        array(
            sr_aero._SobieskiAerodynamics__compute_rho_v(mach, altitude + 1j * h)
        ).imag
        / h
    )
    d_vdh_ref = d_vdh_drhodh_ref[0]
    drhodh_ref = d_vdh_drhodh_ref[1]

    d_vdh_drhodh = array(
        sr_aero._SobieskiAerodynamics__compute_drho_dh_dv_dh(mach, altitude)
    )
    d_vdh = d_vdh_drhodh[0].real
    drhodh = d_vdh_drhodh[1].real
    assert d_vdh_ref == pytest.approx(d_vdh, 1e-4)
    assert drhodh_ref == pytest.approx(drhodh, 1e-4)


def test_jac_aero(problem):
    """"""
    sr = SobieskiAerodynamics("complex128")
    indata = problem.get_default_inputs(names=sr.get_input_data_names())
    assert sr.check_jacobian(
        indata, threshold=THRESHOLD, derr_approx="complex_step", step=1e-30
    )
    #
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
