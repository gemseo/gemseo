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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from numpy import array
from numpy import complex128
from numpy import float64
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def problem():
    return SobieskiProblem()


@pytest.fixture(scope="module")
def dtype():
    return float64


def relative_norm(x, x_ref):
    return norm(x - x_ref) / norm(x_ref)


def test_get_optimum_range(problem):
    reference_range = array([3963.98])
    assert problem.optimum_range == reference_range


def test_get_sob_cstr(problem):
    gs = problem.get_default_inputs(["g_1", "g_2", "g_3"])
    c1 = problem.get_sobieski_constraints(
        gs["g_1"], gs["g_2"], gs["g_3"], true_cstr=True
    )
    assert len(c1) == 10
    c2 = problem.get_sobieski_constraints(
        gs["g_1"], gs["g_2"], gs["g_3"], true_cstr=False
    )
    assert len(c2) == 12


def test_normalize(problem):
    x0 = problem.initial_design
    x0_adim = problem.normalize_inputs(x0)
    x0_dim = problem.unnormalize_inputs(x0_adim)
    assert relative_norm(x0_dim, x0) == 0.0


@pytest.mark.parametrize(
    "dtype,expected", [("complex128", complex128), ("float64", float64)]
)
def test_design_space(dtype, expected):
    design_space = SobieskiProblem(dtype).design_space
    for variable in design_space.values():
        assert variable.value.dtype == expected


def test_constants(problem):
    cref = zeros(5)
    # Constants of problem
    cref[0] = 2000.0  # minimum fuel weight
    cref[1] = 25000.0  # miscellaneous weight
    cref[2] = 6.0  # Maximum load factor
    cref[3] = 4360.0  # Engine weight reference
    cref[4] = 0.01375  # Minimum drag coefficient
    c = problem.constants
    assert relative_norm(c, cref) == 0.0


def test_init():
    cmod = zeros(5)
    # Constants of problem
    cmod[0] = 2000.0  # minimum fuel weight
    cmod[1] = 25000.0  # miscellaneous weight
    cmod[2] = 6.0  # Maximum load factor
    cmod[3] = 4360.0  # Engine weight reference
    cmod[4] = 0.01375  # Minimum drag coefficient
    problem = SobieskiProblem("complex128")
    assert (cmod == problem.constants).all()


def test_wrong_dtype():
    with pytest.raises(ValueError, match="foo"):
        SobieskiProblem("foo")


def test_get_default_inputs_feasible(problem):
    indata = problem.get_default_inputs_feasible("x_1")
    refdata = problem.get_x0_feasible("x_1")
    assert (indata["x_1"] == refdata).all()


def test_get_random_inputs(problem):
    problem.get_random_input(names=None, seed=1)
    assert len(problem.get_random_input(names=["x_1", "x_2"], seed=1)) == 2


def test_get_bounds(problem):
    lb_ref = array((0.1, 0.75, 0.75, 0.1, 0.01, 30000.0, 1.4, 2.5, 40.0, 500.0))
    ub_ref = array((0.4, 1.25, 1.25, 1.0, 0.09, 60000.0, 1.8, 8.5, 70.0, 1500.0))
    l_b, u_b = problem.design_bounds
    assert relative_norm(l_b, lb_ref) == 0.0
    assert relative_norm(u_b, ub_ref) == 0.0


def test_get_bounds_tuple(problem):
    """"""
    bounds = array(
        [
            (0.1, 0.4),
            (0.75, 1.25),
            (0.75, 1.25),
            (0.1, 1),
            (0.01, 0.09),
            (30000.0, 60000.0),
            (1.4, 1.8),
            (2.5, 8.5),
            (40.0, 70.0),
            (500.0, 1500.0),
        ]
    )
    lower, upper = problem.design_bounds
    assert_equal(lower, bounds[:, 0])
    assert_equal(upper, bounds[:, 1])


def test_poly_approx(problem, dtype):
    """test polynomial function approximation."""
    # Reference value from octave computation for polyApprox function
    ff_reference = 1.02046767  # Octave computation
    mach_ini = 1.6
    h_ini = 45000.0
    t_ini = 0.5
    s = array([mach_ini, h_ini, t_ini])
    snew = array([1.5, 50000.0, 0.75], dtype=dtype)
    flag = array([2, 4, 2], dtype=dtype)
    bound = array([0.25, 0.25, 0.25], dtype=dtype)
    ao_coeff = zeros(1, dtype=dtype)
    ai_coeff = zeros(3, dtype=dtype)
    aij_coeff = zeros((3, 3), dtype=dtype)
    ff = problem._SobieskiProblem__base.compute_polynomial_approximation(
        s, snew, flag, bound, ao_coeff, ai_coeff, aij_coeff
    )
    assert ff == pytest.approx(ff_reference, abs=1e-6)


def test_weight(problem, dtype):
    """blackbox_structure function test."""
    # Reference value from octaves computation for blackbox_structure
    # function
    y_1_reference = array([3.23358440e04, 7.30620262e03, 1.00000000e00], dtype=dtype)
    y_14_reference = array([32335.84397838, 7306.20262124], dtype=dtype)
    y_12_reference = array([3.23358440e04, 1.00000000e00], dtype=dtype)
    g_1_reference = array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=dtype)

    i0 = problem.initial_design
    i0 = array(i0, dtype=dtype)
    i0[0] = i0[0]
    #       [0.25 + 1j * h, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0, 1000.0]
    x_1 = i0[:2]
    x_shared = i0[4:]
    #         - x_1(0) : wing taper ratio
    #         - x_1(1) : wingbox x-sectional area as poly. funct
    #         - Z(0) : thickness/chord ratio
    #         - Z(1) : altitude
    #         - Z(2) : Mach
    #         - Z(3) : aspect ratio
    #         - Z(4) : wing sweep
    #         - Z(5) : wing surface area
    y_21 = ones(1, dtype=dtype)
    y_31 = ones(1, dtype=dtype)

    y_1, _, y_12, y_14, g_1 = problem.structure.execute(
        x_shared, y_21, y_31, x_1, true_cstr=True
    )

    # Check results regression
    assert relative_norm(y_1, y_1_reference) == pytest.approx(0.0, abs=1e-2)
    assert relative_norm(y_12, y_12_reference) == pytest.approx(0.0, abs=1e-2)
    assert relative_norm(y_14, y_14_reference) == pytest.approx(0.0, abs=1e-2)
    assert relative_norm(g_1, g_1_reference) == pytest.approx(0.0, abs=1e-2)


def test_dragpolar(problem, dtype):
    """blackbox_aerodynamics function test."""
    # Reference value from octave computation for blackbox_structure
    # function
    y_2_reference = array([3.23358440e04, 1.25620121e04, 2.57409751e00])
    y_21_reference = array([32335.84397838])
    y_23_reference = array([12562.07000284])
    y_24_reference = array([2.57408564])
    g_2_reference = array([1.0])

    i0 = array(problem.initial_design, dtype=dtype)
    x_1 = i0[:2]
    x_2 = array([i0[2]], dtype=dtype)
    x_shared = i0[4:]
    #         - x_1(0) : wing taper ratio
    #         - x_1(1) : wingbox x-sectional area as poly. funct
    #         - Z(0) : thickness/chord ratio
    #         - Z(1) : altitude
    #         - Z(2) : Mach
    #         - Z(3) : aspect ratio
    #         - Z(4) : wing sweep
    #         - Z(5) : wing surface area
    #         y_12 = ones((2),dtype=dtype)
    y_21 = ones(1, dtype=dtype)
    y_31 = ones(1, dtype=dtype)
    y_32 = ones(1, dtype=dtype)

    # Preserve initial values for polynomial calculations

    _, _, y_12, _, _ = problem.structure.execute(
        x_shared, y_21, y_31, x_1, true_cstr=True
    )
    y_2, y_21, y_23, y_24, g_2 = problem.aerodynamics.execute(
        x_shared, y_12, y_32, x_2, true_cstr=True
    )
    # Check results regression
    assert relative_norm(y_2, y_2_reference) == pytest.approx(0.0, abs=1e-2)
    assert relative_norm(y_21, y_21_reference) == pytest.approx(0.0, abs=1e-3)
    assert relative_norm(y_23, y_23_reference) == pytest.approx(0.0, abs=1e-2)
    assert relative_norm(y_24, y_24_reference) == pytest.approx(0.0, abs=1e-6)
    assert relative_norm(g_2, g_2_reference) == pytest.approx(0.0, abs=1e-6)


def test_power(problem, dtype):
    """blackbox_propulsion function test."""
    # Reference value from octave computation for blackbox_structure
    # function
    y_3_reference = array([1.10754577e00, 6.55568459e03, 5.17959175e-01])
    y_34_reference = array([1.10754577])
    y_31_reference = array([6555.68459235])
    y_32_reference = array([0.51796156])
    g_3_reference = array([0.51796156, 1.0, 0.16206032])

    i0 = array(problem.initial_design, dtype=dtype)

    x_1 = i0[:2]
    x_2 = array([i0[2]], dtype=dtype)
    x_3 = array([i0[3]], dtype=dtype)
    x_shared = i0[4:]
    #         - x_1(0) : wing taper ratio
    #         - x_1(1) : wingbox x-sectional area as poly. funct
    #         - Z(0) : thickness/chord ratio
    #         - Z(1) : altitude
    #         - Z(2) : Mach
    #         - Z(3) : aspect ratio
    #         - Z(4) : wing sweep
    #         - Z(5) : wing surface area
    #         y_12 = ones((2),dtype=dtype)
    y_21 = ones(1, dtype=dtype)
    y_31 = ones(1, dtype=dtype)
    y_32 = ones(1, dtype=dtype)

    # Preserve initial values for polynomial calculations

    _, _, y_12, _, _ = problem.structure.execute(
        x_shared, y_21, y_31, x_1, true_cstr=True
    )
    _, y_21, y_23, _, _ = problem.aerodynamics.execute(
        x_shared, y_12, y_32, x_2, true_cstr=True
    )
    y_3, y_34, y_31, y_32, g_3 = problem.propulsion.execute(
        x_shared, y_23, x_3, true_cstr=True
    )

    # Check results regression

    assert relative_norm(y_3, y_3_reference) == pytest.approx(0.0, abs=1e-3)
    assert relative_norm(y_31, y_31_reference) == pytest.approx(0.0, abs=1e-3)
    assert relative_norm(y_32, y_32_reference) == pytest.approx(0.0, abs=1e-6)
    assert relative_norm(y_34, y_34_reference) == pytest.approx(0.0, abs=1e-6)
    assert relative_norm(g_3, g_3_reference) == pytest.approx(0.0, abs=1e-6)


def test_range(problem, dtype):
    """blackbox_mission function test."""
    # Reference value from octave computation for blackbox_structure
    # function
    y_4_reference = array([545.88197472055879])

    # return array((0.25, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0,
    # 1000.0))
    i0 = array(problem.initial_design, dtype=dtype)

    x_1 = i0[:2]
    x_2 = array([i0[2]], dtype=dtype)
    x_3 = array([i0[3]], dtype=dtype)
    x_shared = i0[4:]
    #         - x_1(0) : wing taper ratio
    #         - x_1(1) : wingbox x-sectional area as poly. funct
    #         - Z(0) : thickness/chord ratio
    #         - Z(1) : altitude
    #         - Z(2) : Mach
    #         - Z(3) : aspect ratio
    #         - Z(4) : wing sweep
    #         - Z(5) : wing surface area
    #         y_12 = ones((2),dtype=dtype)
    y_21 = ones(1, dtype=dtype)
    y_31 = ones(1, dtype=dtype)
    y_32 = ones(1, dtype=dtype)

    # Preserve initial values for polynomial calculations
    _, _, y_12, y_14, _ = problem.structure.execute(x_shared, y_21, y_31, x_1)

    _, y_21, y_23, y_24, _ = problem.aerodynamics.execute(x_shared, y_12, y_32, x_2)

    _, y_34, y_31, y_32, _ = problem.propulsion.execute(x_shared, y_23, x_3)

    y_4 = problem.mission.execute(x_shared, y_14, y_24, y_34)

    assert y_4[0] == pytest.approx(y_4_reference[0], abs=1e-3)


def test_range_h35000(problem, dtype):
    """blackbox_mission function test."""
    # Reference value from octave computation for one MDA loop
    # function
    y_4_reference = array([352.508])

    # return array((0.25, 1., 1., 0.5, 0.05, 45000.0, 1.6, 5.5, 55.0,
    # 1000.0))
    i0 = array(problem.initial_design, dtype=dtype)

    x_1 = i0[:2]
    x_2 = array([i0[2]], dtype=dtype)
    x_3 = array([i0[3]], dtype=dtype)
    x_shared = i0[4:]
    x_shared[1] = 35000
    #         - x_1(0) : wing taper ratio
    #         - x_1(1) : wingbox x-sectional area as poly. funct
    #         - Z(0) : thickness/chord ratio
    #         - Z(1) : altitude
    #         - Z(2) : Mach
    #         - Z(3) : aspect ratio
    #         - Z(4) : wing sweep
    #         - Z(5) : wing surface area
    #         y_12 = ones((2),dtype=dtype)
    y_21 = ones(1, dtype=dtype)
    y_31 = ones(1, dtype=dtype)
    y_32 = ones(1, dtype=dtype)

    # Preserve initial values for polynomial calculations
    _, _, y_12, y_14, _ = problem.structure.execute(x_shared, y_21, y_31, x_1)

    _, y_21, y_23, y_24, _ = problem.aerodynamics.execute(x_shared, y_12, y_32, x_2)

    _, y_34, y_31, y_32, _ = problem.propulsion.execute(x_shared, y_23, x_3)

    y_4 = problem.mission.execute(x_shared, y_14, y_24, y_34)
    assert y_4 == pytest.approx(y_4_reference, abs=1e-2)


def test_optimum_gs(problem):
    """MDA analysis of the optimum sample from Sobieski and check range value."""

    # Reference value from octave computation for blackbox_structure function
    #         y_4_reference = problem.get_sobieski_optimum_range()

    x_optimum = problem.optimum_design

    problem._SobieskiProblem__compute_mda(x_optimum)


def test_constraints(problem):
    """MDA analysis of the optimum sample from Sobieski and check range value."""

    # Reference value from octave computation for blackbox_structure function
    #         y_4_reference = problem.get_sobieski_optimum_range()

    x_optimum = problem.optimum_design

    (
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        g_1,
        g_2,
        g_3,
    ) = problem._SobieskiProblem__compute_mda(x_optimum, true_cstr=True)
    constraints_values = problem.get_sobieski_constraints(
        g_1, g_2, g_3, true_cstr=False
    )
    for i in range(constraints_values.shape[0]):
        assert constraints_values[i].real <= 0.0


def test_ineq_constraints(problem):
    """"""
    #         y_4_reference = problem.get_sobieski_optimum_range()

    x_optimum = problem.optimum_design

    g_1, g_2, g_3 = problem.get_constraints(x_optimum, true_cstr=False)
    for g in (g_1, g_2, g_3):
        for i in range(g.shape[0]):
            assert g[i] <= 0.0


def test_x0_gs(problem, dtype):
    """MDA analysis of the initial sample from Sobieski and check range value."""

    # Reference value from octave computation for MDA
    # function
    y_4_reference = array([535.79388428])

    x0 = array(problem.initial_design, dtype=dtype)

    (
        _,
        _,
        _,
        y_4,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = problem._SobieskiProblem__compute_mda(x0)
    assert int(y_4[0].real) == int(y_4_reference[0])


def test_h35000(problem, dtype):
    """MDA analysis of the initial sample from Sobieski with modified altitude to test
    conditions on altitude in code."""

    # Reference value from octave computation for MDA
    # function
    y_4_reference = array([340.738])

    x0 = array(problem.initial_design, dtype=dtype)
    x0[5] = 3.50000000e04
    (
        _,
        _,
        _,
        y_4,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = problem._SobieskiProblem__compute_mda(x0)
    assert int(y_4[0].real) == int(y_4_reference[0])


def test_x0_optimum(problem, dtype):
    """MDA analysis of the initial sample from Sobieski and check range value."""

    # Reference value from octave computation for blackbox_structure
    # function
    y_4_ref = 3963.19894068
    x0 = array(problem.optimum_design, dtype=dtype)

    (
        _,
        _,
        _,
        y_4,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        g_1,
        g_2,
        g_3,
    ) = problem._SobieskiProblem__compute_mda(x0, true_cstr=True)
    constraints_values = problem.get_sobieski_constraints(g_1, g_2, g_3)
    assert (constraints_values <= 1e-6).all()
    assert y_4[0].real == pytest.approx(y_4_ref, abs=1e0)


@pytest.mark.parametrize(
    "design_variables,use_original_order",
    [
        (["x_shared", "x_1", "x_2", "x_3"], False),
        (["x_1", "x_2", "x_3", "x_shared"], True),
    ],
)
def test_original_design_variables_order(design_variables, use_original_order):
    """Check the design space with original variables order."""
    coupling_variables = [
        "y_14",
        "y_32",
        "y_31",
        "y_24",
        "y_34",
        "y_23",
        "y_21",
        "y_12",
    ]

    problem = SobieskiProblem()
    problem.USE_ORIGINAL_DESIGN_VARIABLES_ORDER = use_original_order
    variables_names = design_variables + coupling_variables
    assert problem.design_space.variables_names == variables_names
    assert problem.design_space_with_physical_naming.variables_names == variables_names
    assert problem.design_space_with_physical_naming.variables_names == variables_names
