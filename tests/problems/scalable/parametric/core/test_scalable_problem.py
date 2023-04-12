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
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.scalable_problem import ScalableProblem
from gemseo.utils.string_tools import MultiLineString
from numpy import array
from numpy.random import get_state
from numpy.random import rand
from numpy.random import seed
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal


@pytest.fixture(scope="module")
def default_scalable_problem() -> ScalableProblem():
    """The scalable problem with the default configuration."""
    return ScalableProblem()


@pytest.fixture(scope="module")
def custom_scalable_problem() -> ScalableProblem():
    """The scalable problem with a custom configuration."""
    return ScalableProblem(
        discipline_settings=[
            ScalableDisciplineSettings(1, 3),
            ScalableDisciplineSettings(2, 2),
            ScalableDisciplineSettings(3, 1),
        ],
        d_0=2,
    )


def test_str(default_scalable_problem):
    """Check the string representation of a scalable problem."""
    message = MultiLineString()
    message.add("Scalable problem")
    message.indent()
    message.add("MainDiscipline")
    message.indent()
    message.add("Inputs")
    message.indent()
    message.add("x_0 (1)")
    message.add("y_1 (1)")
    message.add("y_2 (1)")
    message.dedent()
    message.add("Outputs")
    message.indent()
    message.add("f (1)")
    message.add("c_1 (1)")
    message.add("c_2 (1)")
    message.dedent()
    message.dedent()
    message.add("ScalableDiscipline[1]")
    message.indent()
    message.add("Inputs")
    message.indent()
    message.add("x_0 (1)")
    message.add("x_1 (1)")
    message.add("y_2 (1)")
    message.dedent()
    message.add("Outputs")
    message.indent()
    message.add("y_1 (1)")
    message.dedent()
    message.dedent()
    message.add("ScalableDiscipline[2]")
    message.indent()
    message.add("Inputs")
    message.indent()
    message.add("x_0 (1)")
    message.add("x_2 (1)")
    message.add("y_1 (1)")
    message.dedent()
    message.add("Outputs")
    message.indent()
    message.add("y_2 (1)")
    assert str(default_scalable_problem) == str(message)


def test_instance_seed(default_scalable_problem):
    """Check the use of the NumPy seed for reproducibility."""
    state = get_state()

    # Same seed, same random state.
    ScalableProblem()
    assert_equal(get_state(), state)

    # New seed, new random state.
    ScalableProblem(seed=2)
    assert get_state()[1][0] != state[1][0]


def test_total_output_size(custom_scalable_problem):
    """Check the total output size."""
    assert custom_scalable_problem._p == 6


def test_discipline_names(custom_scalable_problem):
    """Check the disciplines of the custom scalable problem."""
    assert [discipline.name for discipline in custom_scalable_problem.disciplines] == [
        "MainDiscipline",
        "ScalableDiscipline[1]",
        "ScalableDiscipline[2]",
        "ScalableDiscipline[3]",
    ]


def test_main_discipline(default_scalable_problem):
    """Check the property main_discipline."""
    assert (
        default_scalable_problem.main_discipline
        == default_scalable_problem.disciplines[0]
    )


def test_scalable_disciplines(default_scalable_problem):
    """Check the property scalable_disciplines."""
    assert (
        default_scalable_problem.scalable_disciplines
        == default_scalable_problem.disciplines[1:]
    )


def test_scalable_discipline_coefficients(default_scalable_problem):
    """Check the coefficients of the scalable disciplines."""
    seed(1)
    for scalable_discipline, coupling_name in zip(
        default_scalable_problem.scalable_disciplines, ["y_2", "y_1"]
    ):
        coefficients = scalable_discipline.coefficients
        assert_equal(coefficients.D_i0, rand(1, 1))
        assert_equal(coefficients.D_ii, rand(1, 1))
        assert_equal(coefficients.C_ij[coupling_name], rand(1, 1))
        assert_equal(coefficients.a_i, rand(1))


def test_main_discipline_coefficients(default_scalable_problem):
    """Check the coefficients of the main disciplines."""
    coefficients = default_scalable_problem.main_discipline._MainDiscipline__t_i
    assert_almost_equal(coefficients, array([[-0.267], [-0.267]]), decimal=3)


def test_coefficients_custom(custom_scalable_problem):
    """Check the coefficients."""
    seed(1)
    for p_i, d_i, scalable_discipline, couplings in zip(
        [3, 2, 1],
        [1, 2, 3],
        custom_scalable_problem.scalable_disciplines,
        [(("y_2", 2), ("y_3", 1)), (("y_1", 3), ("y_3", 1)), (("y_1", 3), ("y_2", 2))],
    ):
        coefficients = scalable_discipline.coefficients
        assert_equal(coefficients.D_i0, rand(p_i, 2))
        assert_equal(coefficients.D_ii, rand(p_i, d_i))
        for coupling in couplings:
            assert_equal(coefficients.C_ij[coupling[0]], rand(p_i, coupling[1]))

        assert_equal(coefficients.a_i, rand(p_i))

    coefficients = custom_scalable_problem.main_discipline._MainDiscipline__t_i
    assert_almost_equal(coefficients[0], array([0.206, 0.206, 0.206]), decimal=3)
    assert_almost_equal(coefficients[1], array([0.206, 0.206]), decimal=3)
    assert_almost_equal(coefficients[2], array([0.206]), decimal=3)


def test_qp_problem(default_scalable_problem):
    """Check the quadratic programming problem resulting from the scalable one."""
    problem = default_scalable_problem.qp_problem
    assert_almost_equal(
        problem.Q,
        array([[2.449, 0.661, 0.041], [0.661, 1.074, 0.025], [0.041, 0.025, 0.017]]),
        decimal=3,
    )
    assert_almost_equal(problem.c, array([[-0.433], [-0.543], [-0.074]]), decimal=3)
    assert_almost_equal(problem.d, 0.25294177528006323, decimal=3)
    assert_almost_equal(
        problem.A,
        array(
            [
                [4.17047674e-01, 7.20339839e-01, 1.05614349e-05],
                [2.24435279e-01, 1.34170651e-01, 9.23405619e-02],
                [1.00000000e00, 0.00000000e00, 0.00000000e00],
                [0.00000000e00, 1.00000000e00, 0.00000000e00],
                [0.00000000e00, 0.00000000e00, 1.00000000e00],
                [-1.00000000e00, -0.00000000e00, -0.00000000e00],
                [-0.00000000e00, -1.00000000e00, -0.00000000e00],
                [-0.00000000e00, -0.00000000e00, -1.00000000e00],
            ]
        ),
        decimal=3,
    )
    assert_almost_equal(
        problem.b,
        array([0.57, 0.669, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        decimal=3,
    )


def test_compute_y(default_scalable_problem):
    """Check the method compute_y."""
    x = array([1.0, 1.0, 1.0])
    alpha = default_scalable_problem._ScalableProblem__alpha
    beta = default_scalable_problem._ScalableProblem__beta
    output = default_scalable_problem.compute_y(x)
    assert output.shape == (2,)
    assert_almost_equal(output, alpha + beta @ x)


def test_feasibility(default_scalable_problem):
    """Check the feasibility level."""
    assert (
        ScalableProblem(alpha=0.9).main_discipline()["c_1"]
        < default_scalable_problem.main_discipline()["c_1"]
        < ScalableProblem(alpha=0.1).main_discipline()["c_1"]
    )


def test_add_random_variables():
    """Check add_random_variables."""
    problem = ScalableProblem()
    assert "u_1" not in problem.scalable_disciplines[0].input_names_to_default_values
    assert "u_1" not in problem.scalable_disciplines[0].input_names

    problem = ScalableProblem(add_random_variables=True)
    assert "u_1" in problem.scalable_disciplines[0].input_names
    assert_equal(
        problem.scalable_disciplines[0].input_names_to_default_values["u_1"],
        array([0.0]),
    )
