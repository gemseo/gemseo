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
from numpy.random import default_rng
from numpy.testing import assert_almost_equal
from numpy.testing import assert_equal

from gemseo import SEED
from gemseo.problems.scalable.parametric.core.scalable_discipline_settings import (
    ScalableDisciplineSettings,
)
from gemseo.problems.scalable.parametric.core.scalable_problem import ScalableProblem
from gemseo.utils.repr_html import REPR_HTML_WRAPPER


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
    expected = """Scalable problem
   MainDiscipline
      Inputs
         x_0 (1)
         y_1 (1)
         y_2 (1)
      Outputs
         f (1)
         c_1 (1)
         c_2 (1)
   ScalableDiscipline[1]
      Inputs
         x_0 (1)
         x_1 (1)
         y_2 (1)
      Outputs
         y_1 (1)
   ScalableDiscipline[2]
      Inputs
         x_0 (1)
         x_2 (1)
         y_1 (1)
      Outputs
         y_2 (1)"""
    assert repr(default_scalable_problem) == str(default_scalable_problem) == expected


def test_repr_html(default_scalable_problem):
    """Check the string representation of a scalable problem."""
    assert default_scalable_problem._repr_html_() == REPR_HTML_WRAPPER.format(
        "Scalable problem<br/>"
        "<ul>"
        "<li>MainDiscipline"
        "<ul>"
        "<li>Inputs"
        "<ul>"
        "<li>x_0 (1)</li>"
        "<li>y_1 (1)</li>"
        "<li>y_2 (1)</li>"
        "</ul>"
        "</li>"
        "<li>Outputs"
        "<ul>"
        "<li>f (1)</li>"
        "<li>c_1 (1)</li>"
        "<li>c_2 (1)</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "<li>ScalableDiscipline[1]"
        "<ul>"
        "<li>Inputs"
        "<ul>"
        "<li>x_0 (1)</li>"
        "<li>x_1 (1)</li>"
        "<li>y_2 (1)</li>"
        "</ul>"
        "</li>"
        "<li>Outputs"
        "<ul>"
        "<li>y_1 (1)</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "<li>ScalableDiscipline[2]"
        "<ul>"
        "<li>Inputs"
        "<ul>"
        "<li>x_0 (1)</li>"
        "<li>x_2 (1)</li>"
        "<li>y_1 (1)</li>"
        "</ul>"
        "</li>"
        "<li>Outputs"
        "<ul>"
        "<li>y_2 (1)</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


def test_instance_seed(default_scalable_problem):
    """Check the use of the NumPy seed for reproducibility."""
    beta = ScalableProblem()._ScalableProblem__beta.sum()

    # Same seed, same beta.
    assert ScalableProblem()._ScalableProblem__beta.sum() == beta

    # New seed, new beta.
    assert ScalableProblem(seed=2)._ScalableProblem__beta.sum() != beta


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
    rng = default_rng(SEED)
    for scalable_discipline, coupling_name in zip(
        default_scalable_problem.scalable_disciplines, ["y_2", "y_1"]
    ):
        coefficients = scalable_discipline.coefficients
        assert_equal(coefficients.D_i0, rng.random((1, 1)))
        assert_equal(coefficients.D_ii, rng.random((1, 1)))
        assert_equal(coefficients.C_ij[coupling_name], rng.random((1, 1)))
        assert_equal(coefficients.a_i, rng.random(1))


def test_main_discipline_coefficients(default_scalable_problem):
    """Check the coefficients of the main disciplines."""
    coefficients = default_scalable_problem.main_discipline._MainDiscipline__t_i
    assert_almost_equal(coefficients, array([[-0.519], [-0.519]]), decimal=3)


def test_coefficients_custom(custom_scalable_problem):
    """Check the coefficients."""
    rng = default_rng(SEED)
    for p_i, d_i, scalable_discipline, couplings in zip(
        [3, 2, 1],
        [1, 2, 3],
        custom_scalable_problem.scalable_disciplines,
        [(("y_2", 2), ("y_3", 1)), (("y_1", 3), ("y_3", 1)), (("y_1", 3), ("y_2", 2))],
    ):
        coefficients = scalable_discipline.coefficients
        assert_equal(coefficients.D_i0, rng.random((p_i, 2)))
        assert_equal(coefficients.D_ii, rng.random((p_i, d_i)))
        for coupling in couplings:
            assert_equal(coefficients.C_ij[coupling[0]], rng.random((p_i, coupling[1])))

        assert_equal(coefficients.a_i, rng.random(p_i))

    coefficients = custom_scalable_problem.main_discipline._MainDiscipline__t_i
    assert_almost_equal(coefficients[0], array([-0.59, -0.59, -0.59]), decimal=3)
    assert_almost_equal(coefficients[1], array([-0.59, -0.59]), decimal=3)
    assert_almost_equal(coefficients[2], array([-0.59]), decimal=3)


def test_qp_problem(default_scalable_problem):
    """Check the quadratic programming problem resulting from the scalable one."""
    problem = default_scalable_problem.qp_problem
    assert_almost_equal(
        problem.Q,
        array([[5.972, 0.793, 2.356], [0.793, 0.209, 0.335], [2.356, 0.335, 1.755]]),
        decimal=3,
    )
    assert_almost_equal(problem.c, array([[-1.931], [-0.281], [-1.423]]), decimal=3)
    assert_almost_equal(problem.d, 0.577, decimal=3)  # noqa: FURB152
    assert_almost_equal(
        problem.A,
        array([
            [0.687, 0.277, 0.038],
            [1.23, 0.168, 0.936],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [-1.0, -0.0, -0.0],
            [-0.0, -1.0, -0.0],
            [-0.0, -0.0, -1.0],
        ]),
        decimal=3,
    )
    assert_almost_equal(
        problem.b,
        array([0.566, 1.277, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
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
