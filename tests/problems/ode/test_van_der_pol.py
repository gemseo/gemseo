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
#        :author: Isabelle Santos
#        :author: Giulio Gargantini
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import unittest

import pytest
from numpy import array
from numpy.linalg import norm

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.problems.ode.van_der_pol import VanDerPol


class TestVanDerPol(unittest.TestCase):
    """Test the VanDerPol class."""


parametrized_algo_names = pytest.mark.parametrize(
    "algo_name",
    [
        "RK45",
        "RK23",
        "DOP853",
        "Radau",
        "BDF",
        "LSODA",
    ],
)


@parametrized_algo_names
def test_run(algo_name) -> None:
    """Solve Van der Pol with the jacobian analytical expression."""
    problem = VanDerPol()
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=10e-6)
    assert problem.result.algorithm_has_converged
    assert norm(problem.result.state_trajectories) > 0
    assert (
        problem.result.algorithm_termination_message
        == "The solver successfully reached the "
        "end of the integration interval."
    )


def test_van_der_pol_jacobian_explicit_expression() -> None:
    """Validate the analytical expression of the jacobian."""
    problem = VanDerPol()
    problem.check_jacobian(array([0.0, 0.0]))
    ODESolverLibraryFactory().execute(problem, algo_name="Radau")
    assert problem.result.algorithm_has_converged


@parametrized_algo_names
def test_van_der_pol_with_initial_state(algo_name) -> None:
    """Solve Van der Pol for an initial condition that is not the default."""
    problem = VanDerPol(state=(1.0, -1.0))
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, first_step=10e-6)
    assert problem.result.algorithm_has_converged
    assert norm(problem.result.state_trajectories) > 0
    assert (
        problem.result.algorithm_termination_message
        == "The solver successfully reached the "
        "end of the integration interval."
    )
