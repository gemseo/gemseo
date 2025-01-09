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

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import interp
from numpy import linspace
from numpy import pi
from numpy import sqrt

from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.problems.ode.orbital_dynamics import OrbitalDynamics

if TYPE_CHECKING:
    from gemseo.typing import RealArray


@pytest.mark.parametrize(
    ("algo_name", "eccentricity", "atol"),
    [
        ("RK45", 0.5, 1.0e-2),
        ("RK23", 0.5, 1.0e-3),
        ("DOP853", 0.5, 1.0e-3),
        ("Radau", 0.5, 1.0e-3),
        ("BDF", 0.5, 1.0e-3),
        ("LSODA", 0.5, 1.0e-2),
        ("RK45", 0, 1.0e-2),
        ("RK45", 0.1, 1.0e-2),
        ("RK45", 0.8, 1.0e-2),
        ("Radau", 0.1, 1.0e-3),
        ("Radau", 0.0, 1.0e-3),
        ("Radau", 0.8, 1.0e-3),
    ],
)
def test_orbital(algo_name, eccentricity, atol) -> None:
    """Solve the orbital problem, checking that some characteristics of the elliptic
    orbit are verified:
    1) Vis Viva relation between position and velocity of the orbiting body;
    2) conservation of energy (kinetic plus potential);
    3) period of rotation, checked against its analytic expression;
    4) comparison with the analytic trajectory.
    """
    times = linspace(0, 7, 50)
    problem = OrbitalDynamics(eccentricity=eccentricity, times=times)
    ODESolverLibraryFactory().execute(problem, algo_name=algo_name, rtol=1.0e-5)
    x_exact, y_exact = problem.compute_analytic_solution(times)

    def check_orbital_constants(
        tt, xx, yy, vvx=None, vvy=None, algo_name_msg="Analytic"
    ):
        """"""

        def radius(x: RealArray, y: RealArray):
            return sqrt(x * x + y * y)

        def velocity(vx: RealArray, vy: RealArray):
            return sqrt(vx * vx + vy * vy)

        def vis_viva(x: RealArray, y: RealArray):
            return sqrt(2 / radius(x, y) - 1)

        def energy(r: RealArray, v: RealArray):
            return v * v / 2 - 1.0 / r

        def check_vis_viva(x_vector, y_vector, vx_vector, vy_vector):
            vel_vector = velocity(vx_vector, vy_vector)
            vis_viva_vector = vis_viva(x_vector, y_vector)
            return allclose(vel_vector, vis_viva_vector, atol=atol)

        def check_period(t_vector, x_vector, y_vector):
            x_interp = interp(2 * pi, t_vector, x_vector)
            y_interp = interp(2 * pi, t_vector, y_vector)
            return allclose((x_interp, y_interp), (1 - eccentricity, 0), atol=1.0e-1)

        def check_energy(x_vector, y_vector, vv_vector):
            r_vector = radius(x_vector, y_vector)
            energy_vector = energy(r_vector, vv_vector)
            init_energy = energy_vector[0]
            return allclose(energy_vector, init_energy, atol=atol)

        if vvx is None:
            vv = vis_viva(xx, yy)
        else:
            if not check_vis_viva(xx, yy, vvx, vvy):
                msg = (
                    f"\nFAILED check vis viva for e = {eccentricity}"
                    f"and algo = {algo_name_msg}"
                )
                raise ValueError(msg)
            vv = velocity(vvx, vvy)

        check_energy_bool = check_energy(xx, yy, vv)
        if not check_energy_bool:
            msg = (
                f"FAILED check energy for e = {eccentricity} and algo = {algo_name_msg}"
            )
            raise ValueError(msg)

        check_period_bool = check_period(tt, xx, yy)
        if not check_period_bool:
            msg = (
                f"FAILED check period for e = {eccentricity}"
                f" and algo = {algo_name_msg}."
            )
            raise ValueError(msg)

    check_orbital_constants(
        tt=times,
        xx=problem.result.state_trajectories[0],
        yy=problem.result.state_trajectories[1],
        vvx=problem.result.state_trajectories[2],
        vvy=problem.result.state_trajectories[3],
        algo_name_msg=algo_name,
    )

    if algo_name == "RK45":
        check_orbital_constants(tt=times, xx=x_exact, yy=y_exact)

    assert problem.result.algorithm_has_converged
    assert allclose(problem.result.state_trajectories[0], x_exact, atol=atol)
    assert allclose(problem.result.state_trajectories[1], y_exact, atol=atol)


@pytest.mark.parametrize("eccentricity", [0.0, 0.2, 0.5, 0.8])
def test_orbital_jacobian_explicit_expression(eccentricity) -> None:
    """Validate the analytical expression of the jacobian."""
    time_vect = array([0.0, 1.0])
    problem = OrbitalDynamics(eccentricity=eccentricity, times=time_vect)
    state = array([
        1 - eccentricity,
        0.0,
        0.0,
        sqrt((1 + eccentricity) / (1 - eccentricity)),
    ])
    problem.check_jacobian(state, step=1e-7, error_max=1e-6)
