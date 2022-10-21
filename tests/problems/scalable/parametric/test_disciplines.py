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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.problems.scalable.parametric.disciplines import TMMainDiscipline
from gemseo.problems.scalable.parametric.disciplines import TMSubDiscipline
from gemseo.problems.scalable.parametric.problem import TMScalableProblem
from numpy import array


def test_scalable_problem():
    pbm = TMScalableProblem()
    assert isinstance(pbm.main_discipline, TMMainDiscipline)
    for discipline in pbm.sub_disciplines:
        assert isinstance(discipline, TMSubDiscipline)
    variables_names = ["x_shared", "x_local_0", "x_local_1", "y_0", "y_1"]
    c_shared = [array([[0.417022]]), array([[0.30233257]])]
    c_local = [array([[0.72032449]]), array([[0.14675589]])]
    c_coupling = [{"y_1": array([[0.00011437]])}, {"y_0": array([[0.09233859]])}]
    c_constraint = [array([0.18626021]), array([0.34556073])]
    assert pbm.n_disciplines == 2
    assert len(pbm.disciplines) == pbm.n_disciplines + 1
    for i in range(2):
        disc = pbm.disciplines[1 + i]
        assert c_shared[i] == pytest.approx(disc.model.c_shared, abs=1e-8)
        assert c_local[i] == pytest.approx(disc.model.c_local, abs=1e-8)
        value = list(disc.model.c_coupling.values())[0]
        name = list(disc.model.c_coupling.keys())[0]
        assert c_coupling[i][name] == pytest.approx(value, abs=1e-8)
    disc = pbm.disciplines[0]
    assert c_constraint[0] == pytest.approx(disc.model.coefficients[0], abs=1e-8)
    assert c_constraint[1] == pytest.approx(disc.model.coefficients[1], abs=1e-8)
    assert set(pbm.design_space.variables_names) == set(variables_names)


def test_main_discipline():
    c_constraint = [array([1.0, 2.0]), array([3.0, 4.0, 5.0])]
    default_inputs = {
        "x_shared": array([0.5]),
        "y_0": array([2.0, 3.0]),
        "y_1": array([4.0, 5.0, 6.0]),
    }
    system = TMMainDiscipline(c_constraint, default_inputs)
    inputs_names = ["x_shared", "y_0", "y_1"]
    outputs_names = ["cstr_0", "cstr_1", "obj"]
    assert set(inputs_names) == set(system.get_input_data_names())
    assert set(outputs_names) == set(system.get_output_data_names())
    system.execute()
    assert system.local_data["obj"] == pytest.approx(array([49.0 / 3.0]), abs=1e-8)
    assert system.local_data["cstr_0"] == pytest.approx(array([-1.0, -0.5]), abs=1e-8)
    assert system.local_data["cstr_1"] == pytest.approx(
        array([-1.0 / 3.0, -0.25, -0.2]), abs=1e-8
    )
    system.linearize(force_all=True)
    assert system.check_jacobian(threshold=1e-6)


def test_sub_discipline():
    index = 0
    default_inputs = {
        "x_shared": array([0.5]),
        "x_local_0": array([1.0, 2.0]),
        "y_1": array([3.0, 4.0, 5.0]),
    }
    c_shared = array([[2.0], [3.0]])
    c_local = array([[2.0, 3.0], [2.0, 3.0]])
    c_coupling = {"y_1": array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])}
    disc = TMSubDiscipline(index, c_shared, c_local, c_coupling, default_inputs)
    inputs_names = ["x_shared", "x_local_0", "y_1"]
    outputs_names = ["y_0"]
    assert set(inputs_names) == set(disc.get_input_data_names())
    assert set(outputs_names) == set(disc.get_output_data_names())
    disc.execute()
    assert disc.local_data["y_0"] == pytest.approx(array([2.52631579, 2.425]), abs=1e-8)
    disc.linearize(force_all=True)
    assert disc.check_jacobian(threshold=1e-6)


def test_noised_sub_discipline():
    index = 0
    default_inputs = {
        "x_shared": array([0.5]),
        "x_local_0": array([1.0, 2.0]),
        "y_1": array([3.0, 4.0, 5.0]),
        "u_local_0": array([0.1, -0.1]),
    }
    c_shared = array([[2.0], [3.0]])
    c_local = array([[2.0, 3.0], [2.0, 3.0]])
    c_coupling = {"y_1": array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])}
    disc = TMSubDiscipline(index, c_shared, c_local, c_coupling, default_inputs)
    inputs_names = ["x_shared", "x_local_0", "y_1", "u_local_0"]
    outputs_names = ["y_0"]
    assert set(inputs_names) == set(disc.get_input_data_names())
    assert set(outputs_names) == set(disc.get_output_data_names())
    disc.execute()
    assert disc.local_data["y_0"] == pytest.approx(array([2.62631579, 2.325]), abs=1e-8)
    disc.linearize(force_all=True)
    assert disc.check_jacobian(threshold=1e-6)
