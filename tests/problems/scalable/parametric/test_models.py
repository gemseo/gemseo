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
from gemseo.problems.scalable.parametric.core.models import TMMainModel
from gemseo.problems.scalable.parametric.core.models import TMSubModel
from gemseo.problems.scalable.parametric.core.variables import get_constraint_name
from gemseo.problems.scalable.parametric.core.variables import get_coupling_name
from gemseo.problems.scalable.parametric.core.variables import get_x_local_name
from gemseo.problems.scalable.parametric.core.variables import OBJECTIVE_NAME
from gemseo.problems.scalable.parametric.core.variables import X_SHARED_NAME
from numpy import array


@pytest.fixture
def varnames():
    """Variables names."""
    xsh = X_SHARED_NAME
    xl_0 = get_x_local_name(0)
    xl_1 = get_x_local_name(1)
    y_0 = get_coupling_name(0)
    y_1 = get_coupling_name(1)
    return xsh, xl_0, xl_1, y_0, y_1


def test_tm_main_model(varnames):
    """Test the behavior of TMMainModel.

    :param list(str) varnames: list of variable names.
    """
    xsh, _, _, y_0, y_1 = varnames
    c_0 = get_constraint_name(0)
    c_1 = get_constraint_name(1)
    obj = OBJECTIVE_NAME
    c_constraint = [array([1.0, 2.0]), array([3.0, 4.0, 5.0])]
    default_inputs = {
        xsh: array([0.5]),
        y_0: array([2.0, 3.0]),
        y_1: array([4.0, 5.0, 6.0]),
    }
    model = TMMainModel(c_constraint, default_inputs)
    inputs_names = [xsh, y_0, y_1]
    outputs_names = [c_0, c_1, obj]
    assert set(inputs_names) == set(model.inputs_names)
    assert set(outputs_names) == set(model.outputs_names)
    for name in inputs_names:
        assert model.inputs_sizes[name] == len(default_inputs[name])
    assert model.outputs_sizes[obj] == 1
    assert model.outputs_sizes[c_0] == len(c_constraint[0])
    assert model.outputs_sizes[c_1] == len(c_constraint[1])

    result = model()
    assert result[obj] == pytest.approx(array([49.0 / 3.0]), abs=1e-8)
    assert result[c_0] == pytest.approx(array([-1.0, -0.5]), abs=1e-8)
    assert result[c_1] == pytest.approx(array([-1.0 / 3.0, -0.25, -0.2]), abs=1e-8)

    result = model(jacobian=True)
    assert isinstance(result, dict)
    assert obj in result
    assert c_0 in result
    assert c_1 in result

    assert xsh in result[obj]
    assert result[obj][xsh].shape == (1, 1)

    assert y_0 in result[obj]
    assert result[obj][y_0].shape == (1, 2)

    assert y_1 in result[obj]
    assert result[obj][y_1].shape == (1, 3)

    assert y_0 in result[c_0]
    assert result[c_0][y_0].shape == (2, 2)

    assert y_1 in result[c_1]
    assert result[c_1][y_1].shape == (3, 3)


def test_tm_sub_model(varnames):
    """Test the behavior of TMSubModel.

    :param list(str) varnames: list of variable names.
    """
    xsh, xl_0, xl_1, y_0, y_1 = varnames
    index = 0
    default_inputs = {
        xsh: array([0.5]),
        xl_0: array([1.0, 2.0]),
        y_1: array([3.0, 4.0, 5.0]),
    }
    c_shared = array([[2.0], [3.0]])
    c_local = array([[2.0, 3.0], [2.0, 3.0]])
    c_coupling = {y_1: array([[3.0, 4.0, 5.0], [3.0, 4.0, 5.0]])}

    inputs_names = [xsh, xl_0, y_1]
    outputs_names = [y_0]

    model = TMSubModel(index, c_shared, c_local, c_coupling, default_inputs)
    assert model.name == "SubModel_0"
    assert set(inputs_names) == set(model.inputs_names)
    assert set(outputs_names) == set(model.outputs_names)
    for name in inputs_names:
        assert model.inputs_sizes[name] == len(default_inputs[name])
    c_y1 = c_coupling[y_1]
    assert model.outputs_sizes[y_0] == len(c_local) == len(c_shared) == len(c_y1)

    out = model()
    assert isinstance(out, dict)
    assert set(out.keys()) == {y_0}
    assert out[y_0] == pytest.approx(array([2.52631579, 2.425]), abs=1e-8)

    jac = model(jacobian=True)
    assert isinstance(jac, dict)
    assert y_0 in jac
    assert y_1 not in jac
    assert y_1 in jac[y_0]
    assert jac[y_0][y_1].shape == (2, 3)
    assert xsh in jac[y_0]
    assert jac[y_0][xsh].shape == (2, 1)
    assert xl_0 in jac[y_0]
    assert jac[y_0][xl_0].shape == (2, 2)
    assert xl_1 not in jac[y_0]

    # with noise
    noised_out = model(noise=array([0.1, -0.1]))
    assert (noised_out[y_0] == out[y_0] + array([0.1, -0.1])).all()
