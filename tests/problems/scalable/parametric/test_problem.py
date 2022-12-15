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
from gemseo.problems.scalable.parametric.core.problem import TMProblem
from gemseo.problems.scalable.parametric.core.variables import get_coupling_name
from gemseo.problems.scalable.parametric.core.variables import get_x_local_name
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


def test_tm_problem(varnames):
    """Test the behavior of TMProblem."""
    y_0 = get_coupling_name(0)
    y_1 = get_coupling_name(1)
    pbm = TMProblem()
    c_shared = [array([[0.417022]]), array([[0.30233257]])]
    c_local = [array([[0.72032449]]), array([[0.14675589]])]
    c_coupling = [{y_1: array([[0.00011437]])}, {y_0: array([[0.09233859]])}]
    c_constraint = [array([0.18626021]), array([0.34556073])]
    assert pbm.n_submodels == 2
    assert len(pbm.models) == pbm.n_submodels + 1
    for i in range(2):
        submodel = pbm.models[1 + i]
        assert c_shared[i] == pytest.approx(submodel.c_shared, abs=1e-8)
        assert c_local[i] == pytest.approx(submodel.c_local, abs=1e-8)
        value = list(submodel.c_coupling.values())[0]
        name = list(submodel.c_coupling.keys())[0]
        assert c_coupling[i][name] == pytest.approx(value, abs=1e-8)
    main_model = pbm.models[0]
    assert c_constraint[0] == pytest.approx(main_model.coefficients[0], abs=1e-8)
    assert c_constraint[1] == pytest.approx(main_model.coefficients[1], abs=1e-8)
    assert set(pbm.design_space.names) == set(varnames)

    pbm.design_space.names.append("x")
    pbm.reset_design_space()
    assert "x" not in pbm.design_space.names

    assert "x_shared (1)" in pbm.__str__()

    pbm = TMProblem(n_local=[1, 1, 1], n_coupling=[1, 1, 1])
    assert "y_2" in pbm.models[1].inputs_names
    assert y_1 in pbm.models[1].inputs_names
    assert y_0 in pbm.models[2].inputs_names
    assert "y_2" in pbm.models[2].inputs_names
    assert y_0 in pbm.models[3].inputs_names
    assert y_1 in pbm.models[3].inputs_names
    pbm = TMProblem(n_local=[1, 1, 1], n_coupling=[1, 1, 1], full_coupling=False)
    assert "y_2" in pbm.models[1].inputs_names
    assert y_0 in pbm.models[2].inputs_names
    assert y_1 in pbm.models[3].inputs_names
