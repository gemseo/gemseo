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
from gemseo.problems.scalable.parametric.core.design_space import TMDesignSpace
from gemseo.problems.scalable.parametric.core.variables import get_coupling_name
from gemseo.problems.scalable.parametric.core.variables import get_x_local_name
from gemseo.problems.scalable.parametric.core.variables import X_SHARED_NAME
from numpy import array
from numpy import ones
from numpy import zeros


@pytest.fixture
def varnames():
    """Variables names."""
    xsh = X_SHARED_NAME
    xl_0 = get_x_local_name(0)
    xl_1 = get_x_local_name(1)
    y_0 = get_coupling_name(0)
    y_1 = get_coupling_name(1)
    return xsh, xl_0, xl_1, y_0, y_1


def test_tm_design_space(varnames):
    """Test if names, sizes, bounds and default values are correct.

    :param list(str) varnames: list of variable names.
    """
    xsh, xl_0, xl_1, y_0, y_1 = varnames
    variables_sizes = {xsh: 2, xl_0: 2, xl_1: 3, y_0: 3, y_1: 2}
    n_shared = variables_sizes[xsh]
    n_local = [variables_sizes[xl_0], variables_sizes[xl_1]]
    n_coupling = [variables_sizes[y_0], variables_sizes[y_1]]
    default_inputs = {xsh: array([0.6, 0.7])}
    design_space = TMDesignSpace(n_shared, n_local, n_coupling, default_inputs)
    assert set(variables_sizes.keys()) == set(design_space.names)
    for name, size in design_space.sizes.items():
        assert variables_sizes[name] == size
        assert (design_space.lower_bounds[name] == zeros(size)).all()
        assert (design_space.upper_bounds[name] == ones(size)).all()
        if name in default_inputs:
            assert (design_space.default_values[name] == default_inputs[name]).all()
        else:
            assert (design_space.default_values[name] == zeros(size) + 0.5).all()
