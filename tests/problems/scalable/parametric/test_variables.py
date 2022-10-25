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
from gemseo.problems.scalable.parametric.core.variables import check_consistency
from gemseo.problems.scalable.parametric.core.variables import get_constraint_name
from gemseo.problems.scalable.parametric.core.variables import get_coupling_name
from gemseo.problems.scalable.parametric.core.variables import get_u_local_name
from gemseo.problems.scalable.parametric.core.variables import get_x_local_name
from gemseo.problems.scalable.parametric.core.variables import U_LOCAL_NAME_BASIS
from gemseo.problems.scalable.parametric.core.variables import X_LOCAL_NAME_BASIS
from gemseo.problems.scalable.parametric.core.variables import X_SHARED_NAME


def test_check_consistency():
    """Test if the problem configuration is consistent:

    - numbers of variables are strictly positive integers.
    - length(n_local) == length(n_coupling)
    """
    n_shared = 1
    n_local = [1, 2]
    n_coupling = [3, 4]
    check_consistency(n_shared, n_local, n_coupling)
    with pytest.raises(TypeError):
        check_consistency(0, n_local, n_coupling)
    with pytest.raises(TypeError):
        check_consistency(0.5, n_local, n_coupling)
    with pytest.raises(TypeError):
        check_consistency(n_shared, 1, n_coupling)
    with pytest.raises(TypeError):
        check_consistency(n_shared, [0.5, 2], n_coupling)
    with pytest.raises(TypeError):
        check_consistency(n_shared, [0, 2], n_coupling)
    with pytest.raises(TypeError):
        check_consistency(n_shared, n_local, 1)
    with pytest.raises(TypeError):
        check_consistency(n_shared, n_local, [0, 2])
    with pytest.raises(TypeError):
        check_consistency(n_shared, n_local, [0, 2.5])
    with pytest.raises(ValueError):
        check_consistency(n_shared, n_local, [1, 2, 3])


def test_names():
    """Check variables names."""
    assert X_SHARED_NAME == "x_shared"
    assert X_LOCAL_NAME_BASIS == "x_local"
    assert U_LOCAL_NAME_BASIS == "u_local"
    assert get_x_local_name(0) == "x_local_0"
    assert get_u_local_name(0) == "u_local_0"
    assert get_coupling_name(0) == "y_0"
    assert get_constraint_name(0) == "cstr_0"
