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
"""Test for the module variable_names."""

from __future__ import annotations

import pytest

from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    constraint_variable_base_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    coupling_variable_base_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    get_constraint_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    get_coupling_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    get_u_local_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    get_x_local_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    local_design_variable_base_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    objective_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    shared_design_variable_name,
)
from gemseo.problem.mdo.scalable.parametric.standalone.variable_name import (
    uncertain_variable_base_name,
)


def test_get_u_local_name() -> None:
    """Check the function get_u_local_name."""
    assert get_u_local_name(1) == "u_1"


def test_get_x_local_name() -> None:
    """Check the function get_x_local_name."""
    assert get_x_local_name(1) == "x_1"


def test_get_coupling_name() -> None:
    """Check the function get_coupling_name."""
    assert get_coupling_name(1) == "y_1"


def test_get_constraint_name() -> None:
    """Check the function get_constraint_name."""
    assert get_constraint_name(1) == "c_1"


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        (shared_design_variable_name, "x_0"),
        (objective_name, "f"),
        (local_design_variable_base_name, "x"),
        (uncertain_variable_base_name, "u"),
        (constraint_variable_base_name, "c"),
        (coupling_variable_base_name, "y"),
    ],
)
def test_names(variable, value) -> None:
    """Check the names of the variables."""
    assert variable == value
