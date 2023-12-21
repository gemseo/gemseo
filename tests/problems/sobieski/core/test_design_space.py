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
"""Tests for the class SobieskiDesignSpace."""

from __future__ import annotations

import pytest

from gemseo.problems.sobieski.core.design_space import SobieskiDesignSpace


@pytest.mark.parametrize(
    ("use_original_names", "design_variable_name", "coupling_variable_name"),
    [(True, "x_shared", "y_14"), (False, "t_c", "cl")],
)
@pytest.mark.parametrize("copy", [False, True])
@pytest.mark.parametrize("filter_coupling_variables", [False, True])
def test_filter_variables(
    copy,
    use_original_names,
    design_variable_name,
    coupling_variable_name,
    filter_coupling_variables,
):
    """Check filter_design_variables and filter_coupling_variables."""
    original_design_space = SobieskiDesignSpace(use_original_names)
    v_type = "coupling" if filter_coupling_variables else "design"
    design_space = getattr(original_design_space, f"filter_{v_type}_variables")(copy)
    assert (id(design_space) == id(original_design_space)) is not copy
    assert (coupling_variable_name in design_space) is filter_coupling_variables
    assert (design_variable_name not in design_space) is filter_coupling_variables
