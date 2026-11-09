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
"""The design space of the welded beam."""

from __future__ import annotations

from numpy import arange

from gemseo.problem.optimization._welded_beam.data import catalog
from gemseo.problem.optimization._welded_beam.data import initial_dimensions
from gemseo.problem.optimization._welded_beam.data import initial_material
from gemseo.problem.optimization._welded_beam.data import step
from gemseo.space.design import DesignSpace


def create_design_space() -> DesignSpace:
    """Create the design space of the welded beam.

    Its variables are
    the catalog variable `material`,
    the discrete variables `weld_thickness` and `bar_thickness` in [0.0625, 2],
    the real variable `weld_length` in [0.1, 10]
    and the discrete variable `bar_height` in [2, 20],
    the discrete values being spaced by 0.0625.
    Their values are the initial point,
    see `initial_material` and `initial_dimensions` of the module
    [data][gemseo.problem.optimization._welded_beam.data].

    The catalog variable `material` needs a driver encoding the catalog variables,
    whereas the discipline takes the properties of the material instead;
    meanwhile,
    use [filter()][gemseo.space.design.DesignSpace.filter]
    to remove `material` from the design space.

    Returns:
        A new design space at each call.
    """
    thicknesses = step * arange(1, 33)
    heights = step * arange(32, 321)
    design_space = DesignSpace()
    design_space.add_catalog_variable("material", catalog, value=initial_material)
    design_space.add_discrete_variable(
        "weld_thickness", thicknesses, value=initial_dimensions["weld_thickness"]
    )
    design_space.add_real_variable(
        "weld_length",
        lower_bound=0.1,
        upper_bound=10.0,
        value=initial_dimensions["weld_length"],
    )
    design_space.add_discrete_variable(
        "bar_height", heights, value=initial_dimensions["bar_height"]
    )
    design_space.add_discrete_variable(
        "bar_thickness", thicknesses, value=initial_dimensions["bar_thickness"]
    )
    return design_space
