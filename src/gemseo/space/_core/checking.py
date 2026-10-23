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
"""Checks shared by the variables and spaces of variables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.space.design import DesignSpace

if TYPE_CHECKING:
    from gemseo.space.base import BaseVariableSpace


def check_design_space(
    space: BaseVariableSpace,
    requester_name: str,
    alternative_name: str,
) -> None:
    """Check that a space of variables is a design space.

    A design space is the only space defining bounds,
    hence a current value, a normalization policy and a membership test,
    which everything driven by an optimization algorithm relies on.

    Args:
        space: The space of variables.
        requester_name: The name of the class requiring a design space,
            preceded by its article, e.g. `"An MDOScenario"`.
        alternative_name: The name of the class to be used instead
            when the space is not a design space,
            preceded by its article, e.g. `"an EvaluationScenario"`.

    Raises:
        TypeError: When the space of variables is not a design space.
    """
    if not isinstance(space, DesignSpace):
        msg = (
            f"{requester_name} requires a design space; "
            f"got a {space.__class__.__name__}; "
            f"use {alternative_name} to sample a random space."
        )
        raise TypeError(msg)
