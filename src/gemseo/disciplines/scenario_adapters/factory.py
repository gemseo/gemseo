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
"""A factory of scenario adapters."""
from __future__ import annotations

from gemseo.core.factory import Factory
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter


class ScenarioAdapterFactory(Factory):
    """A factory of scenario adapters."""

    def __init__(self):  # noqa: D107
        super().__init__(MDOScenarioAdapter, ("gemseo.disciplines.scenario_adapters",))
