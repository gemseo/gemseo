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
"""Tests for the class ScenarioResultFactory."""

from __future__ import annotations

from gemseo.scenarios.scenario_results.scenario_result_factory import (
    ScenarioResultFactory,
)


def test_factory():
    """Check that the base class and another one are available."""
    assert ScenarioResultFactory().is_available("ScenarioResult")
    assert ScenarioResultFactory().is_available("BiLevelScenarioResult")
