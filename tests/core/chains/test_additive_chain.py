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
from __future__ import annotations

from numpy import allclose

from gemseo.core.chains.additive_chain import AdditiveDisciplineChain
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.utils.derivatives.check.discipline import DisciplineJacobianChecker


def test_double_mission_chain() -> None:
    """Test that AdditiveDisciplineChain sums outputs of repeated disciplines."""
    disciplines = [SobieskiMission(), SobieskiMission()]
    chain = AdditiveDisciplineChain(disciplines, outputs_to_sum=["y_4"])

    chain.execute()
    mission = SobieskiMission()
    mission.execute()
    assert allclose(chain.io.output_data["y_4"], mission.io.output_data["y_4"] * 2.0)

    checker = DisciplineJacobianChecker(chain)
    assert checker.check(atol=1e-5, rtol=1e-5)
