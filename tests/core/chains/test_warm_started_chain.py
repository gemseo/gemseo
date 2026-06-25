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

import pytest

from gemseo.core.chains.warm_started_chain import WarmStartedDisciplineChain
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.utils.testing.helpers import assert_exception


@pytest.mark.parametrize(
    ("variable_names", "expected"), [(["y_21"], True), ([], False)]
)
def test_warm_started_discipline_chain(variable_names, expected) -> None:
    """Test that the variables are warm-started properly."""
    disciplines = [SobieskiStructure(), SobieskiAerodynamics()]
    chain = WarmStartedDisciplineChain(
        disciplines=disciplines, variable_names_to_warm_start=variable_names
    )
    out = chain.execute()
    y_12 = out["y_12"]
    chain.cache.clear()
    out = chain.execute()
    assert (y_12 != out["y_12"]).any() == expected


def test_warm_started_discipline_chain_jac(snapshot) -> None:
    """Test that the Jacobian of a WarmStartedDisciplineChain raises an exception."""
    chain = WarmStartedDisciplineChain(
        [SobieskiMission()], variable_names_to_warm_start=[]
    )
    with assert_exception(NotImplementedError, snapshot):
        chain.check_jacobian()


@pytest.mark.parametrize("variable_names", [("y_4", "i_dont_exist"), ("i_dont_exist",)])
def test_warm_started_mdo_chain_variables(variable_names, snapshot) -> None:
    """Test an exception if a variable that is not in the chain is warm started."""
    with assert_exception(ValueError, snapshot):
        WarmStartedDisciplineChain(
            [SobieskiMission()], variable_names_to_warm_start=variable_names
        )
