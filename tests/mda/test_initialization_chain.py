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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.initialization_chain import MDOInitializationChain
from gemseo.mda.initialization_chain import order_disciplines_from_default_inputs
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)

DISC_DESCR_1 = [
    ("B", ["d", "c"], ["e"]),
    ("C", ["e", "g"], ["f"]),
    ("A", ["b", "f"], ["c"]),
]


@pytest.fixture()
def disciplines1() -> list[MDODiscipline]:
    """Return the disciplines for test case 1.

    Returns:
         The disciplines.
    """
    disciplines = create_disciplines_from_desc(DISC_DESCR_1)
    # Delete c default input of B
    disciplines[0].default_inputs.pop("c")
    # Delete e default input of E
    disciplines[1].default_inputs.pop("e")
    return disciplines


def test_get_initialization_disciplines(disciplines1):
    """Tests the execution sequence correctness."""
    assert order_disciplines_from_default_inputs(disciplines1) == [
        disciplines1[2],
        disciplines1[0],
        disciplines1[1],
    ]


def test_fail_get_initialization_disciplines(disciplines1):
    """Tests that the algorithm fails when not enough default inputs are present."""
    disciplines1[1].default_inputs.pop("g")
    missing_inputs = order_disciplines_from_default_inputs(disciplines1, False)
    assert missing_inputs == ["g"]
    with pytest.raises(
        ValueError,
        match=r"Cannot compute the inputs g, for the following disciplines C.",
    ):
        order_disciplines_from_default_inputs(disciplines1, True)

    missing_inputs = order_disciplines_from_default_inputs(disciplines1, False)
    assert missing_inputs == ["g"]


def test_create_init_chain(disciplines1):
    """Tests the creation of the process."""
    chain = MDOInitializationChain(disciplines1)
    chain.execute()
