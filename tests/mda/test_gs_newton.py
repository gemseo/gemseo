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

from typing import TYPE_CHECKING

import pytest

from gemseo.mda.gs_newton import MDAGSNewton
from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.settings.mda import MDAGaussSeidel_Settings
from gemseo.settings.mda import MDAGSNewton_Settings
from gemseo.settings.mda import MDANewtonRaphson_Settings

if TYPE_CHECKING:
    from gemseo.core.discipline.discipline import Discipline


@pytest.fixture
def disciplines() -> list[Discipline]:
    """The Sellar 1 and 2 disciplines."""
    return [Sellar1(), Sellar2()]


def test_settings_as_model(disciplines) -> None:
    """Test that the Pydantic settings model are properly passed."""
    gsnewton_settings = MDAGSNewton_Settings(
        gauss_seidel_settings=MDAGaussSeidel_Settings(tolerance=1e-3, max_mda_iter=8),
        newton_settings=MDANewtonRaphson_Settings(tolerance=1e-4, max_mda_iter=16),
    )

    mda = MDAGSNewton(disciplines, settings_model=gsnewton_settings)

    assert mda.mda_sequence[0].settings.tolerance == 1e-3
    assert mda.mda_sequence[0].settings.max_mda_iter == 8

    assert mda.mda_sequence[1].settings.tolerance == 1e-4
    assert mda.mda_sequence[1].settings.max_mda_iter == 16


def test_settings_as_key_value_pairs(disciplines) -> None:
    """Test that the key/value settings pairs are properly passed."""
    gsnewton_settings = MDAGSNewton_Settings(
        gauss_seidel_settings={"tolerance": 1e-3, "max_mda_iter": 8},
        newton_settings={"tolerance": 1e-4, "max_mda_iter": 16},
    )

    mda = MDAGSNewton(disciplines, settings_model=gsnewton_settings)

    assert mda.mda_sequence[0].settings.tolerance == 1e-3
    assert mda.mda_sequence[0].settings.max_mda_iter == 8

    assert mda.mda_sequence[1].settings.tolerance == 1e-4
    assert mda.mda_sequence[1].settings.max_mda_iter == 16
