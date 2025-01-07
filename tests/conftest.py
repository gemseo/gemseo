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
"""Test helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import NamedTuple

import pytest

from gemseo.problems.mdo.sellar.sellar_1 import Sellar1
from gemseo.problems.mdo.sellar.sellar_2 import Sellar2
from gemseo.problems.mdo.sellar.sellar_system import SellarSystem
from gemseo.problems.mdo.sellar.utils import set_data_converter
from gemseo.utils.testing.pytest_conftest import *  # noqa: F401,F403

if TYPE_CHECKING:
    from gemseo import Discipline

DOC_EXAMPLE_MARK = "doc_examples"


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip by default some marked tests."""
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(
            reason=f"use '-m {DOC_EXAMPLE_MARK}' to run this test"
        )
        for item in items:
            if DOC_EXAMPLE_MARK in item.keywords:
                item.add_marker(skip_me)


class SellarDisciplines(NamedTuple):
    sellar1: Discipline
    sellar2: Discipline
    sellar_system: Discipline


@pytest.fixture
def sellar_disciplines() -> SellarDisciplines:
    """The disciplines of the Sellar problem.

    Returns:
        * A Sellar1 discipline.
        * A Sellar2 discipline.
        * A SellarSystem discipline.
    """
    # This handles running the test suite for checking data conversion.
    with set_data_converter():
        yield SellarDisciplines(Sellar1(), Sellar2(), SellarSystem())
