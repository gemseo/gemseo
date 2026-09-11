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

import operator
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple

import matplotlib
import pytest
from numpy import array

from gemseo import set_data_converters
from gemseo.problem.mdo.sellar import with_2d_array
from gemseo.problem.mdo.sellar.sellar_1 import Sellar1
from gemseo.problem.mdo.sellar.sellar_2 import Sellar2
from gemseo.problem.mdo.sellar.sellar_system import SellarSystem
from gemseo.problem.mdo.sellar.variable import x_shared
from gemseo.util.discipline import DummyDiscipline
from gemseo.util.testing.pytest_conftest import *  # noqa: F401,F403

if TYPE_CHECKING:
    from collections.abc import Generator

    from gemseo.core.discipline.discipline import Discipline

# Use a non GUI rendering backend for matplotlib.
matplotlib.use("agg")

doc_example_mark = "doc_examples"


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip by default some marked tests."""
    if not config.getoption("-m"):
        skip_me = pytest.mark.skip(
            reason=f"use '-m {doc_example_mark}' to run this test"
        )
        for item in items:
            if doc_example_mark in item.keywords:
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
    return SellarDisciplines(Sellar1(), Sellar2(), SellarSystem())


@pytest.fixture
def two_virtual_disciplines() -> list[Discipline]:
    """Create two dummy disciplines that can only be executed in virtual mode."""

    class VirtualDummy(DummyDiscipline):
        virtual_execution = True

    disc_1 = VirtualDummy("d1")
    disc_1.io.input_grammar.update_from_names(["x"])
    disc_1.io.output_grammar.update_from_names(["y"])
    disc_1.io.input_grammar.defaults = {"x": array([1.0])}
    disc_1.default_output_data = {"y": array([2.0])}

    disc_2 = VirtualDummy("d2")
    disc_2.io.input_grammar.update_from_names(["y"])
    disc_2.io.output_grammar.update_from_names(["z"])
    disc_2.io.input_grammar.defaults = {"y": array([3.0])}
    disc_2.default_output_data = {"z": array([4.0])}

    return [disc_1, disc_2]


@pytest.fixture
def sellar_with_2d_array() -> Generator[None, Any, None]:
    """Change the data-converter temporary for sellar with 2d x_shared array."""
    if with_2d_array:  # pragma: no cover
        set_data_converters(
            {x_shared: operator.itemgetter(0)},
            {x_shared: lambda a: array([a])},
            {},
        )
        yield
        set_data_converters({}, {}, {})
    else:
        yield
