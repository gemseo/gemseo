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
#        :author: Gilberto RUIZ
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from datetime import datetime
from re import match

import pytest

from gemseo.core.parallel_execution.callable_parallel_execution import (
    CallableParallelExecution,
)
from gemseo.utils.base_name_generator import BaseNameGenerator
from gemseo.utils.name_generator import NameGenerator


@pytest.mark.parametrize(
    "naming",
    [
        BaseNameGenerator.Naming.UUID,
        BaseNameGenerator.Naming.NUMBERED,
        BaseNameGenerator.Naming.DATED_UUID,
        "WRONGMETHOD",
    ],
)
def test_generate_name(naming) -> None:
    """Test the name generation."""
    name_generator = NameGenerator(naming_method=naming)
    name_1 = name_generator.generate_name()
    name_2 = name_generator.generate_name()
    if naming == BaseNameGenerator.Naming.UUID:
        assert match(r"[0-9a-fA-F]{12}$", name_1) is not None
        assert name_1 != name_2
    elif naming == BaseNameGenerator.Naming.NUMBERED:
        assert name_1 == "1"
        assert name_2 == "2"
    elif naming == BaseNameGenerator.Naming.DATED_UUID:
        assert name_1 != name_2
        elements_1, elements_2 = name_1.split("_"), name_2.split("_")
        assert elements_1[0] == elements_2[0]
        assert datetime.strptime(
            "_".join(elements_1[0:2]), "%Y-%m-%d_%Hh%Mmin%Ss"
        ) <= datetime.strptime("_".join(elements_2[0:2]), "%Y-%m-%d_%Hh%Mmin%Ss")
        assert match(r"[0-9a-fA-F]{12}$", elements_1[2]) is not None
    else:
        assert name_1 is None
        assert name_2 is None


def f(_):
    """Helper function to use ine CallableParallelExecution."""
    return NameGenerator(naming_method=NameGenerator.Naming.UUID).generate_name()


def test_unique_directory_name_in_multiprocessing():
    """Test of _generate_unique_directory_name in multiprocessing.

    Reproducer of bug #7 when using UUID1, leading to frequent UUID collisions.
    """
    parallel_execution = CallableParallelExecution([f], n_processes=5)
    out = parallel_execution.execute([None] * 100)
    assert len(set(out)) == 100
