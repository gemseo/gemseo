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

from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.formulations.idf_chain import IDFChain
from gemseo.utils.discipline import DummyDiscipline as D


def test_idf_chain_is_mdo_chain():
    """Check that an IDFChain is an MDOChain."""
    assert issubclass(IDFChain, MDOChain)


@pytest.mark.parametrize(
    ("execution_sequence", "n_processes", "classes"),
    [
        ([[(D(),)]], 1, (D,)),
        ([[(D(),)]], 2, (D,)),
        ([[(D(), D())]], 1, (MDOParallelChain,)),
        ([[(D(), D())]], 2, (MDOParallelChain,)),
        ([[(D(), D()), (D(),)]], 1, (MDOChain,)),
        ([[(D(), D()), (D(),)]], 2, (MDOParallelChain,)),
        ([[(D(),)], [(D(),)]], 1, (D, D)),
        ([[(D(),)], [(D(),)]], 2, (D, D)),
        ([[(D(), D()), (D(),)], [(D(),)]], 1, (MDOChain,)),
        ([[(D(), D()), (D(),)], [(D(),)]], 2, (MDOParallelChain,)),
    ],
)
@pytest.mark.parametrize("use_threading", [False, True])
def test_idf_chain(execution_sequence, n_processes, use_threading, classes):
    """Check the types of the disciplines chained by IDFChain."""
    idf_chain = IDFChain(execution_sequence, n_processes, use_threading)
    for discipline, cls in zip(idf_chain.disciplines, classes, strict=False):
        assert isinstance(discipline, cls)
