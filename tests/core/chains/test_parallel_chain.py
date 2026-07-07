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

from itertools import permutations
from typing import TYPE_CHECKING

import pytest

from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core.chains.parallel_chain import ParallelDisciplineChain
from gemseo.core.discipline import Discipline
from gemseo.utils.derivatives.check.discipline import DisciplineJacobianChecker

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping


_APPROXIMATION_MODE = "complex_step"
_APPROXIMATION_STEP = 1e-30
_CHECK_JACOBIAN_KWARGS = {"atol": 1e-6, "rtol": 1e-6}


@pytest.mark.parametrize("use_deep_copy", [True, False])
@pytest.mark.parametrize("perm", list(permutations(range(4))))
def test_parallel_chain_combinatorial_thread(
    sobieski_disciplines, perm, use_deep_copy
) -> None:
    chain = ParallelDisciplineChain(
        [sobieski_disciplines[p] for p in perm],
        use_threading=True,
        use_deep_copy=use_deep_copy,
    )
    chain.linearize(compute_all_jacobians=True)
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        chain.io.input_grammar.defaults,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


@pytest.mark.skip_under_windows
@pytest.mark.parametrize("perm", list(permutations(range(4)))[:2])
def test_parallel_chain_combinatorial_mprocess(sobieski_disciplines, perm) -> None:
    chain = ParallelDisciplineChain(
        [sobieski_disciplines[p] for p in perm], use_threading=False
    )
    checker = DisciplineJacobianChecker(chain)
    assert checker.check(
        chain.io.input_grammar.defaults,
        approximation_mode=_APPROXIMATION_MODE,
        step=_APPROXIMATION_STEP,
        **_CHECK_JACOBIAN_KWARGS,
    )


def test_workflow_dataflow(sobieski_disciplines) -> None:
    chain = ParallelDisciplineChain(sobieski_disciplines)
    assert isinstance(
        chain.get_process_flow().get_execution_flow(), ParallelExecSequence
    )
    assert chain.get_process_flow().get_data_flow() == []


def test_non_ndarray_inputs():
    """Check that ParallelDisciplineChain handles inputs that are not NumPy arrays."""

    class StringDuplicator(Discipline):
        """A discipline duplicating an input string, e.g. "foo" -> "foofoo"."""

        def __init__(self):  # noqa: D107
            super().__init__()
            self.io.input_grammar.update_from_types({"in": str})
            self.io.output_grammar.update_from_types({"out": str})
            self.io.input_grammar.defaults["in"] = "foo"

        def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
            self.io.output_data["out"] = input_data["in"] * 2

    mdo_parallel_chain = ParallelDisciplineChain([StringDuplicator()])
    mdo_parallel_chain.execute()
    assert mdo_parallel_chain.output_data["out"] == "foofoo"
    mdo_parallel_chain.execute({"in": "bar"})
    assert mdo_parallel_chain.output_data["out"] == "barbar"
