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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard, Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import NamedTuple

import pytest

from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.execution_status import ExecutionStatus
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario


class Disciplines(NamedTuple):
    d1: SobieskiMission
    d2: SobieskiAerodynamics
    d3: SobieskiPropulsion
    d4: SobieskiStructure


@pytest.fixture
def disciplines() -> Disciplines:
    return Disciplines(
        SobieskiMission(),
        SobieskiAerodynamics(),
        SobieskiPropulsion(),
        SobieskiStructure(),
    )


def test_atomic_exec_sequence() -> None:
    tmp = ExecutionSequence(SobieskiAerodynamics())
    assert tmp.__str__() == "SobieskiAerodynamics(DONE)"
    assert tmp.__repr__() == ("SobieskiAerodynamics(DONE, " + str(tmp.uuid) + ")")


def test_atom(disciplines) -> None:
    _ = ExecutionSequence(disciplines.d1)


def test_parent_assignment(disciplines) -> None:
    atom = ExecutionSequence(disciplines.d1)
    seq = SequentialExecSequence()
    seq.extend(atom)
    assert atom.parent == seq
    disciplines.d1.parent = seq  # should not raise exception
    atom2 = ExecutionSequence(disciplines.d2)

    with pytest.raises(RuntimeError):
        atom2.parent = seq


def test_par(disciplines) -> None:
    seq0 = ParallelExecSequence([disciplines.d1, disciplines.d2])
    seq1 = ParallelExecSequence([disciplines.d3, disciplines.d4])
    seq2 = SequentialExecSequence(seq0)
    seq2.extend(seq1)
    seq0.extend(seq1)


def test_seq(disciplines) -> None:
    seq0 = SequentialExecSequence()
    str(seq0)
    repr(seq0)
    seq0.extend(disciplines.d1)
    seq0.extend(disciplines.d2)
    seq1 = SequentialExecSequence()
    seq1.extend(disciplines.d3)
    seq1.extend(disciplines.d4)
    seq0.extend(seq1)


def test_serial_par_extend(disciplines) -> None:
    seq2 = SequentialExecSequence([disciplines.d1, disciplines.d2])
    seq4 = ParallelExecSequence([disciplines.d3, disciplines.d4])
    seq2.extend(seq4)


def test_loop(disciplines) -> None:
    seq1 = SequentialExecSequence([disciplines.d1, disciplines.d2])
    _ = LoopExecSequence(disciplines.d3, seq1)


def test_serial_execution(disciplines) -> None:
    seq = SequentialExecSequence([disciplines.d1, disciplines.d1])
    seq.enable()
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    with pytest.raises(ValueError):
        SequentialExecSequence().enable()


def test_serial_execution_failed(disciplines) -> None:
    seq = SequentialExecSequence([disciplines.d1, disciplines.d2])
    seq.enable()
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.FAILED
    assert seq.status == ExecutionStatus.Status.FAILED
    disciplines.d1.execution_status.value = (
        ExecutionStatus.Status.DONE
    )  # check DONE ignored
    assert seq.status == ExecutionStatus.Status.FAILED


def test_parallel_execution(disciplines) -> None:
    seq = ParallelExecSequence([disciplines.d1, disciplines.d2])
    seq.enable()
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d2.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    for state in seq.get_statuses().values():
        assert state == ExecutionStatus.Status.DONE


def test_parallel_execution_failed(disciplines) -> None:
    seq = ParallelExecSequence([disciplines.d1, disciplines.d2])
    seq.enable()
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.FAILED
    assert seq.status == ExecutionStatus.Status.FAILED


def test_loop_execution(disciplines) -> None:
    seq = LoopExecSequence(
        disciplines.d3,
        SequentialExecSequence([disciplines.d1, disciplines.d2]),
    )
    seq.enable()
    disciplines.d3.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    disciplines.d3.execution_status.value = ExecutionStatus.Status.RUNNING
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d2.execution_status.value = ExecutionStatus.Status.RUNNING
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.iteration_count == 1
    assert seq.status == ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d1.execution_status.value = ExecutionStatus.Status.RUNNING
    disciplines.d1.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d2.execution_status.value = ExecutionStatus.Status.RUNNING
    disciplines.d2.execution_status.value = ExecutionStatus.Status.DONE
    disciplines.d3.execution_status.value = ExecutionStatus.Status.DONE
    assert seq.status == ExecutionStatus.Status.DONE
    assert seq.iteration_count == 2


def test_loop_execution_failed(disciplines) -> None:
    seq = LoopExecSequence(
        disciplines.d3,
        SequentialExecSequence([disciplines.d1, disciplines.d2]),
    )
    seq.enable()
    disciplines.d3.execution_status.value = ExecutionStatus.Status.FAILED
    assert seq.status == ExecutionStatus.Status.FAILED


def status_of(seq, disc, n=0):
    return seq.get_statuses()[seq.disc_to_uuids[disc][n]]


def test_sub_scenario() -> None:
    d1 = SobieskiPropulsion()
    design_space = SobieskiDesignSpace()
    sc_prop = MDOScenario(
        [d1],
        "y_34",
        design_space.filter("x_3", copy=True),
        formulation_name="DisciplinaryOpt",
        name="PropulsionScenario",
    )
    d2 = MDOScenarioAdapter(sc_prop, [], [])
    seq = SequentialExecSequence([d1])
    seq.extend(d2.get_process_flow().get_execution_flow())
    seq.enable()
    d1.execution_status.value = ExecutionStatus.Status.DONE
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) is ExecutionStatus.Status.DONE
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert status_of(seq, d1) == ExecutionStatus.Status.RUNNING
    assert status_of(seq, sc_prop) is ExecutionStatus.Status.DONE
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    d1.execution_status.value = ExecutionStatus.Status.DONE
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) == ExecutionStatus.Status.DONE
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    sc_prop.execution_status.value = ExecutionStatus.Status.RUNNING
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) == ExecutionStatus.Status.RUNNING
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    d1.execution_status.value = ExecutionStatus.Status.RUNNING
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) == ExecutionStatus.Status.RUNNING
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    d1.execution_status.value = ExecutionStatus.Status.DONE
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) == ExecutionStatus.Status.RUNNING
    # when done iteration_sequence is enabled again thus atom is in pending
    # state and not done
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)
    sc_prop.execution_status.value = ExecutionStatus.Status.DONE
    assert status_of(seq, d1) == ExecutionStatus.Status.DONE
    assert status_of(seq, sc_prop) == ExecutionStatus.Status.DONE
    with pytest.raises(IndexError):
        status_of(seq, d1, 1)


def test_visitor_pattern(disciplines) -> None:
    class Visitor:
        def __init__(self) -> None:
            self.result = []

        def visit_atomic(self, atom) -> None:
            self.result.append(atom.process)

        def visit_serial(self, serial) -> None:
            self.result.append(serial)

        def visit_parallel(self, parallel) -> None:
            self.result.append(parallel)

        def visit_loop(self, loop) -> None:
            self.result.append(loop)

    serial = SequentialExecSequence()
    serial.extend([disciplines.d1, disciplines.d2])
    parallel = ParallelExecSequence([disciplines.d3, disciplines.d4])
    serial.extend(parallel)
    loop = LoopExecSequence(disciplines.d1, serial)

    visitor = Visitor()
    loop.accept(visitor)
    # prefix order expected
    expected = [
        loop,
        disciplines.d1,
        serial,
        disciplines.d1,
        disciplines.d2,
        parallel,
        disciplines.d3,
        disciplines.d4,
    ]
    assert expected == visitor.result
