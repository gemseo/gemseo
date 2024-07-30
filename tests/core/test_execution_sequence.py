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

import unittest

from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import AtomicExecSequence
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.problems.mdo.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo_scenario import MDOScenario


class TestExecSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.d1 = SobieskiMission()
        self.d2 = SobieskiAerodynamics()
        self.d3 = SobieskiPropulsion()
        self.d4 = SobieskiStructure()

    def test_atomic_exec_sequence(self) -> None:
        self.assertRaises(Exception, AtomicExecSequence, "not_a_discipline")
        tmp = AtomicExecSequence(SobieskiAerodynamics())
        assert str(tmp) == "SobieskiAerodynamics(None)"
        assert repr(tmp) == ("SobieskiAerodynamics(None, " + str(tmp.uuid) + ")")

    def test_loop_exec_sequence(self) -> None:
        self.assertRaises(
            Exception,
            LoopExecSequence,
            SobieskiAerodynamics(),
            "not_a_compositeexecsequence",
        )

    def test_atom(self) -> None:
        _ = ExecutionSequenceFactory.atom(self.d1)

    def test_parent_assignment(self) -> None:
        atom = ExecutionSequenceFactory.atom(self.d1)
        seq = ExecutionSequenceFactory.serial().extend(atom)
        assert atom.parent == seq
        self.d1.parent = seq  # should not raise exception
        atom2 = ExecutionSequenceFactory.atom(self.d2)

        def test_parent() -> None:
            atom2.parent = seq

        self.assertRaises(RuntimeError, test_parent)

    def test_par(self) -> None:
        seq0 = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq1 = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        seq2 = ExecutionSequenceFactory.serial(seq0)
        seq2.extend(seq1)
        seq0.extend(seq1)

    def test_seq(self) -> None:
        seq0 = SerialExecSequence()
        str(seq0)
        repr(seq0)
        seq0.extend(self.d1)
        seq0.extend(self.d2)
        seq1 = SerialExecSequence()
        seq1.extend(self.d3)
        seq1.extend(self.d4)
        seq0.extend(seq1)

    def test_serial_par_extend(self) -> None:
        seq2 = ExecutionSequenceFactory.serial([self.d1, self.d2])
        seq4 = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        seq2.extend(seq4)

    def test_loop(self) -> None:
        seq1 = ExecutionSequenceFactory.serial([self.d1, self.d2])
        _ = ExecutionSequenceFactory.loop(self.d3, seq1)

    def test_serial_execution(self) -> None:
        seq = ExecutionSequenceFactory.serial([self.d1, self.d1])
        seq.enable()
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.DONE
        seq = SerialExecSequence()
        self.assertRaises(Exception, seq.enable)

    def test_serial_execution_failed(self) -> None:
        seq = ExecutionSequenceFactory.serial([self.d1, self.d2])
        seq.enable()
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.FAILED
        assert seq.status == MDODiscipline.ExecutionStatus.FAILED
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING  # check PENDING ignored
        assert seq.status == MDODiscipline.ExecutionStatus.FAILED

    def test_parallel_execution(self) -> None:
        seq = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq.enable()
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d2.status = MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d2.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d2.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.DONE
        for state in seq.get_statuses().values():
            assert state == MDODiscipline.ExecutionStatus.DONE

    def test_parallel_execution_failed(self) -> None:
        seq = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq.enable()
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d2.status = MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.FAILED
        assert seq.status == MDODiscipline.ExecutionStatus.FAILED

    def test_loop_execution(self) -> None:
        seq = ExecutionSequenceFactory.loop(
            self.d3, ExecutionSequenceFactory.serial([self.d1, self.d2])
        )
        seq.enable()
        self.d3.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.PENDING
        self.d3.status = MDODiscipline.ExecutionStatus.RUNNING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        self.d2.status = MDODiscipline.ExecutionStatus.PENDING
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d2.status = MDODiscipline.ExecutionStatus.RUNNING
        self.d2.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.iteration_count == 1
        assert seq.status == MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.PENDING
        self.d2.status = MDODiscipline.ExecutionStatus.PENDING
        self.d1.status = MDODiscipline.ExecutionStatus.RUNNING
        self.d1.status = MDODiscipline.ExecutionStatus.DONE
        self.d2.status = MDODiscipline.ExecutionStatus.RUNNING
        self.d2.status = MDODiscipline.ExecutionStatus.DONE
        self.d3.status = MDODiscipline.ExecutionStatus.DONE
        assert seq.status == MDODiscipline.ExecutionStatus.DONE
        assert seq.iteration_count == 2

    def test_loop_execution_failed(self) -> None:
        seq = ExecutionSequenceFactory.loop(
            self.d3, ExecutionSequenceFactory.serial([self.d1, self.d2])
        )
        seq.enable()
        self.d3.status = MDODiscipline.ExecutionStatus.FAILED
        assert seq.status == MDODiscipline.ExecutionStatus.FAILED

    def status_of(self, seq, disc, n=0):
        return seq.get_statuses()[seq.disc_to_uuids[disc][n]]

    def test_sub_scenario(self) -> None:
        d1 = SobieskiPropulsion()
        design_space = SobieskiDesignSpace()
        sc_prop = MDOScenario(
            [d1],
            "DisciplinaryOpt",
            "y_34",
            design_space.filter("x_3", copy=True),
            name="PropulsionScenario",
        )
        d2 = MDOScenarioAdapter(sc_prop, [], [])
        seq = ExecutionSequenceFactory.serial([d1])
        seq.extend(d2.get_expected_workflow())
        seq.enable()
        d1.status = MDODiscipline.ExecutionStatus.PENDING
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.PENDING
        assert self.status_of(seq, sc_prop) is None
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.RUNNING
        assert self.status_of(seq, sc_prop) is None
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        d1.status = MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, sc_prop) == MDODiscipline.ExecutionStatus.PENDING
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        sc_prop.status = MDODiscipline.ExecutionStatus.RUNNING
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, sc_prop) == MDODiscipline.ExecutionStatus.RUNNING
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        d1.status = MDODiscipline.ExecutionStatus.RUNNING
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, sc_prop) == MDODiscipline.ExecutionStatus.RUNNING
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        d1.status = MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, sc_prop) == MDODiscipline.ExecutionStatus.RUNNING
        # when done iteration_sequence is enabled again thus atom is in pending
        # state and not done
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))
        sc_prop.status = MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, d1) == MDODiscipline.ExecutionStatus.DONE
        assert self.status_of(seq, sc_prop) == MDODiscipline.ExecutionStatus.DONE
        self.assertRaises(IndexError, lambda: self.status_of(seq, d1, 1))

    def test_visitor_pattern(self) -> None:
        class Visitor:
            def __init__(self) -> None:
                self.result = []

            def visit_atomic(self, atom) -> None:
                self.result.append(atom.discipline)

            def visit_serial(self, serial) -> None:
                self.result.append(serial)

            def visit_parallel(self, parallel) -> None:
                self.result.append(parallel)

            def visit_loop(self, loop) -> None:
                self.result.append(loop)

        serial = ExecutionSequenceFactory.serial().extend([self.d1, self.d2])
        parallel = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        serial.extend(parallel)
        loop = ExecutionSequenceFactory.loop(self.d1, serial)

        visitor = Visitor()
        loop.accept(visitor)
        # prefix order expected
        expected = [loop, self.d1, serial, self.d1, self.d2, parallel, self.d3, self.d4]
        assert expected == visitor.result
