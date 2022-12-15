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
from copy import deepcopy

from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import AtomicExecSequence
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import LoopExecSequence
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.disciplines.scenario_adapter import MDOScenarioAdapter
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiAerodynamics
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.problems.sobieski.disciplines import SobieskiPropulsion
from gemseo.problems.sobieski.disciplines import SobieskiStructure


class TestExecSequence(unittest.TestCase):
    def setUp(self):
        self.d1 = SobieskiMission()
        self.d2 = SobieskiAerodynamics()
        self.d3 = SobieskiPropulsion()
        self.d4 = SobieskiStructure()

    def test_atomic_exec_sequence(self):
        self.assertRaises(Exception, AtomicExecSequence, "not_a_discipline")
        tmp = AtomicExecSequence(SobieskiAerodynamics())
        assert tmp.__str__() == "SobieskiAerodynamics(None)"
        assert tmp.__repr__() == ("SobieskiAerodynamics(None, " + str(tmp.uuid) + ")")

    def test_loop_exec_sequence(self):
        self.assertRaises(
            Exception,
            LoopExecSequence,
            SobieskiAerodynamics(),
            "not_a_compositeexecsequence",
        )

    def test_atom(self):
        _ = ExecutionSequenceFactory.atom(self.d1)

    def test_parent_assignment(self):
        atom = ExecutionSequenceFactory.atom(self.d1)
        seq = ExecutionSequenceFactory.serial().extend(atom)
        self.assertEqual(atom.parent, seq)
        self.d1.parent = seq  # should not raise exception
        atom2 = ExecutionSequenceFactory.atom(self.d2)

        def test_parent():
            atom2.parent = seq

        self.assertRaises(RuntimeError, test_parent)

    def test_par(self):
        seq0 = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq1 = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        seq2 = ExecutionSequenceFactory.serial(seq0)
        seq2.extend(seq1)
        seq0.extend(seq1)

    def test_seq(self):
        seq0 = SerialExecSequence()
        str(seq0)
        repr(seq0)
        seq0.extend(self.d1)
        seq0.extend(self.d2)
        seq1 = SerialExecSequence()
        seq1.extend(self.d3)
        seq1.extend(self.d4)
        seq0.extend(seq1)

    def test_serial_par_extend(self):
        seq2 = ExecutionSequenceFactory.serial([self.d1, self.d2])
        seq4 = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        seq2.extend(seq4)

    def test_loop(self):
        seq1 = ExecutionSequenceFactory.serial([self.d1, self.d2])
        _ = ExecutionSequenceFactory.loop(self.d3, seq1)

    def test_serial_execution(self):
        seq = ExecutionSequenceFactory.serial([self.d1, self.d1])
        seq.enable()
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_DONE)
        seq = SerialExecSequence()
        self.assertRaises(Exception, seq.enable)

    def test_serial_execution_failed(self):
        seq = ExecutionSequenceFactory.serial([self.d1, self.d2])
        seq.enable()
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_FAILED
        self.assertEqual(seq.status, MDODiscipline.STATUS_FAILED)
        self.d1.status = MDODiscipline.STATUS_PENDING  # check PENDING ignored
        self.assertEqual(seq.status, MDODiscipline.STATUS_FAILED)

    def test_parallel_execution(self):
        seq = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq.enable()
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d2.status = MDODiscipline.STATUS_PENDING
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d2.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d2.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_DONE)
        for state in seq.get_statuses().values():
            self.assertEqual(MDODiscipline.STATUS_DONE, state)

    def test_parallel_execution_failed(self):
        seq = ExecutionSequenceFactory.parallel([self.d1, self.d2])
        seq.enable()
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d2.status = MDODiscipline.STATUS_PENDING
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d1.status = MDODiscipline.STATUS_FAILED
        self.assertEqual(seq.status, MDODiscipline.STATUS_FAILED)

    def test_loop_execution(self):
        seq = ExecutionSequenceFactory.loop(
            self.d3, ExecutionSequenceFactory.serial([self.d1, self.d2])
        )
        seq.enable()
        self.d3.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_PENDING)
        self.d3.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.d2.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d2.status = MDODiscipline.STATUS_RUNNING
        self.d2.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.iteration_count, 1)
        self.assertEqual(seq.status, MDODiscipline.STATUS_RUNNING)
        self.d1.status = MDODiscipline.STATUS_PENDING
        self.d2.status = MDODiscipline.STATUS_PENDING
        self.d1.status = MDODiscipline.STATUS_RUNNING
        self.d1.status = MDODiscipline.STATUS_DONE
        self.d2.status = MDODiscipline.STATUS_RUNNING
        self.d2.status = MDODiscipline.STATUS_DONE
        self.d3.status = MDODiscipline.STATUS_DONE
        self.assertEqual(seq.status, MDODiscipline.STATUS_DONE)
        self.assertEqual(seq.iteration_count, 2)

    def test_loop_execution_failed(self):
        seq = ExecutionSequenceFactory.loop(
            self.d3, ExecutionSequenceFactory.serial([self.d1, self.d2])
        )
        seq.enable()
        self.d3.status = MDODiscipline.STATUS_FAILED
        self.assertEqual(seq.status, MDODiscipline.STATUS_FAILED)

    def status_of(self, seq, disc, n=0):
        #         print seq
        #         print seq.disc_to_uuids[disc]
        #         print seq.get_statuses()
        return seq.get_statuses()[seq.disc_to_uuids[disc][n]]

    def test_sub_scenario(self):
        d1 = SobieskiPropulsion()
        design_space = SobieskiProblem().design_space
        sc_prop = MDOScenario(
            disciplines=[d1],
            formulation="DisciplinaryOpt",
            objective_name="y_34",
            design_space=deepcopy(design_space).filter("x_3"),
            name="PropulsionScenario",
        )
        d2 = MDOScenarioAdapter(sc_prop, [], [])
        seq = ExecutionSequenceFactory.serial([d1])
        seq.extend(d2.get_expected_workflow())
        seq.enable()
        d1.status = MDODiscipline.STATUS_PENDING
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_PENDING)
        self.assertEqual(self.status_of(seq, sc_prop), None)
        self.assertEqual(self.status_of(seq, d1, 1), None)
        d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_RUNNING)
        self.assertEqual(self.status_of(seq, sc_prop), None)
        self.assertEqual(self.status_of(seq, d1, 1), None)
        d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, sc_prop), MDODiscipline.STATUS_PENDING)
        self.assertEqual(self.status_of(seq, d1, 1), None)
        sc_prop.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, sc_prop), MDODiscipline.STATUS_RUNNING)
        self.assertEqual(self.status_of(seq, d1, 1), MDODiscipline.STATUS_PENDING)
        d1.status = MDODiscipline.STATUS_RUNNING
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, sc_prop), MDODiscipline.STATUS_RUNNING)
        self.assertEqual(self.status_of(seq, d1, 1), MDODiscipline.STATUS_RUNNING)
        d1.status = MDODiscipline.STATUS_DONE
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, sc_prop), MDODiscipline.STATUS_RUNNING)
        # when done iteration_sequence is enabled again thus atom is in pending
        # state and not done
        self.assertEqual(self.status_of(seq, d1, 1), MDODiscipline.STATUS_PENDING)
        sc_prop.status = MDODiscipline.STATUS_DONE
        self.assertEqual(self.status_of(seq, d1), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, sc_prop), MDODiscipline.STATUS_DONE)
        self.assertEqual(self.status_of(seq, d1, 1), MDODiscipline.STATUS_DONE)

    def test_visitor_pattern(self):
        class Visitor:
            def __init__(self):
                self.result = []

            def visit_atomic(self, atom):
                self.result.append(atom.discipline)

            def visit_serial(self, serial):
                self.result.append(serial)

            def visit_parallel(self, parallel):
                self.result.append(parallel)

            def visit_loop(self, loop):
                self.result.append(loop)

        serial = ExecutionSequenceFactory.serial().extend([self.d1, self.d2])
        parallel = ExecutionSequenceFactory.parallel([self.d3, self.d4])
        serial.extend(parallel)
        loop = ExecutionSequenceFactory.loop(self.d1, serial)

        visitor = Visitor()
        loop.accept(visitor)
        # prefix order expected
        expected = [loop, self.d1, serial, self.d1, self.d2, parallel, self.d3, self.d4]
        self.assertEqual(expected, visitor.result)
