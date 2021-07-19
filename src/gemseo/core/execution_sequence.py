# -*- coding: utf-8 -*-
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
"""
Abstraction for workflow
************************
"""
from __future__ import division, unicode_literals

import logging
from uuid import uuid4

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.py23_compat import OrderedDict  # automatically dict from py36

LOGGER = logging.getLogger(__name__)

STATUS_FAILED = MDODiscipline.STATUS_FAILED
STATUS_DONE = MDODiscipline.STATUS_DONE
STATUS_PENDING = MDODiscipline.STATUS_PENDING
STATUS_RUNNING = MDODiscipline.STATUS_RUNNING


class ExecutionSequence(object):
    """A base class for execution sequences.

    The execution sequence structure is introduced to reflect the main workflow
    implicitly executed by |g| regarding the given scenario/formulation executed. That
    structure allows to identify single executions of a same discipline that may be run
    several times at various stages in the given scenario/formulation.
    """

    START_STR = "["
    END_STR = "]"

    def __init__(self, sequence=None):  # pylint: disable=unused-argument
        # use an OrderedDict to get disc_to_uuids lists ordered regarding
        # a discipline repetitive appearance: useful for testing and debug
        self.uuid = str(uuid4())
        self.uuid_to_disc = OrderedDict()
        self.disc_to_uuids = {}
        self._status = None
        self._enabled = False
        self._parent = None

    def accept(self, visitor):
        """Accept a visitor object (see Visitor pattern). Have to be implemented by
        subclasses.

        :param visitor: a visitor object
        """
        raise NotImplementedError()

    def set_observer(self, obs):
        """Register the given observer object which is intended to be notified via its
        update() method each time an underlying discipline changes its status. To be
        implemented in subclasses.

        :returns: the disciplines list.
        """
        raise NotImplementedError()

    @property
    def status(self):
        """Get status value.

        :returns: the status value (MDODiscipline.STATUS_XXX values).
        """
        return self._status

    @status.setter
    def status(self, status):
        """Set status value.

        :param status: (MDODiscipline.STATUS_XXX values).
        """
        self._status = status

    @property
    def parent(self):
        """Get the containing execution sequence.

        :returns: a composite execution sequence.
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        """Set the containing execution sequence as parent. self should be included in
        parent.sequence_list.

        :returns: the status value.
        """
        if self not in parent.sequence_list:
            raise RuntimeError(
                "parent " + str(parent) + " do not include child " + str(self)
            )
        self._parent = parent

    def enabled(self):
        """Get activation state.

        :returns: boolean True if enabled.
        """
        return self._enabled

    def enable(self):
        """Set the execution sequence as activated (enabled)."""
        self.status = STATUS_PENDING
        self._enabled = True

    def disable(self):
        """Set the execution sequence as deactivated (disabled)."""
        self._enabled = False

    def _compute_disc_to_uuids(self):
        """
        Update discipline to uuids mapping from uuids to discipline mapping
        Note: a discipline might correspond to several
        AtomicExecutionSeuqence hence might correspond to several uuids.

        """
        self.disc_to_uuids = {}
        for key, value in self.uuid_to_disc.items():
            self.disc_to_uuids.setdefault(value, []).append(key)


class AtomicExecSequence(ExecutionSequence):
    """An execution sequence to represent the single execution of a given discipline."""

    def __init__(self, discipline=None):
        super(AtomicExecSequence, self).__init__(discipline)
        if not isinstance(discipline, MDODiscipline):
            raise Exception(
                "Atomic sequence shall be a discipline"
                + ", got "
                + str(type(discipline))
                + " instead !"
            )
        self.discipline = discipline
        self.uuid_to_disc = {self.uuid: discipline}
        self.disc_to_uuids = {discipline: [self.uuid]}
        self._observer = None

    def __str__(self):
        return self.discipline.name + "(" + str(self.status) + ")"

    def __repr__(self):
        return (
            self.discipline.name + "(" + str(self.status) + ", " + str(self.uuid) + ")"
        )

    def accept(self, visitor):
        """Accept a visitor object (see Visitor pattern)

        :param visitor: a visitor object implementing visit_atomic() method
        """
        visitor.visit_atomic(self)

    def set_observer(self, obs):
        """Register given observer obs to be notified (obs.update()) when discipline
        status changes.

        :param obs: the observe object implementing update() method
        """
        self._observer = obs

    def enable(self):
        """Subscribe to status changes of the discipline (notified via
        update_status())"""
        super(AtomicExecSequence, self).enable()
        self.discipline.add_status_observer(self)

    def disable(self):
        """Unsubscribe from receiving status changes of the discipline."""
        super(AtomicExecSequence, self).disable()
        self.discipline.remove_status_observer(self)

    def get_state_dict(self):
        """Get the dictionary of statuses mapping atom uuid to status.

        :returns: the status
        """

        return {self.uuid: self.status}

    def update_status(self, discipline):
        """
        Update status from given discipline.
        Reflect the status then notifies the parent and the observer if any.
        Note: update_status if discipline status change actually
        compared to current, otherwise do nothing.

        :param discipline: the discipline whose status changed

        """
        if self._enabled and self.status != discipline.status:
            self.status = discipline.status or STATUS_PENDING
            if self.status == STATUS_DONE or self.status == STATUS_FAILED:
                self.disable()
            if self._parent:
                self._parent.update_child_status(self)
            if self._observer:
                self._observer.update(self)

    def force_statuses(self, status):
        """Force the self status and the status of subsequences without notifying the
        parent (as the force_status is called by a parent), but notify the observer is
        status changed.

        :param: status value (see MDODiscipline.STATUS_XXX values)
        """
        old_status = self._status
        self._status = status
        if old_status != status and self._observer:
            self._observer.update(self)


class CompositeExecSequence(ExecutionSequence):
    """A base class for execution sequence made of other execution sequences.

    Intented to be subclassed.
    """

    START_STR = "'"
    END_STR = "'"

    def __init__(self, sequence=None):
        super(CompositeExecSequence, self).__init__(sequence)
        self.sequence_list = []
        self.disciplines = []

    def __str__(self):
        str_out = self.START_STR
        for seq in self.sequence_list:
            str_out += str(seq) + ", "
        str_out += self.END_STR
        return str_out

    def accept(self, visitor):
        """Accept a visitor object (see Visitor pattern) and then make its children
        accept it too.

        :param visitor: a visitor object implementing visit_serial() method
        """
        self._accept(visitor)
        for seq in self.sequence_list:
            seq.accept(visitor)

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern). To be specifically implemented
        by subclasses to call relevant visitor method depending the subclass type.

        :param visitor: a visitor object implementing visit_serial() method
        """
        raise NotImplementedError()

    def set_observer(self, obs):
        """Set observer obs to subsequences. Override super.set_observer()

        :param obs: observer object implementing update() method
        """
        for seq in self.sequence_list:
            seq.set_observer(obs)

    def disable(self):
        """Unsubscribe subsequences from receiving status changes of disciplines."""
        super(CompositeExecSequence, self).disable()
        for seq in self.sequence_list:
            seq.disable()

    def force_statuses(self, status):
        """Force the self status and the status of subsequences.

        params: status value (see MDODiscipline.STATUS_XXX values)
        """
        self.status = status
        for seq in self.sequence_list:
            seq.force_statuses(status)

    def get_state_dict(self):
        """Get the dictionary of statuses mapping atom uuid to status.

        :returns: the status
        """
        state_dict = {}
        for seq in self.sequence_list:
            state_dict.update(seq.get_state_dict())
        return state_dict

    def update_child_status(self, child):
        """Manage status change of child execution sequences. Propagates status change
        to the parent (containing execution sequence)

        :param child: the child execution sequence (contained in sequence_list)
            whose status has changed
        """
        old_status = self.status
        self._update_child_status(child)
        if self._parent and self.status != old_status:
            self._parent.update_child_status(self)

    def _update_child_status(self, child):
        """Handle child execution change. To be implemented in subclasses.

        :param child: the child execution sequence (contained in sequence_list)
            whose status has changed
        """
        raise NotImplementedError()


class ExtendableExecSequence(CompositeExecSequence):
    """A base class for composite execution sequence that are extendable.

    Intented to be subclassed.
    """

    def __init__(self, sequence=None):
        super(ExtendableExecSequence, self).__init__(sequence)
        if sequence is not None:
            self.extend(sequence)

    def extend(self, sequence):
        """Extend the execution sequence with another ExecutionSequence or a discipline.

        :param sequence: another execution sequence or
        """
        seq_class = sequence.__class__
        self_class = self.__class__
        if isinstance(sequence, list):
            # In this case we are initializing the sequence
            # or extending by a list of disciplines
            self._extend_with_disc_list(sequence)
        elif isinstance(sequence, MDODiscipline):
            # Sequence is extended by a single discipline: generate a new
            # uuid
            self._extend_with_disc_list([sequence])
        elif isinstance(sequence, AtomicExecSequence):
            # Sequence is extended by an AtomicSequence:
            # we extend
            self._extend_with_atomic_sequence(sequence)
        elif seq_class != self_class:
            # We extend by a different type of ExecSequence
            # So we append the other sequence as a sub structure
            self._extend_with_diff_sequence_kind(sequence)
        else:
            # We extend by a same type of ExecSequence
            # So we just extend the sequence
            self._extend_with_same_sequence_kind(sequence)
        self._compute_disc_to_uuids()  # refresh disc_to_uuids
        for seq in self.sequence_list:
            seq.parent = self
        return self

    def _extend_with_disc_list(self, sequence):
        """Extend by a list of disciplines.

        :param sequence: a list of MDODiscipline objects
        """
        seq_list = [AtomicExecSequence(disc) for disc in sequence]
        self.sequence_list.extend(seq_list)
        uuids_dict = {atom.uuid: atom.discipline for atom in seq_list}
        self.uuid_to_disc.update(uuids_dict)

    def _extend_with_atomic_sequence(self, sequence):
        """Extend by a list of AtomicExecutionSequence.

        :param sequence: a list of MDODiscipline objects
        """
        self.sequence_list.append(sequence)
        self.uuid_to_disc[sequence.uuid] = sequence

    def _extend_with_same_sequence_kind(self, sequence):
        """Extend by another ExecutionSequence of same type.

        :param sequence: an ExecutionSequence of same type as self
        """
        self.sequence_list.extend(sequence.sequence_list)
        self.uuid_to_disc.update(sequence.uuid_to_disc)

    def _extend_with_diff_sequence_kind(self, sequence):
        """Extend by another ExecutionSequence of different type.

        :param sequence: an ExecutionSequence of type different from self's one
        """
        self.sequence_list.append(sequence)
        self.uuid_to_disc.update(sequence.uuid_to_disc)

    def _update_child_status(self, child):
        """Manage status change of child execution sequences. Done status management is
        handled in subclasses.

        :param child: the child execution sequence (contained in sequence_list)
            whose status has changed
        """
        if child.status == STATUS_FAILED:
            self.status = STATUS_FAILED
        elif child.status == STATUS_DONE:
            self._update_child_done_status(child)
        else:
            self.status = child.status

    def _update_child_done_status(self, child):
        """Handle done status of child execution sequences. To be implemented in
        subclasses.

        :param child: the child execution sequence (contained in sequence_list)
            whose status has changed
        """
        raise NotImplementedError()


class SerialExecSequence(ExtendableExecSequence):
    """A class to describe a serial execution of disciplines."""

    START_STR = "["
    END_STR = "]"

    def __init__(self, sequence=None):
        super(SerialExecSequence, self).__init__(sequence)
        self.exec_index = None

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern)

        :param visitor: a visitor object implementing visit_serial() method
        """
        visitor.visit_serial(self)

    def enable(self):
        """Activate first child execution sequence."""
        super(SerialExecSequence, self).enable()
        self.exec_index = 0
        if self.sequence_list:
            self.sequence_list[self.exec_index].enable()
        else:
            raise Exception("Serial execution is empty")

    def _update_child_done_status(self, child):
        """Activate next child to given child execution sequence. Disable itself when
        all children done.

        :param child: the child execution sequence in done state.
        """
        if child.status == STATUS_DONE:
            child.disable()
            self.exec_index += 1
            if self.exec_index < len(self.sequence_list):
                self.sequence_list[self.exec_index].enable()
            else:  # last seq done
                self.status = STATUS_DONE
                self.disable()


class ParallelExecSequence(ExtendableExecSequence):
    """A class to describe a parallel execution of disciplines."""

    START_STR = "("
    END_STR = ")"

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern)

        :param visitor: a visitor object implementing visit_parallel() method
        """
        visitor.visit_parallel(self)

    def enable(self):
        """Activate all child execution sequences."""
        super(ParallelExecSequence, self).enable()
        for seq in self.sequence_list:
            seq.enable()

    def _update_child_done_status(self, child):  # pylint: disable=unused-argument
        """Disable itself when all children done.

        :param child: the child execution sequence in done state.
        """
        all_done = True
        for seq in self.sequence_list:
            all_done = all_done and (seq.status == STATUS_DONE)
        if all_done:
            self.status = STATUS_DONE
            self.disable()


class LoopExecSequence(CompositeExecSequence):
    """A class to describe a loop with a controller discipline and an execution_sequence
    as iterate."""

    START_STR = "{"
    END_STR = "}"

    def __init__(self, controller, sequence):
        if isinstance(controller, AtomicExecSequence):
            control = controller
        elif not isinstance(controller, MDODiscipline):
            raise Exception(
                "Controller of a loop shall be a discipline"
                + ", got "
                + str(type(controller))
                + " instead !"
            )
        else:
            control = AtomicExecSequence(controller)
        if not isinstance(sequence, CompositeExecSequence):
            raise Exception(
                "Sequence of a loop shall be a composite execution sequence"
                + ", got "
                + str(type(sequence))
                + " instead !"
            )
        super(LoopExecSequence, self).__init__()
        self.sequence_list = [control, sequence]
        self.atom_controller = control
        self.atom_controller.parent = self
        self.iteration_sequence = sequence
        self.iteration_sequence.parent = self
        self.uuid_to_disc.update(sequence.uuid_to_disc)
        self.uuid_to_disc[self.atom_controller.uuid] = controller
        self._compute_disc_to_uuids()
        self.iteration_count = 0

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern)

        :param visitor: a visitor object implementing visit_loop() method
        """
        visitor.visit_loop(self)

    def enable(self):
        """Active controller execution sequence."""
        super(LoopExecSequence, self).enable()
        self.atom_controller.enable()
        self.iteration_count = 0

    def _update_child_status(self, child):
        """Activate iteration successively regarding controller status. Count iterations
        regarding iteration_sequence status.

        :param child: the child execution sequence in done state.
        """
        self.status = self.atom_controller.status
        if child == self.atom_controller:
            if self.status == STATUS_RUNNING:
                if not self.iteration_sequence.enabled():
                    self.iteration_sequence.enable()
            elif self.status == STATUS_DONE:
                self.disable()
                self.force_statuses(STATUS_DONE)
        if child == self.iteration_sequence:
            if child.status == STATUS_DONE:
                self.iteration_count += 1
                self.iteration_sequence.enable()
        if child.status == STATUS_FAILED:
            self.status = STATUS_FAILED


class ExecutionSequenceFactory(object):
    """A factory class for ExecutionSequence objects.

    Allow to create AtomicExecutionSequence, SerialExecutionSequence,
    ParallelExecutionSequence and LoopExecutionSequence. Main |g| workflow is intended
    to be expressed with those four ExecutionSequence types
    """

    @staticmethod
    def atom(discipline):
        """Returns a structure representing the execution of a discipline. This function
        is intended to be called by MDOFormulation.get_expected_workflow methods.

        :param discipline: a discipline
        :returns: the structure used within XDSM workflow representation
        """
        return AtomicExecSequence(discipline)

    @staticmethod
    def serial(sequence=None):
        """Returns a structure representing the serial execution of the given
        disciplines. This function is intended to be called by
        MDOFormulation.get_expected_workflow methods.

        :param sequence: any number of discipline
            or the return value of a serial, parallel or loop call
        :returns: a serial execution sequence
        """
        return SerialExecSequence(sequence)

    @staticmethod
    def parallel(sequence=None):
        """Returns a structure representing the parallel execution of the given
        disciplines. This function is intended to be called by
        MDOFormulation.get_expected_workflow methods.

        :param sequence: any number of discipline or
            the return value of a serial, parallel or loop call
        :returns: a parallel execution sequence
        """
        return ParallelExecSequence(sequence)

    @staticmethod
    def loop(control, composite_sequence):
        """Returns a structure representing a loop execution of a This function is
        intended to be called by MDOFormulation.get_expected_workflow methods.

        :param control: the discipline object, controller of the loop
        :param composite_sequence: any number of discipline
            or the return value of a serial, parallel or loop call
        :returns: a loop execution sequence
        """
        return LoopExecSequence(control, composite_sequence)
