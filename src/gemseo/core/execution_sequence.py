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
"""Abstraction for workflow."""
from __future__ import annotations

import logging
from abc import abstractmethod
from typing import Iterable
from uuid import uuid4

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

LOGGER = logging.getLogger(__name__)

STATUS_FAILED = MDODiscipline.STATUS_FAILED
STATUS_DONE = MDODiscipline.STATUS_DONE
STATUS_PENDING = MDODiscipline.STATUS_PENDING
STATUS_RUNNING = MDODiscipline.STATUS_RUNNING


class ExecutionSequence(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A base class for execution sequences.

    The execution sequence structure is introduced to reflect the main workflow
    implicitly executed by |g| regarding the given scenario/formulation executed. That
    structure allows to identify single executions of a same discipline that may be run
    several times at various stages in the given scenario/formulation.
    """

    START_STR = "["
    END_STR = "]"

    def __init__(self, sequence=None) -> None:
        """
        Args:
            sequence: This argument is not used.
        """  # noqa: D205, D212, D415
        self.uuid = str(uuid4())
        self.uuid_to_disc = {}
        self.disc_to_uuids = {}
        self._status = None
        self._enabled = False
        self._parent = None

    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor object (see Visitor pattern).

        Have to be implemented by subclasses.

        Args:
            visitor: A visitor object.
        """

    @abstractmethod
    def set_observer(self, obs):
        """Register an observer.

        This observer is intended to be notified via its :meth:`update` method
        each time an underlying discipline changes its status.
        To be implemented in subclasses.

        Returns:
            The disciplines.
        """

    @property
    def status(self):
        """Get the value of the status.

        One of :attr:`.MDODiscipline.AVAILABLE_STATUSES`.

        Returns:
            The value of the status.
        """
        return self._status

    @status.setter
    def status(self, status):
        """Set the value of the status.

        One of :attr:`.MDODiscipline.AVAILABLE_STATUSES`.

        Args:
            status: The value of the status
        """
        self._status = status

    @property
    def parent(self):
        """Get the containing execution sequence.

        Returns:
             The execution sequence containing the current one.
        """
        return self._parent

    @parent.setter
    def parent(self, parent):
        """Set the containing execution sequence as parent.

        Args:
            parent: An execution sequence.

        Raises:
            RuntimeError: When the current execution sequence is not a child
                of the given parent execution sequence.
        """
        if self not in parent.sequences:
            raise RuntimeError(f"parent {parent} does not include child {self}")
        self._parent = parent

    def enabled(self):
        """Get activation state.

        Returns:
            Whether the execution sequence is enabled.
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
        """Update discipline to uuids mapping from uuids to discipline mapping.

        Note:
            A discipline might correspond to several AtomicExecutionSeuqence hence
            might correspond to several uuids.
        """
        self.disc_to_uuids = {}
        for key, value in self.uuid_to_disc.items():
            self.disc_to_uuids.setdefault(value, []).append(key)


class AtomicExecSequence(ExecutionSequence):
    """An execution sequence to represent the single execution of a given discipline."""

    def __init__(self, discipline: MDODiscipline | None = None) -> None:
        """
        Args:
            discipline: A discipline.
        """  # noqa: D205, D212, D415
        super().__init__()
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
        """Accept a visitor object (see Visitor pattern).

        Args:
            visitor: An object implementing the :meth:`visit_atomic` method.
        """
        visitor.visit_atomic(self)

    def set_observer(self, obs):
        """Register a given observer to be notified when discipline status changes.

        Args:
            obs: An object implementing the :meth:`update` method for notification.
        """
        self._observer = obs

    def enable(self):
        """Subscribe to status changes of the discipline.

        Notified via the :meth:`update_status` method.
        """
        super().enable()
        self.discipline.add_status_observer(self)

    def disable(self):
        """Unsubscribe from receiving status changes of the discipline."""
        super().disable()
        self.discipline.remove_status_observer(self)

    def get_statuses(self):
        """Get the dictionary of statuses mapping atom uuid to status.

        Args:
            The statuses mapping atom uuid to status.
        """
        return {self.uuid: self.status}

    def update_status(self, discipline):
        """Update status from given discipline.

        Reflect the status then notifies the parent and the observer if any.
        Note: update_status if discipline status change actually
        compared to current, otherwise do nothing.

        Args:
            discipline: The discipline whose status changed.
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
        """Force the self status and the status of subsequences.

        This is done without notifying the
        parent (as the force_status is called by a parent), but notify the observer is
        status changed.

        Args:
            status: The value of the status,
                one of :attr:`.MDODiscipline.AVAILABLE_STATUSES`.
        """
        old_status = self._status
        self._status = status
        if old_status != status and self._observer:
            self._observer.update(self)


class CompositeExecSequence(ExecutionSequence):
    """A base class for execution sequence made of other execution sequences.

    Intended to be subclassed.
    """

    START_STR = "'"
    END_STR = "'"

    sequences: list[ExecutionSequence]
    """The inner execution sequences."""

    disciplines: list[MDODiscipline]
    """The disciplines."""

    def __init__(self, sequence=None) -> None:  # noqa:D107
        super().__init__()
        self.sequences = []
        self.disciplines = []

    def __str__(self) -> str:
        string = self.START_STR
        for sequence in self.sequences:
            string += str(sequence) + ", "
        string += self.END_STR
        return string

    def accept(self, visitor) -> None:
        """Accept a visitor object and then make its children accept it too.

        Args:
            visitor: A visitor object implementing the :meth:`visit_serial` method.
        """
        self._accept(visitor)
        for sequence in self.sequences:
            sequence.accept(visitor)

    @abstractmethod
    def _accept(self, visitor) -> None:
        """Accept a visitor object (see Visitor pattern).

        To be specifically implemented
        by subclasses to call relevant visitor method depending on the subclass type.

        Args:
            visitor: An object implementing the :meth:`visit_serial` method.
        """

    def set_observer(self, obs) -> None:
        """Set observer obs to subsequences.

        Override super.set_observer()

        Args:
            obs: An object implementing the meth:`update` method.
        """
        for sequence in self.sequences:
            sequence.set_observer(obs)

    def disable(self) -> None:
        """Unsubscribe subsequences from receiving status changes of disciplines."""
        super().disable()
        for sequence in self.sequences:
            sequence.disable()

    def force_statuses(self, status) -> None:
        """Force the self status and the status of subsequences.

        Args:
            status: The value of the status,
                one of :attr:`.MDODiscipline.AVAILABLE_STATUSES`.
        """
        self.status = status
        for sequence in self.sequences:
            sequence.force_statuses(status)

    def get_statuses(self):
        """Get the dictionary of statuses mapping atom uuid to status.

        Returns:
            The statuses related to the atom uuid.
        """
        uuids_to_statuses = {}
        for sequence in self.sequences:
            uuids_to_statuses.update(sequence.get_statuses())
        return uuids_to_statuses

    def update_child_status(self, child) -> None:
        """Manage status change of child execution sequences.

        Propagates status change
        to the parent (containing execution sequence).

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """
        old_status = self.status
        self._update_child_status(child)
        if self._parent and self.status != old_status:
            self._parent.update_child_status(self)

    @abstractmethod
    def _update_child_status(self, child):
        """Handle child execution change.

        To be implemented in subclasses.

        Args:
            child: the child execution sequence (contained in sequences)
                whose status has changed.
        """


class ExtendableExecSequence(CompositeExecSequence):
    """A base class for composite execution sequence that are extendable.

    Intended to be subclassed.
    """

    def __init__(self, sequence=None):  # noqa:D107
        super().__init__()
        if sequence is not None:
            self.extend(sequence)

    def extend(self, sequence):
        """Extend the execution sequence with another sequence or discipline(s).

        Args:
            sequence: Either another execution sequence or one or several disciplines.

        Returns:
            The extended execution sequence.
        """
        seq_class = sequence.__class__
        self_class = self.__class__
        if isinstance(sequence, list):
            # In this case we are initializing the sequence
            # or extending by a list of disciplines
            self._extend_with_disciplines(sequence)
        elif isinstance(sequence, MDODiscipline):
            # Sequence is extended by a single discipline: generate a new
            # uuid
            self._extend_with_disciplines([sequence])
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
        for sequence in self.sequences:
            sequence.parent = self
        return self

    def _extend_with_disciplines(self, disciplines: Iterable[MDODiscipline]) -> None:
        """Extend the sequence with disciplines.

        Args:
            disciplines: A collection of disciplines.
        """
        sequences = [AtomicExecSequence(discipline) for discipline in disciplines]
        self.sequences.extend(sequences)
        self.uuid_to_disc.update(
            {sequence.uuid: sequence.discipline for sequence in sequences}
        )

    def _extend_with_atomic_sequence(self, sequence):
        """Extend by a list of AtomicExecutionSequence.

        Args:
            sequence: A list of MDODiscipline objects.
        """
        self.sequences.append(sequence)
        self.uuid_to_disc[sequence.uuid] = sequence

    def _extend_with_same_sequence_kind(self, sequence):
        """Extend by another ExecutionSequence of same type.

        Args:
            sequence: An ExecutionSequence of same type as self.
        """
        self.sequences.extend(sequence.sequences)
        self.uuid_to_disc.update(sequence.uuid_to_disc)

    def _extend_with_diff_sequence_kind(self, sequence):
        """Extend by another ExecutionSequence of different type.

        Args:
            sequence: An ExecutionSequence of type different from self's one.
        """
        self.sequences.append(sequence)
        self.uuid_to_disc.update(sequence.uuid_to_disc)

    def _update_child_status(self, child):
        """Manage status change of child execution sequences.

        Done status management is handled in subclasses.

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """
        if child.status == STATUS_FAILED:
            self.status = STATUS_FAILED
        elif child.status == STATUS_DONE:
            self._update_child_done_status(child)
        else:
            self.status = child.status

    @abstractmethod
    def _update_child_done_status(self, child):
        """Handle done status of child execution sequences.

        To be implemented in subclasses.

        Args:
            child: The child execution sequence (contained in sequences)
                whose status has changed.
        """


class SerialExecSequence(ExtendableExecSequence):
    """A class to describe a serial execution of disciplines."""

    START_STR = "["
    END_STR = "]"

    def __init__(self, sequence=None):  # noqa:D107
        super().__init__(sequence)
        self.exec_index = None

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern).

        Args:
            visitor: An object implementing the :meth:`visit_serial` method.
        """
        visitor.visit_serial(self)

    def enable(self):
        """Activate first child execution sequence."""
        super().enable()
        self.exec_index = 0
        if self.sequences:
            self.sequences[self.exec_index].enable()
        else:
            raise Exception("Serial execution is empty")

    def _update_child_done_status(self, child):
        """Activate next child to given child execution sequence.

        Disable itself when all children done.

        Args:
            child: The child execution sequence in done state.
        """
        if child.status == STATUS_DONE:
            child.disable()
            self.exec_index += 1
            if self.exec_index < len(self.sequences):
                self.sequences[self.exec_index].enable()
            else:  # last seq done
                self.status = STATUS_DONE
                self.disable()


class ParallelExecSequence(ExtendableExecSequence):
    """A class to describe a parallel execution of disciplines."""

    START_STR = "("
    END_STR = ")"

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern).

        Args:
            visitor: An object implementing the :meth:`visit_serial` method.
        """
        visitor.visit_parallel(self)

    def enable(self):
        """Activate all child execution sequences."""
        super().enable()
        for sequence in self.sequences:
            sequence.enable()

    def _update_child_done_status(self, child):  # pylint: disable=unused-argument
        """Disable itself when all children done.

        Args:
            child: The child execution sequence in done state.
        """
        all_done = True
        for sequence in self.sequences:
            all_done = all_done and (sequence.status == STATUS_DONE)
        if all_done:
            self.status = STATUS_DONE
            self.disable()


class LoopExecSequence(CompositeExecSequence):
    """A loop with a controller discipline and an execution_sequence as iterate."""

    START_STR = "{"
    END_STR = "}"

    def __init__(self, controller, sequence):
        """
        Args:
            controller: A controller.
            sequence: A sequence.
        """  # noqa: D205, D212, D415
        if isinstance(controller, AtomicExecSequence):
            control = controller
        elif not isinstance(controller, MDODiscipline):
            raise Exception(
                "Controller of a loop shall be a discipline, "
                f"got {type(controller)} instead."
            )
        else:
            control = AtomicExecSequence(controller)
        if not isinstance(sequence, CompositeExecSequence):
            raise Exception(
                "Sequence of a loop shall be a composite execution sequence, "
                f"got {type(sequence)} instead."
            )
        super().__init__()
        self.sequences = [control, sequence]
        self.atom_controller = control
        self.atom_controller.parent = self
        self.iteration_sequence = sequence
        self.iteration_sequence.parent = self
        self.uuid_to_disc.update(sequence.uuid_to_disc)
        self.uuid_to_disc[self.atom_controller.uuid] = controller
        self._compute_disc_to_uuids()
        self.iteration_count = 0

    def _accept(self, visitor):
        """Accept a visitor object (see Visitor pattern).

        Args:
            visitor: An object implementing the :meth:`visit_serial` method.
        """
        visitor.visit_loop(self)

    def enable(self):
        """Active controller execution sequence."""
        super().enable()
        self.atom_controller.enable()
        self.iteration_count = 0

    def _update_child_status(self, child):
        """Activate iteration successively regarding controller status.

        Count iterations regarding iteration_sequence status.

        Args:
            child: The child execution sequence in done state.
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


class ExecutionSequenceFactory:
    """A factory class for ExecutionSequence objects.

    Allow to create AtomicExecutionSequence, SerialExecutionSequence,
    ParallelExecutionSequence and LoopExecutionSequence. Main |g| workflow is intended to
    be expressed with those four ExecutionSequence types
    """

    @staticmethod
    def atom(discipline):
        """Return a structure representing the execution of a discipline.

        This function
        is intended to be called by MDOFormulation.get_expected_workflow methods.

        Args:
            discipline: A discipline.

        Returns:
            The structure used within XDSM workflow representation.
        """
        return AtomicExecSequence(discipline)

    @staticmethod
    def serial(sequence=None):
        """Return a structure representing the serial execution of disciplines.

        This function is intended to be called by MDOFormulation.get_expected_workflow
        methods.

        Args:
            sequence: Any number of discipline
                or the return value of a serial, parallel or loop call.

        Returns:
            A serial execution sequence.
        """
        return SerialExecSequence(sequence)

    @staticmethod
    def parallel(sequence=None):
        """Return a structure representing the parallel execution of disciplines.

        This function is intended to be called by MDOFormulation.get_expected_workflow
        methods.

        Args:
            sequence: Any number of discipline or
                the return value of a serial, parallel or loop call.

        Returns:
            A parallel execution sequence.
        """
        return ParallelExecSequence(sequence)

    @staticmethod
    def loop(control, composite_sequence):
        """Return a structure representing a loop execution of a function.

        It is intended to be called by MDOFormulation.get_expected_workflow methods.

        Args:
            control: The discipline object, controller of the loop.
            composite_sequence: Any number of discipline
                or the return value of a serial, parallel or loop call.

        Returns:
            A loop execution sequence.
        """
        return LoopExecSequence(control, composite_sequence)
