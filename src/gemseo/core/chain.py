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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Chains of disciplines.

Can be both sequential or parallel execution processes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from strenum import LowercaseStrEnum
from strenum import StrEnum

from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.chain_rule import traverse_add_diff_io
from gemseo.core.derivatives.jacobian_operator import JacobianOperator
from gemseo.core.discipline import MDODiscipline
from gemseo.core.discipline_data import DisciplineData
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.core.parallel_execution.disc_parallel_execution import DiscParallelExecution
from gemseo.core.parallel_execution.disc_parallel_linearization import (
    DiscParallelLinearization,
)
from gemseo.utils.compatibility.scipy import array_classes
from gemseo.utils.data_conversion import deepcopy_dict_of_arrays
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.enumeration import merge_enums

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray

LOGGER = logging.getLogger(__name__)


# TODO: One class per module.
class MDOChain(MDODiscipline):
    """Chain of disciplines that is based on a predefined order of execution."""

    class _DerivationMode(LowercaseStrEnum):
        """The derivation modes."""

        REVERSE = "reverse"
        """The reverse Jacobian accumulation, chain rule from outputs to inputs."""

        AUTO = "auto"
        """Automatic switch between direct, reverse or adjoint depending on data
        sizes."""

    LinearizationMode = merge_enums(
        "LinearizationMode",
        StrEnum,
        ApproximationMode,
        _DerivationMode,
    )

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            name: The name of the discipline.
                If ``None``, use the class name.
            grammar_type: The type of the input and output grammars.
        """  # noqa: D205, D212, D415
        super().__init__(name, grammar_type=grammar_type)
        self._disciplines = disciplines
        self.initialize_grammars()
        self._coupling_structure = None
        self._last_diff_inouts = None

    def set_disciplines_statuses(
        self,
        status: str,
    ) -> None:
        """Set the sub-disciplines statuses.

        Args:
            status: The status to be set.
        """
        for discipline in self.disciplines:
            discipline.status = status
            discipline.set_disciplines_statuses(status)

    def initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        self.input_grammar.clear()
        self.output_grammar.clear()
        for discipline in self.disciplines:
            self.input_grammar.update(
                discipline.input_grammar, exclude_names=self.output_grammar.keys()
            )
            self.output_grammar.update(discipline.output_grammar)

    def _run(self) -> None:
        for discipline in self.disciplines:
            self.local_data.update(discipline.execute(self.local_data))

    def reverse_chain_rule(
        self,
        chain_outputs: Iterable[str],
        discipline: MDODiscipline,
    ) -> None:
        """Chain the derivatives with a new discipline in the chain in reverse mode.

        Perform chain ruling:
        (notation: D is total derivative, d is partial derivative)

        D out    d out      dinpt_1    d output      dinpt_2
        -----  = -------- . ------- + -------- . --------
        D new_in  d inpt_1  d new_in   d inpt_2   d new_in


        D out    d out        d out      dinpt_2
        -----  = -------- + -------- . --------
        D z      d z         d inpt_2     d z


        D out    d out      [dinpt_1   d out      d inpt_1    dinpt_2 ]
        -----  = -------- . [------- + -------- . --------  . --------]
        D z      d inpt_1   [d z       d inpt_1   d inpt_2     d z    ]

        Args:
            discipline: The new discipline to compose in the chain.
            chain_outputs: The outputs to lineariza.
        """
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = discipline.get_input_data()
        # The graph traversal algorithm avoid to compute unnecessary Jacobians
        discipline.linearize(last_cached, execute=False, compute_all_jacobians=False)

        for output_name in chain_outputs:
            if output_name in self.jac:
                # This output has already been taken from previous disciplines
                # Derivatives must be composed using the chain rule

                # Make a copy of the keys because the dict is changed in the
                # loop
                existing_inputs = self.jac[output_name].keys()
                common_inputs = set(existing_inputs) & set(discipline.jac)
                for input_name in common_inputs:
                    # Store reference to the current Jacobian
                    curr_jac = self.jac[output_name][input_name]
                    for new_in, new_jac in discipline.jac[input_name].items():
                        # Chain rule the derivatives
                        # TODO: sum BEFORE dot
                        if isinstance(new_jac, JacobianOperator):
                            # NumPy array @ JacobianOperator is not supported, thus
                            # imposing to explictly use the __rmatmul__ method.
                            loc_dot = new_jac.__rmatmul__(curr_jac)
                        else:
                            loc_dot = curr_jac @ new_jac

                        # when input_name==new_in, we are in the case of an
                        # input being also an output
                        # in this case we must only compose the derivatives
                        if new_in in self.jac[output_name] and input_name != new_in:
                            # The output is already linearized wrt this
                            # input_name. We are in the case:
                            # d o     d o    d o     di_2
                            # ----  = ---- + ----- . -----
                            # d z     d z    d i_2    d z
                            if isinstance(loc_dot, JacobianOperator):
                                self.jac[output_name][new_in] = (
                                    loc_dot + self.jac[output_name][new_in]
                                )
                            else:
                                self.jac[output_name][new_in] += loc_dot
                        else:
                            # The output is not yet linearized wrt this
                            # input_name.  We are in the case:
                            #  d o      d o     di_1   d o     di_2
                            # -----  = ------ . ---- + ----  . ----
                            #  d x      d i_1   d x    d i_2    d x
                            self.jac[output_name][new_in] = loc_dot

            elif output_name in discipline.jac:
                # Output of the chain not yet filled in jac,
                # Take the jacobian dict of the current discipline to
                # Initialize. Make a copy !
                self.jac[output_name] = MDOChain.copy_jacs(discipline.jac[output_name])

    def _compute_diff_in_outs(
        self, inputs: Iterable[str], outputs: Iterable[str]
    ) -> None:
        if self._coupling_structure is None:
            self._coupling_structure = MDOCouplingStructure(self.disciplines)

        diff_ios = (set(inputs), set(outputs))
        if self._last_diff_inouts != diff_ios:
            traverse_add_diff_io(self._coupling_structure.graph.graph, inputs, outputs)
            self._last_diff_inouts = diff_ios

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._compute_diff_in_outs(inputs, outputs)

        # Initializes self jac with copy of last discipline (reverse mode)
        last_discipline = self.disciplines[-1]
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = last_discipline.get_input_data()

        # The graph traversal algorithm avoid to compute unnecessary Jacobians
        last_discipline.linearize(
            last_cached, execute=False, compute_all_jacobians=False
        )
        self.jac = self.copy_jacs(last_discipline.jac)

        # reverse mode of remaining disciplines
        remaining_disciplines = self.disciplines[:-1]
        for discipline in remaining_disciplines[::-1]:
            self.reverse_chain_rule(outputs, discipline)

        # Remove differentiations that should not be there,
        # because inputs are not inputs of the chain
        for output_jacobian in self.jac.values():
            # Copy keys because the dict in changed in the loop
            input_names_before_loop = list(output_jacobian.keys())
            for input_name in input_names_before_loop:
                if input_name not in inputs:
                    del output_jacobian[input_name]

        # Add differentiations that should be there,
        # because inputs of the chain but not of all disciplines.
        self._init_jacobian(
            inputs,
            outputs,
            fill_missing_keys=True,
            init_type=MDODiscipline.InitJacobianType.SPARSE,
        )

    @staticmethod
    def copy_jacs(
        jacobian: dict[str, dict[str, ndarray]],
    ) -> dict[str, dict[str, ndarray]]:
        """Deepcopy a Jacobian dictionary.

        Args:
            jacobian: The Jacobian dictionary,
                which is a nested dictionary as ``{'out': {'in': derivatives}}``.

        Returns:
            The deepcopy of the Jacobian dictionary.
        """
        jacobian_copy = {}
        for output_name, output_jacobian in jacobian.items():
            if isinstance(output_jacobian, dict):
                output_jacobian_copy = {}
                jacobian_copy[output_name] = output_jacobian_copy
                for input_name, derivatives in output_jacobian.items():
                    output_jacobian_copy[input_name] = derivatives.copy()
            elif isinstance(output_jacobian, (array_classes, JacobianOperator)):
                jacobian_copy[output_name] = output_jacobian.copy()

        return jacobian_copy

    def reset_statuses_for_run(self) -> None:  # noqa: D102
        super().reset_statuses_for_run()
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def get_expected_workflow(self) -> None:  # noqa: D102
        sequence = ExecutionSequenceFactory.serial()
        for discipline in self.disciplines:
            sequence.extend(discipline.get_expected_workflow())
        return sequence

    def get_expected_dataflow(self) -> None:  # noqa: D102
        disciplines = self.get_disciplines_in_dataflow_chain()
        graph = DependencyGraph(disciplines)
        disciplines_couplings = graph.get_disciplines_couplings()

        # Add discipline inner couplings (ex. MDA case)
        for discipline in disciplines:
            disciplines_couplings.extend(discipline.get_expected_dataflow())

        return disciplines_couplings

    def get_disciplines_in_dataflow_chain(self) -> list[MDODiscipline]:  # noqa: D102
        dataflow = []
        for disc in self.disciplines:
            dataflow.extend(disc.get_disciplines_in_dataflow_chain())
        return dataflow

    def _set_cache_tol(
        self,
        cache_tol: float,
    ) -> None:
        super()._set_cache_tol(cache_tol)
        for discipline in self.disciplines:
            discipline.cache_tol = cache_tol or 0.0


class MDOParallelChain(MDODiscipline):
    """Chain of processes that executes disciplines in parallel."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        use_threading: bool = True,
        n_processes: int | None = None,
        use_deep_copy: bool = False,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            name: The name of the discipline.
                If ``None``, use the class name.
            grammar_type: The type of the input and output grammars.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
                If ``None``, uses the number of disciplines.
            use_deep_copy: Whether to deepcopy the discipline input data.

        Notes:
            The actual number of processes could be lower than ``n_processes``
            if there are less than ``n_processes`` disciplines.
            ``n_processes`` can be lower than the total number of CPUs on the machine.
            Each discipline may itself run on several CPUs.
        """  # noqa: D205, D212, D415
        super().__init__(name, grammar_type=grammar_type)
        self._disciplines = disciplines
        self._use_deep_copy = use_deep_copy
        self.initialize_grammars()
        if n_processes is None:
            n_processes = len(self.disciplines)

        parallel_execution = DiscParallelExecution(
            self.disciplines, n_processes, use_threading=use_threading
        )
        self.parallel_execution = parallel_execution
        parallel_linearization = DiscParallelLinearization(
            self.disciplines, n_processes, use_threading=use_threading
        )
        self.parallel_lin = parallel_linearization

    def initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        self.input_grammar.clear()
        self.output_grammar.clear()
        for discipline in self.disciplines:
            self.input_grammar.update(discipline.input_grammar)
            self.output_grammar.update(discipline.output_grammar)

    def _get_input_data_copies(self) -> list[DisciplineData]:
        """Return copies of the input data, one per discipline.

        Returns:
            One copy of the input data per discipline.
        """
        # Avoid overlaps with dicts in // by doing a deepcopy
        # The outputs of a discipline may be a coupling, and shall therefore
        # not be passed as input of another since the execution are assumed
        # to be independent here
        if self._use_deep_copy:
            return [
                DisciplineData(deepcopy_dict_of_arrays(self.local_data))
                for _ in range(len(self._disciplines))
            ]

        for value in self.local_data.values():
            value.flags.writeable = False

        return [self.local_data] * len(self._disciplines)

    def _run(self) -> None:
        self.parallel_execution.execute(self._get_input_data_copies())

        # Update data according to input order of priority
        for discipline in self.disciplines:
            self.local_data.update({
                output_name: discipline.local_data[output_name]
                for output_name in discipline.get_output_data_names()
            })

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self._set_disciplines_diff_outputs(outputs)
        self._set_disciplines_diff_inputs(inputs)
        jacobians = self.parallel_lin.execute(self._get_input_data_copies())
        self.jac = {}
        # Update jacobians according to input order of priority
        for discipline_jacobian in jacobians:
            for output_name, output_jacobian in discipline_jacobian.items():
                chain_jacobian = self.jac.get(output_name)
                if chain_jacobian is None:
                    chain_jacobian = {}
                    self.jac[output_name] = chain_jacobian
                chain_jacobian.update(output_jacobian)

        self._init_jacobian(
            inputs,
            outputs,
            fill_missing_keys=True,
            init_type=self.InitJacobianType.SPARSE,
        )

    def add_differentiated_inputs(  # noqa: D102
        self,
        inputs: Iterable[str] | None = None,
    ) -> None:
        MDODiscipline.add_differentiated_inputs(self, inputs)
        self._set_disciplines_diff_inputs(inputs)

    def _set_disciplines_diff_inputs(
        self,
        inputs: Iterable[str],
    ) -> None:
        """Add the inputs to the right sub discipline's differentiated inputs.

        Args:
            inputs: The names of the inputs to be added.
        """
        diff_inpts = set(inputs)
        for discipline in self.disciplines:
            inputs_set = set(discipline.get_input_data_names()) & diff_inpts
            if inputs_set:
                discipline.add_differentiated_inputs(list(inputs_set))

    def add_differentiated_outputs(  # noqa: D102
        self,
        outputs: Iterable[str] | None = None,
    ) -> None:
        MDODiscipline.add_differentiated_outputs(self, outputs)
        self._set_disciplines_diff_outputs(outputs)

    def _set_disciplines_diff_outputs(self, outputs: Iterable[str]) -> None:
        """Add the outputs to the right-sub discipline's differentiated outputs.

        Args:
            outputs: The outputs to be added.
        """
        diff_outpts = set(outputs)
        for discipline in self.disciplines:
            outputs_set = set(discipline.get_output_data_names()) & diff_outpts
            if outputs_set:
                discipline.add_differentiated_outputs(list(outputs_set))

    def reset_statuses_for_run(self) -> None:  # noqa: D102
        super().reset_statuses_for_run()
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def get_expected_workflow(self) -> SerialExecSequence:  # noqa: D102
        sequence = ExecutionSequenceFactory.parallel()
        for discipline in self.disciplines:
            sequence.extend(discipline.get_expected_workflow())
        return sequence

    def get_expected_dataflow(  # noqa: D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return []

    def get_disciplines_in_dataflow_chain(self) -> list[MDODiscipline]:  # noqa: D102
        return [
            sub_discipline
            for discipline in self.disciplines
            for sub_discipline in discipline.get_disciplines_in_dataflow_chain()
        ]

    def _set_cache_tol(
        self,
        cache_tol: float,
    ) -> None:
        super()._set_cache_tol(cache_tol)
        for discipline in self.disciplines:
            discipline.cache_tol = cache_tol or 0.0


class MDOAdditiveChain(MDOParallelChain):
    """Execute disciplines in parallel and sum specified outputs across disciplines."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        outputs_to_sum: Iterable[str],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        use_threading: bool = True,
        n_processes: int | None = None,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            outputs_to_sum: The names of the outputs to sum.
            name: The name of the discipline.
                If ``None``, use the class name.
            grammar_type: The type of the input and output grammars.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
                If ``None``, uses the number of disciplines.

        Notes:
            The actual number of processes could be lower than ``n_processes``
            if there are less than ``n_processes`` disciplines.
            ``n_processes`` can be lower than the total number of CPUs on the machine.
            Each discipline may itself run on several CPUs.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines, name, grammar_type, use_threading, n_processes)
        self._outputs_to_sum = outputs_to_sum

    def _run(self) -> None:
        # Run the disciplines in parallel
        MDOParallelChain._run(self)

        # Sum the required outputs across disciplines
        for output_name in self._outputs_to_sum:
            disciplinary_outputs = [
                discipline.local_data[output_name]
                for discipline in self.disciplines
                if output_name in discipline.local_data
            ]
            self.local_data[output_name] = (
                sum(disciplinary_outputs) if disciplinary_outputs else None
            )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        # Differentiate the disciplines in parallel
        MDOParallelChain._compute_jacobian(self, inputs, outputs)

        # Sum the Jacobians of the required outputs across disciplines
        for output_name in self._outputs_to_sum:
            self.jac[output_name] = {}
            for input_name in inputs:
                disciplinary_jacobians = [
                    discipline.jac[output_name][input_name]
                    for discipline in self.disciplines
                    if input_name in discipline.jac[output_name]
                ]

                assert disciplinary_jacobians
                self.jac[output_name][input_name] = sum(disciplinary_jacobians)


class MDOWarmStartedChain(MDOChain):
    """Chain capable of warm starting a given list of variables.

    The values of the variables to warm start are stored after each run and used to
    initialize the next one.

    This Chain cannot be linearized.
    """

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        variable_names_to_warm_start: Sequence[str],
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            variable_names_to_warm_start: The names of the variables to be warm started.
                These names must be outputs of the disciplines in the chain.
                If the list is empty, no variables are warm started.
            name: The name of the discipline.
                If ``None``, use the class name.
            grammar_type: The type of the input and output grammars.

        Raises:
            ValueError: If the variable names to warm start are not outputs of the
                chain.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines=disciplines, name=name, grammar_type=grammar_type)
        self._variable_names_to_warm_start = variable_names_to_warm_start
        self._warm_start_variable_names_to_values = {}
        if variable_names_to_warm_start and not self.is_all_outputs_existing(
            variable_names_to_warm_start
        ):
            all_output_names = self.get_output_data_names()
            missing_output_names = set(variable_names_to_warm_start).difference(
                all_output_names
            )
            raise ValueError(
                "The following variable names are not "
                f"outputs of the chain: {missing_output_names}."
                f" Available outputs are: {all_output_names}."
            )

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} cannot be linearized.")

    def _run(self) -> None:
        if self._warm_start_variable_names_to_values:
            self.local_data.update(self._warm_start_variable_names_to_values)
        super()._run()
        if self._variable_names_to_warm_start:
            self._warm_start_variable_names_to_values = {
                name: self.local_data[name]
                for name in self._variable_names_to_warm_start
            }
