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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Chains of disciplines: sequential and parallel execution processes
******************************************************************
"""
from __future__ import division, unicode_literals

import logging
from copy import deepcopy

from numpy import dot, ndarray, zeros

from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.jacobian_assembly import JacobianAssembly
from gemseo.core.parallel_execution import (
    DiscParallelExecution,
    DiscParallelLinearization,
)

LOGGER = logging.getLogger(__name__)


class MDOChain(MDODiscipline):
    """Chain of processes that is based on a predefined order of execution."""

    AVAILABLE_MODES = [
        JacobianAssembly.DIRECT_MODE,
        JacobianAssembly.REVERSE_MODE,
        JacobianAssembly.AUTO_MODE,
    ]

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + ("disciplines",)

    def __init__(
        self, disciplines, name=None, grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE
    ):
        """Constructor of the chain.

        :param disciplines: the disciplines list
        :param name: the name of the discipline
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        """
        super(MDOChain, self).__init__(name, grammar_type=grammar_type)
        self.disciplines = disciplines
        self.initialize_grammars()
        self.default_inputs = {}
        self._update_default_inputs()

    def set_disciplines_statuses(self, status):
        """Sets the sub disciplines statuses.

        :param status: the status
        """
        for discipline in self.disciplines:
            discipline.status = status
            discipline.set_disciplines_statuses(status)

    def initialize_grammars(self):
        """Defines all inputs and outputs of the chain."""
        self.input_grammar.clear()
        self.output_grammar.clear()
        for discipline in self.disciplines:
            self.input_grammar.update_from_if_not_in(
                discipline.input_grammar, self.output_grammar
            )
            self.output_grammar.update_from(discipline.output_grammar)

    def _update_default_inputs(self):
        """Computes the default inputs from the disciplines default inputs."""
        self_inputs = self.get_input_data_names()
        for disc in self.disciplines:
            for key, value in disc.default_inputs.items():
                if key in self_inputs:
                    self.default_inputs[key] = value

    def _run(self):
        """Run discipline."""
        for discipline in self.disciplines:
            outs = discipline.execute(self.local_data)
            self.local_data.update(outs)

    def reverse_chain_rule(self, chain_outputs, discipline):
        """Chains derivatives of self, with a new discipline in the chain in reverse
        mode.

        Performs chain ruling:
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

        :param discipline: new discipline to compose in the chain
        :param chain_outputs: the chain_outputs to linearize
        """
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = discipline.cache.get_last_cached_inputs()
        discipline.linearize(last_cached, force_no_exec=True, force_all=True)

        for output in chain_outputs:
            if output in self.jac:
                # This output has already been taken from previous disciplines
                # Derivatives must be composed using the chain rule

                # Make a copy of the keys because the dict is changed in the
                # loop
                existing_inputs = self.jac[output].keys()
                common_inputs = set(existing_inputs) & set(discipline.jac)
                for input_name in common_inputs:
                    # Store reference to the current Jacobian
                    curr_j = self.jac[output][input_name]
                    for new_in, new_jac in discipline.jac[input_name].items():
                        # Chain rule the derivatives
                        # TODO: sum BEFORE dot
                        loc_dot = dot(curr_j, new_jac)
                        # when input_name==new_in, we are in the case of an
                        # input being also an output
                        # in this case we must only compose the derivatives
                        if new_in in self.jac[output] and input_name != new_in:
                            # The output is already linearized wrt this
                            # input_name. We are in the case:
                            # d o     d o    d o     di_2
                            # ----  = ---- + ----- . -----
                            # d z     d z    d i_2    d z
                            self.jac[output][new_in] += loc_dot
                        else:
                            # The output is not yet linearized wrt this
                            # input_name.  We are in the case:
                            #  d o      d o     di_1   d o     di_2
                            # -----  = ------ . ---- + ----  . ----
                            #  d x      d i_1   d x    d i_2    d x
                            self.jac[output][new_in] = loc_dot

            elif output in discipline.jac:
                # Output of the chain not yet filled in jac,
                # Take the jacobian dict of the current discipline to
                # Initialize. Make a copy !
                self.jac[output] = MDOChain.copy_jacs(discipline.jac[output])

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Actual computation of the jacobians.

        :param inputs: linearization should be performed with respect
            to inputs list.
        :param outputs: linearization should be performed on
            chain_outputs list.
        """

        # Initializes self jac with copy of last discipline (reverse mode)
        last_discipline = self.disciplines[-1]
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = last_discipline.cache.get_last_cached_inputs()
        last_discipline.linearize(last_cached, force_no_exec=True, force_all=True)
        self.jac = self.copy_jacs(last_discipline.jac)

        # reverse mode of remaining disciplines
        remaining_disciplines = self.disciplines[:-1]
        for discipline in remaining_disciplines[::-1]:
            self.reverse_chain_rule(outputs, discipline)

        # Remove differentiations that should not be there,
        # because inputs are not inputs of the chain
        for in_dict in self.jac.values():
            # Copy keys because the dict in changed in the loop
            input_keys_cp = list(in_dict.keys())
            for input_name in input_keys_cp:
                if input_name not in inputs:
                    del in_dict[input_name]

        # Add differentiations that should be there,
        # because inputs inputs of the chain but not
        # of all disciplines
        for out_name, jac_loc in self.jac.items():
            n_outs = len(self.get_outputs_by_name(out_name))
            for input_name in inputs:
                if input_name not in jac_loc:
                    n_inpts = len(self.get_inputs_by_name(input_name))
                    jac_loc[input_name] = zeros((n_outs, n_inpts))

    @staticmethod
    def copy_jacs(jac_dict):
        """Hard copies Jacobian dict.

        :param jac_dict: dict of dict of ndarrays, or dict of ndarrays
        :returns: deepcopy of the input
        """
        out_d = {}

        for outpt, sub_jac in jac_dict.items():
            if isinstance(sub_jac, dict):
                loc_jac_cp = {}
                out_d[outpt] = loc_jac_cp
                for inpt, jac in sub_jac.items():
                    loc_jac_cp[inpt] = jac.copy()
            elif isinstance(sub_jac, ndarray):
                out_d[outpt] = sub_jac.copy()
        return out_d

    def reset_statuses_for_run(self):
        """Sets all the statuses to PENDING."""
        super(MDOChain, self).reset_statuses_for_run()
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def get_expected_workflow(self):
        """Returns the expected execution sequence, used for xdsm representation.

        See MDOFormulation.get_expected_workflow.
        """
        sequence = ExecutionSequenceFactory.serial()
        for disc in self.disciplines:
            sequence.extend(disc.get_expected_workflow())
        return sequence

    def get_expected_dataflow(self):
        """Returns the expected data exchange sequence, used for xdsm representation See
        MDOFormulation.get_expected_dataflow."""
        all_disc = list(set(self.disciplines))
        graph = DependencyGraph(all_disc)
        res = graph.get_disciplines_couplings()

        # Add discipline inner couplings (ex. MDA case)
        for disc in all_disc:
            res.extend(disc.get_expected_dataflow())

        return res

    def _set_cache_tol(self, cache_tol):
        """Sets to the cache input tolerance To be overloaded by subclasses.

        :param cache_tol: float, cache tolerance
        """
        super(MDOChain, self)._set_cache_tol(cache_tol)
        for disc in self.disciplines:
            disc.cache_tol = cache_tol or 0.0


class MDOParallelChain(MDODiscipline):
    """Chain of processes that executes disciplines in parallel."""

    def __init__(
        self,
        disciplines,
        name=None,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        use_threading=True,
        n_processes=None,
    ):
        """Constructor of the chain.

        :param disciplines: the disciplines list
        :param name: the name of the discipline
        :param grammar_type: the type of grammar to use for IO declaration
            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
        :param n_processes: maximum number of processors on which to run,
            by default the number of disciplines
        """
        super(MDOParallelChain, self).__init__(name, grammar_type=grammar_type)
        self.disciplines = disciplines
        self.initialize_grammars()
        self.default_inputs = {}
        self._update_default_inputs()
        if n_processes is None:
            n_processes = len(self.disciplines)
        dpe = DiscParallelExecution(
            self.disciplines, n_processes, use_threading=use_threading
        )
        self.parallel_execution = dpe
        dpl = DiscParallelLinearization(
            self.disciplines, n_processes, use_threading=use_threading
        )
        self.parallel_lin = dpl

    def initialize_grammars(self):
        """Defines all inputs and outputs of the chain."""
        self.input_grammar.clear()
        self.output_grammar.clear()
        for discipline in self.disciplines:
            self.input_grammar.update_from(discipline.input_grammar)
            self.output_grammar.update_from(discipline.output_grammar)

    def _update_default_inputs(self):
        """Computes the default inputs from the disciplines default inputs."""
        self_inputs = self.get_input_data_names()
        for disc in self.disciplines:
            for key, value in disc.default_inputs.items():
                if key in self_inputs:
                    self.default_inputs[key] = value

    def _get_inputs_list(self):
        """Returns a list of inputs dict for parallel execution."""
        n_disc = len(self.disciplines)
        # Avoid overlaps with dicts in // by doing a deepcopy
        # The outputs of a discipline may be a coupling, and shall therefore
        # not be passed as input of another since the execution are assumed
        # to be independent here
        all_inpts = [deepcopy(self.local_data) for _ in range(n_disc)]
        return all_inpts

    def _run(self):
        """Run discipline."""
        all_inpts = self._get_inputs_list()
        self.parallel_execution.execute(all_inpts)

        # Update data according to input order of priority
        for discipline in self.disciplines:
            out_set = discipline.get_output_data_names()
            disc_data = discipline.local_data
            out_dict = {out_k: disc_data[out_k] for out_k in out_set}
            self.local_data.update(out_dict)

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Actual computation of the jacobians.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization
            should be performed wrt all inputs  (Default value = None)
        :param outputs: linearization should be performed on
            chain_outputs list.
            If None, linearization should be
            performed on all chain_outputs (Default value = None)
        """
        self._set_disciplines_diff_outputs(outputs)
        self._set_disciplines_diff_inputs(inputs)
        all_inpts = self._get_inputs_list()
        jacobians = self.parallel_lin.execute(all_inpts)
        self.jac = {}
        # Update jacobians according to input order of priority
        for disc_jacobian in jacobians:
            for out_k, jac_loc in disc_jacobian.items():
                self_jac_o = self.jac.get(out_k)
                if self_jac_o is None:
                    self_jac_o = {}
                    self.jac[out_k] = self_jac_o
                self_jac_o.update(jac_loc)
        self._init_jacobian(inputs, outputs, with_zeros=True, fill_missing_keys=True)

    def add_differentiated_inputs(self, inputs=None):
        MDODiscipline.add_differentiated_inputs(self, inputs)
        self._set_disciplines_diff_inputs(inputs)

    def _set_disciplines_diff_inputs(self, inputs):
        """Adds inputs to the right sub discipline's differentiated inputs.

        :param inputs: the inputs list
        """
        diff_inpts = set(inputs)
        for disc in self.disciplines:
            inpt_set = set(disc.get_input_data_names()) & diff_inpts
            if inpt_set:
                disc.add_differentiated_inputs(list(inpt_set))

    def add_differentiated_outputs(self, outputs=None):
        MDODiscipline.add_differentiated_outputs(self, outputs)
        self._set_disciplines_diff_outputs(outputs)

    def _set_disciplines_diff_outputs(self, outputs):
        """Adds outputs to the right sub discipline's differentiated outputs.

        :param outputs: the outputs list
        """
        diff_outpts = set(outputs)
        for disc in self.disciplines:
            outpt_set = set(disc.get_output_data_names()) & diff_outpts
            if outpt_set:
                disc.add_differentiated_outputs(list(outpt_set))

    def reset_statuses_for_run(self):
        """Sets all the statuses to PENDING."""
        super(MDOParallelChain, self).reset_statuses_for_run()
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def get_expected_workflow(self):
        """Returns the expected execution sequence, used for xdsm representation.

        See MDOFormulation.get_expected_workflow.
        """
        sequence = ExecutionSequenceFactory.parallel()
        for disc in self.disciplines:
            sequence.extend(disc.get_expected_workflow())
        return sequence

    def get_expected_dataflow(self):
        """Returns the expected data exchange sequence, used for xdsm representation See
        MDOFormulation.get_expected_dataflow."""
        return []

    def _set_cache_tol(self, cache_tol):
        """Sets to the cache input tolerance To be overloaded by subclasses.

        :param cache_tol: float, cache tolerance
        """
        super(MDOParallelChain, self)._set_cache_tol(cache_tol)
        for disc in self.disciplines:
            disc.cache_tol = cache_tol or 0.0


class MDOAdditiveChain(MDOParallelChain):
    """Chain of processes that executes disciplines in parallel and sums specified
    outputs across disciplines."""

    def __init__(
        self,
        disciplines,
        outputs_to_sum,
        name=None,
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,
        use_threading=True,
        n_processes=None,
    ):
        """Constructor.

        :param disciplines: the disciplines list
        :type disciplines: list(MDODiscipline)
        :param outputs_to_sum: names list of the outputs to sum
        :type outputs_to_sum: list(str)
        :param name: name of the discipline
        :type name: str, optional
        :param grammar_type: the type of grammar to use for IO declaration
                            either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE
        :type grammar_type: str, optional
        :param use_threading: if True, use Threads instead of processes
            to parallelize the execution
        :type use_threading: bool, optional
        :param n_processes: maximum number of processors on which to run,
            by default the number of disciplines
        :type n_processes: int, optional
        """
        super(MDOAdditiveChain, self).__init__(
            disciplines, name, grammar_type, use_threading, n_processes
        )
        self._outputs_to_sum = outputs_to_sum

    def _run(self):
        """Runs the disciplines and computes the sum."""
        # Run the disciplines in parallel
        MDOParallelChain._run(self)

        # Sum the required outputs across disciplines
        for out_name in self._outputs_to_sum:
            terms = [
                disc.local_data[out_name]
                for disc in self.disciplines
                if out_name in disc.local_data
            ]
            sum_value = sum(terms) if terms else None
            self.local_data[out_name] = sum_value

    def _compute_jacobian(self, inputs=None, outputs=None):
        """Actual computation of the Jacobians.

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization
            should be performed wrt all inputs  (Default value = None)
        :param outputs: linearization should be performed on
            chain_outputs list.
            If None, linearization should be
            performed on all chain_outputs (Default value = None)
        """
        # Differentiate the disciplines in parallel
        MDOParallelChain._compute_jacobian(self, inputs, outputs)

        # Sum the Jacobians of the required outputs across disciplines
        for out_name in self._outputs_to_sum:
            self.jac[out_name] = dict()
            for in_name in inputs:
                terms = [
                    disc.jac[out_name][in_name]
                    for disc in self.disciplines
                    if in_name in disc.jac[out_name]
                ]

                assert terms
                sum_value = sum(terms)
                self.jac[out_name][in_name] = sum_value
