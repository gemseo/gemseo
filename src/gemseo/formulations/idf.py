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
#    INITIAL AUTHORS - initial API and implementation and/or
#                        initial documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Individual Discipline Feasible (IDF) formulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from numpy import abs as np_abs
from numpy import concatenate
from numpy import ndarray
from numpy import zeros

from gemseo.core.chain import MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.consistency_constraint import ConsistencyCstr
from gemseo.core.mdofunctions.taylor_polynomials import compute_linear_approximation
from gemseo.mda.mda_chain import MDAChain

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.algos.design_space import DesignSpace

LOGGER = logging.getLogger(__name__)


class IDF(MDOFormulation):
    """The Individual Discipline Feasible (IDF) formulation.

    This formulation draws an optimization architecture where the coupling variables of
    strongly coupled disciplines is made consistent by adding equality constraints on
    the coupling variables at top level, the optimization problem with respect to the
    local, global design variables and coupling variables is made at the top level.

    The disciplinary analysis is made at each optimization iteration while the
    multidisciplinary analysis is made at the optimum.
    """

    def __init__(
        self,
        disciplines: list[MDODiscipline],
        objective_name: str,
        design_space: DesignSpace,
        maximize_objective: bool = False,
        normalize_constraints: bool = True,
        n_processes: int = 1,
        use_threading: bool = True,
        start_at_equilibrium: bool = False,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
    ) -> None:
        """
        Args:
            normalize_constraints: If ``True``,
                the outputs of the coupling consistency constraints are scaled.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            use_threading: Whether to use threads instead of processes
                to parallelize the execution;
                multiprocessing will copy (serialize) all the disciplines,
                while threading will share all the memory.
                This is important to note
                if you want to execute the same discipline multiple times,
                you shall use multiprocessing.
            start_at_equilibrium: If ``True``,
                an MDA is used to initialize the coupling variables.
        """  # noqa: D205, D212, D415
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        if n_processes > 1:
            LOGGER.info(
                "Running IDF formulation in parallel on n_processes = %s",
                n_processes,
            )
            self._parallel_exec = MDOParallelChain(
                self.disciplines,
                use_threading=use_threading,
                grammar_type=grammar_type,
                n_processes=n_processes,
            )
        else:
            self._parallel_exec = None

        self.coupling_structure = MDOCouplingStructure(disciplines)
        self.all_couplings = self.coupling_structure.all_couplings
        self._update_design_space()
        self.normalize_constraints = normalize_constraints
        self._build_constraints()
        self._build_objective_from_disc(objective_name)

        if start_at_equilibrium:
            self._compute_equilibrium()

    def _compute_equilibrium(self) -> None:
        """Run an MDA to compute the initial target couplings at equilibrium.

        The values at equilibrium are set in the initial design space.
        """
        current_x = self.design_space.get_current_value(as_dict=True)
        # run MDA to initialize target coupling variables
        mda = MDAChain(self.disciplines)
        res = mda.execute(current_x)

        for name in self.all_couplings:
            value = res[name]
            LOGGER.info(
                "IDF: changing the initial value of %s from %s to %s (equilibrium)",
                name,
                current_x[name],
                value,
            )
            self.design_space.set_current_variable(name, value)

    def _update_design_space(self) -> None:
        """Update the design space with the required variables."""
        strong_couplings = set(self.all_couplings)
        variable_names = set(self.opt_problem.design_space.variable_names)
        if not strong_couplings.issubset(variable_names):
            missing = strong_couplings - variable_names
            raise ValueError(
                "IDF formulation needs coupling variables as design variables, "
                f"missing variables: {missing}."
            )
        self._set_default_input_values_from_design_space()

    def get_top_level_disc(self) -> list[MDODiscipline]:  # noqa:D102
        # All functions and constraints are built from the top level disc
        # If we are in parallel mode: return the parallel execution
        if self._parallel_exec is not None:
            return [self._parallel_exec]
        # Otherwise the disciplines are top level
        return self.disciplines

    def _get_normalization_factor(
        self,
        output_couplings: Iterable[str],
    ) -> ndarray:
        """Compute [abs(ub-lb)] for all output couplings.

        Args:
            output_couplings: The names of the variables for normalization.

        Returns:
            The concatenation of the normalization factors for all output couplings.
        """
        norm_fact = []
        for output in output_couplings:
            u_b = self.design_space.get_upper_bound(output)
            l_b = self.design_space.get_lower_bound(output)
            norm_fact.append(np_abs(u_b - l_b))
        return concatenate(norm_fact)

    def _build_constraints(self) -> None:
        """Build the constraints.

        In IDF formulation,
        the consistency constraints are "y - y_copy = 0".
        """
        # Building constraints per generator couplings
        for discipline in self.disciplines:
            couplings = self.coupling_structure.get_output_couplings(
                discipline, strong=False
            )
            if couplings:
                cstr = ConsistencyCstr(couplings, self)
                if cstr.linear_candidate:
                    cstr = compute_linear_approximation(
                        cstr, zeros(cstr.input_dimension)
                    )
                self.opt_problem.add_eq_constraint(cstr)

    def get_expected_workflow(  # noqa:D102
        self,
    ) -> list[ExecutionSequence, tuple[ExecutionSequence]]:
        return ExecutionSequenceFactory.parallel(self.disciplines)

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return []
