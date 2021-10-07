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
#    INITIAL AUTHORS - initial API and implementation and/or
#                        initial documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Individual Discipline Feasible (IDF) formulation."""
from __future__ import division, unicode_literals

import logging
from typing import Iterable, List, Sequence, Tuple

from numpy import abs as np_abs
from numpy import concatenate, eye, ndarray, ones_like, zeros

from gemseo.algos.design_space import DesignSpace
from gemseo.core.chain import MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequence, ExecutionSequenceFactory
from gemseo.core.formulation import MDOFormulation
from gemseo.core.mdofunctions.consistency_constraint import ConsistencyCstr
from gemseo.core.mdofunctions.function_from_discipline import FunctionFromDiscipline
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.mda.mda_chain import MDAChain

LOGGER = logging.getLogger(__name__)


class IDF(MDOFormulation):
    """The Individual Discipline Feasible (IDF) formulation.

    This formulation draws an optimization architecture
    where the coupling variables of strongly coupled disciplines is made consistent
    by adding equality constraints on the coupling variables at top level,
    the optimization problem
    with respect to the local, global design variables and coupling variables
    is made at the top level.

    The disciplinary analysis is made at a each optimization iteration
    while the multidisciplinary analysis is made at the optimum.
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        objective_name,  # type: str
        design_space,  # type: DesignSpace
        maximize_objective=False,  # type: bool
        normalize_constraints=True,  # type: bool
        parallel_exec=False,  # type: bool
        use_threading=True,  # type: bool
        start_at_equilibrium=False,  # type: bool
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
    ):  # type: (...) -> None
        """
        Args:
            normalize_constraints: If True,
                the outputs of the coupling consistency constraints are scaled.
            parallel_exec: If True,
                all constraints and objectives are computed in parallel.
                At every iteration,
                all disciplines are executed in parallel.
                Otherwise,
                a separate constraint is created for each discipline with couplings.
            use_threading: If True and parallel_exec=True,
                the disciplines are run in parallel using multi-threading.
                If False and parallel_exec=True, multi-processing is used.
            start_at_equilibrium: If True,
                an MDA is used to initialize the coupling variables.
        """
        super(IDF, self).__init__(
            disciplines,
            objective_name,
            design_space,
            maximize_objective=maximize_objective,
            grammar_type=grammar_type,
        )
        if parallel_exec:
            self._parallel_exec = MDOParallelChain(
                self.disciplines, use_threading=use_threading, grammar_type=grammar_type
            )
        else:
            self._parallel_exec = None

        self.coupling_structure = MDOCouplingStructure(disciplines)
        self.all_couplings = self.coupling_structure.get_all_couplings()
        self._update_design_space()
        self.normalize_constraints = normalize_constraints
        self._build_constraints()
        self._build_objective_from_disc(objective_name)

        if start_at_equilibrium:
            self._compute_equilibrium()

    def _compute_equilibrium(self):  # type: (...) -> None
        """Run an MDA to compute the initial target couplings at equilibrium.

        The values at equilibrium are set in the initial design space.
        """
        current_x = self.design_space.get_current_x_dict()
        # run MDA to initialize target coupling variables
        mda = MDAChain(self.disciplines)
        res = mda.execute(current_x)

        for name in self.all_couplings:
            value = res[name]
            LOGGER.info(
                "IDF: changing the initial value of %s " "from %s to %s (equilibrium)",
                name,
                str(current_x[name]),
                str(value),
            )
            self.design_space.set_current_variable(name, value)

    def _update_design_space(self):
        """Update the design space with the required variables."""
        strong_couplings = set(self.all_couplings)
        variables_names = set(self.opt_problem.design_space.variables_names)
        if not strong_couplings.issubset(variables_names):
            missing = strong_couplings - variables_names
            raise ValueError(
                "IDF formulation needs coupling variables as design variables, "
                "missing variables: %s" % missing
            )
        self._set_defaultinputs_from_ds()

    def get_top_level_disc(self):  # type: (...) -> List[MDODiscipline]
        # All functions and constraints are built from the top level disc
        # If we are in parallel mode: return the parallel execution
        if self._parallel_exec is not None:
            return [self._parallel_exec]
        # Otherwise the disciplines are top level
        return self.disciplines

    def _get_normalization_factor(
        self,
        output_couplings,  # type:Iterable[str]
    ):  # type: (...) -> ndarray
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

    def _generate_consistency_cstr(
        self,
        output_couplings,  # type: Sequence[str]
    ):  # type: (...) -> MDOFunction
        """Generate the consistency constraints for a discipline.

        Args:
            output_couplings: The names of the output couplings.

        Returns:
            A function computing the consistency constraints.
        """
        coupl_func = FunctionFromDiscipline(output_couplings, self)
        dv_names_of_disc = coupl_func.args

        if self.normalize_constraints:
            norm_fact = self._get_normalization_factor(output_couplings)
        else:
            norm_fact = 1.0

        def coupl_min_x(
            x_vec,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Function to compute the consistency constraints.

            Args:
                x_vect: The design variable vector.

            Returns:
                The value of the consistency constraints.
                Equal to zero if the disciplines are at equilibrium.
            """
            x_sw = self.mask_x_swap_order(output_couplings, x_vec)
            coupl = coupl_func(x_vec)
            if self.normalize_constraints:
                return (coupl - x_sw) / norm_fact
            return coupl - x_sw

        def coupl_min_x_jac(
            x_vec,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Function to compute the gradient of the consistency constraints.

            Args:
                x_vect: The design variable vector.

            Returns:
                The value of the gradient of the consistency constraints.
            """
            coupl_jac = coupl_func.jac(x_vec)  # pylint: disable=E1102

            if len(coupl_jac.shape) > 1:
                # IN this case it is harder since a block diagonal
                # matrix with -Id should be placed for each output
                # coupling, at the right place
                n_outs = coupl_jac.shape[0]
                x_jac_2d = zeros((n_outs, len(x_vec)), dtype=x_vec.dtype)
                x_names = self.get_optim_variables_names()
                o_min = 0
                o_max = 0
                for out in output_couplings:
                    # self._reference_input_data[out].size
                    o_len = self._get_dv_length(out)
                    i_min = 0
                    i_max = 0
                    o_max += o_len
                    for x_i in x_names:
                        # self._reference_input_data[x_i].size
                        x_len = self._get_dv_length(x_i)
                        i_max += x_len
                        if x_i == out:
                            x_jac_2d[o_min:o_max, i_min:i_max] = eye(x_len)
                        i_min = i_max
                    o_min = o_max
                x_jac = x_jac_2d
            else:
                # This is surprising but there is a duality between the masking
                # operation in the function inputs and the unmasking of its
                # outputs
                x_jac = self.unmask_x_swap_order(output_couplings, ones_like(x_vec))
            if self.normalize_constraints:
                return (coupl_jac - x_jac) / norm_fact[:, None]
            return coupl_jac - x_jac

        expr = ""
        for out_c in output_couplings:
            expr += out_c + "(" + ", ".join(dv_names_of_disc) + ") - "
            expr += str(out_c) + "" + "\n"

        name = coupl_func.name
        return MDOFunction(
            coupl_min_x,
            name,
            args=dv_names_of_disc,
            expr=expr,
            jac=coupl_min_x_jac,
            outvars=coupl_func.outvars,
            f_type=MDOFunction.TYPE_EQ,
        )

    def _build_constraints(self):  # type: (...) -> None
        """Build the constraints.

        In IDF formulation,
        the consistency constraints are "y - y_copy = 0".
        """
        # Building constraints per generator couplings
        for discipline in self.disciplines:
            couplings = self.coupling_structure.output_couplings(
                discipline, strong=False
            )
            if couplings:
                cstr = ConsistencyCstr(couplings, self)
                self.opt_problem.add_eq_constraint(cstr)

    def get_expected_workflow(
        self,
    ):  # type: (...) -> List[ExecutionSequence,Tuple[ExecutionSequence]]
        return ExecutionSequenceFactory.parallel(self.disciplines)

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        return []
