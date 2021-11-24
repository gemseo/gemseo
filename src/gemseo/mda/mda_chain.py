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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""An advanced MDA splitting algorithm based on graphs."""
from __future__ import division, unicode_literals

import logging
from itertools import repeat
from multiprocessing import cpu_count
from os.path import join, split
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from gemseo.api import create_mda
from gemseo.core.chain import MDOChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.mda.mda import MDA

LOGGER = logging.getLogger(__name__)
N_CPUS = cpu_count()


class MDAChain(MDA):
    """A chain of sub-MDAs.

    The execution sequence is provided by the :class:`.DependencyGraph`.
    """

    _ATTR_TO_SERIALIZE = MDA._ATTR_TO_SERIALIZE + (
        "mdo_chain",
        "_chain_linearize",
        "lin_cache_tol_fact",
        "assembly",
        "coupling_structure",
        "linear_solver",
        "linear_solver_options",
        "linear_solver_tolerance",
        "matrix_type",
        "use_lu_fact",
        "all_couplings",
    )

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        sub_mda_class="MDAJacobi",  # type: str
        max_mda_iter=20,  # type: int
        name=None,  # type: Optional[str]
        n_processes=N_CPUS,  # type: int
        chain_linearize=False,  # type: bool
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        use_lu_fact=False,  # type: bool
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        sub_coupling_structures=None,  # type: Optional[Iterable[MDOCouplingStructure]]
        log_convergence=False,  # type: bool
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
        **sub_mda_options  # type: Optional[Union[float, int, bool, str]]
    ):
        """
        Args:
            sub_mda_class: The class name of the sub-MDA.
            n_processes: The number of processes.
            chain_linearize: Whether to linearize the chain of execution.
                Otherwise, linearize the overall MDA with base class method.
                This last option is preferred to minimize computations in adjoint mode,
                while in direct mode, linearizing the chain may be cheaper.
            sub_coupling_structures: The coupling structures to be used by the sub-MDAs.
                If None, they are created from the sub-disciplines.
            **sub_mda_options: The options to be passed to the sub-MDAs.
        """
        self.n_processes = n_processes
        self.mdo_chain = None
        self._chain_linearize = chain_linearize
        self.sub_mda_list = []

        # compute execution sequence of the disciplines
        super(MDAChain, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            use_lu_fact=use_lu_fact,
            grammar_type=grammar_type,
            coupling_structure=coupling_structure,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
        )

        if (
            not self.coupling_structure.get_all_couplings()
            and not self._chain_linearize
        ):
            LOGGER.warning("No coupling in MDA, switching chain_linearize to True")
            self._chain_linearize = True

        self._create_mdo_chain(
            disciplines,
            sub_mda_class=sub_mda_class,
            sub_coupling_structures=sub_coupling_structures,
            **sub_mda_options
        )

        self.log_convergence = log_convergence

        self._initialize_grammars()
        self._check_consistency()
        self._set_default_inputs()
        self._compute_input_couplings()

        # cascade the tolerance
        for sub_mda in self.sub_mda_list:
            sub_mda.tolerance = self.tolerance

    @MDA.log_convergence.setter
    def log_convergence(
        self,
        value,  # type: bool
    ):  # type: (...) -> None
        self._log_convergence = value
        for mda in self.sub_mda_list:
            mda.log_convergence = value

    def _create_mdo_chain(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        sub_mda_class="MDAJacobi",  # type: str
        sub_coupling_structures=None,  # type: Optional[Iterable[MDOCouplingStructure]]
        **sub_mda_options  # type: Optional[Union[float,int,bool,str]]
    ):
        """Create an MDO chain from the execution sequence of the disciplines.

        Args:
            sub_mda_class: The name of the class of the sub-MDAs.
            disciplines: The disciplines.
            sub_coupling_structures: The coupling structures to be used by the sub-MDAs.
                If None, they are created from the sub-disciplines.
            **sub_mda_options: The options to be used to initialize the sub-MDAs.
        """
        chained_disciplines = []
        self.sub_mda_list = []

        if sub_coupling_structures is None:
            sub_coupling_structures = repeat(None)

        sub_coupling_structures_iterator = iter(sub_coupling_structures)

        for parallel_tasks in self.coupling_structure.sequence:
            # to parallelize, check if 1 < len(parallel_tasks)
            # for now, parallel tasks are run sequentially
            for coupled_disciplines in parallel_tasks:
                first_disc = coupled_disciplines[0]
                if len(coupled_disciplines) > 1 or (
                    len(coupled_disciplines) == 1
                    and self.coupling_structure.is_self_coupled(first_disc)
                ):
                    # several disciplines coupled

                    # order the MDA disciplines the same way as the
                    # original disciplines
                    sub_mda_disciplines = []
                    for disc in disciplines:
                        if disc in coupled_disciplines:
                            sub_mda_disciplines.append(disc)

                    # create a sub-MDA
                    sub_mda = create_mda(
                        sub_mda_class,
                        sub_mda_disciplines,
                        max_mda_iter=self.max_mda_iter,
                        tolerance=self.tolerance,
                        linear_solver_tolerance=self.linear_solver_tolerance,
                        grammar_type=self.grammar_type,
                        use_lu_fact=self.use_lu_fact,
                        linear_solver=self.linear_solver,
                        linear_solver_options=self.linear_solver_options,
                        coupling_structure=next(sub_coupling_structures_iterator),
                        **sub_mda_options
                    )
                    sub_mda.n_processes = self.n_processes

                    chained_disciplines.append(sub_mda)
                    self.sub_mda_list.append(sub_mda)
                else:
                    # single discipline
                    chained_disciplines.append(first_disc)

        # create the MDO chain that sequentially evaluates the sub-MDAs and the
        # single disciplines
        self.mdo_chain = MDOChain(
            chained_disciplines, name="MDA chain", grammar_type=self.grammar_type
        )

    def _initialize_grammars(self):  # type: (...) -> None
        """Define all inputs and outputs of the chain."""
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        self.input_grammar.update_from(self.mdo_chain.input_grammar)
        self.output_grammar.update_from(self.mdo_chain.output_grammar)

    def _check_consistency(self):
        """Check if there is no more than 1 equation per variable.

        For instance if a strong coupling is not also a self coupling.
        """
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        super(MDAChain, self)._check_consistency()

    def _run(self):  # type -> None
        if self.warm_start:
            self._couplings_warm_start()
        self.local_data = self.mdo_chain.execute(self.local_data)
        return self.local_data

    def _compute_jacobian(
        self,
        inputs=None,  # type: Optional[Sequence[str]]
        outputs=None,  # type: Optional[Sequence[str]]
    ):  # type: (...) -> None
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)
            self.mdo_chain.add_differentiated_outputs(outputs)
            # the Jacobian of the MDA chain is the Jacobian of the MDO chain
            last_cached = self.cache.get_last_cached_inputs()
            self.mdo_chain.linearize(last_cached)
            self.jac = self.mdo_chain.jac
        else:
            super(MDAChain, self)._compute_jacobian(inputs, outputs)

    def add_differentiated_inputs(
        self,
        inputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        MDA.add_differentiated_inputs(self, inputs)
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)

    def add_differentiated_outputs(
        self,
        outputs=None,  # type: Optional[Iterable[str]]
    ):  # type: (...) -> None
        MDA.add_differentiated_outputs(self, outputs=outputs)
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_outputs(outputs)

    @property
    def normed_residual(self):  # type: (...) -> float
        """The normed_residuals, computed from the sub-MDAs residuals."""
        return sum((mda.normed_residual ** 2 for mda in self.sub_mda_list)) ** 0.5

    @normed_residual.setter
    def normed_residual(
        self,
        normed_residual,  # type: float
    ):  # type: (...) ->None
        """Set the normed_residual.

        Has no effect,
        since the normed residuals are defined by sub-MDAs residuals
        (see associated property).

        Here for compatibility with mother class.
        """
        pass

    def get_expected_dataflow(
        self,
    ):  # type: (...) -> List[Tuple[MDODiscipline,MDODiscipline,List[str]]]
        return self.mdo_chain.get_expected_dataflow()

    def get_expected_workflow(self):  # type: (...) ->SerialExecSequence
        exec_s = SerialExecSequence(self)
        workflow = self.mdo_chain.get_expected_workflow()
        exec_s.extend(workflow)
        return exec_s

    def reset_statuses_for_run(self):  # type: (...) -> None
        super(MDAChain, self).reset_statuses_for_run()
        self.mdo_chain.reset_statuses_for_run()

    def plot_residual_history(
        self,
        show=False,  # type: bool
        save=True,  # type: bool
        n_iterations=None,  # type: Optional[int]
        logscale=None,  # type: Optional[Tuple[int,int]]
        filename=None,  # type: Optional[str]
        figsize=(50, 10),  # type: Tuple[int,int]
    ):  # type: (...) -> None
        for sub_mda in self.sub_mda_list:
            if filename is not None:
                s_filename = split(filename)
                filename = join(
                    s_filename[0],
                    "{}_{}".format(sub_mda.__class__.__name__, s_filename[1]),
                )
            sub_mda.plot_residual_history(
                show, save, n_iterations, logscale, filename, figsize
            )
