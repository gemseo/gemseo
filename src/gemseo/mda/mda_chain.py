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
#        :author: Jean-Christophe Giret
"""An advanced MDA splitting algorithm based on graphs."""
from __future__ import annotations

import logging
from itertools import repeat
from multiprocessing import cpu_count
from os.path import join
from os.path import split
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence

from numpy import array

from gemseo.api import create_mda
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.mda.mda import MDA

LOGGER = logging.getLogger(__name__)
N_CPUS = cpu_count()


class MDAChain(MDA):
    """A chain of MDAs.

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
        "inner_mdas",
    )

    inner_mdas: list[MDA]
    """The ordered MDAs."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        inner_mda_name: str = "MDAJacobi",
        max_mda_iter: int = 20,
        name: str | None = None,
        n_processes: int = N_CPUS,
        chain_linearize: bool = False,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        use_lu_fact: bool = False,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        coupling_structure: MDOCouplingStructure | None = None,
        sub_coupling_structures: Iterable[MDOCouplingStructure] | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] = None,
        mdachain_parallelize_tasks: bool = False,
        mdachain_parallel_options: Mapping[str, int | bool] | None = None,
        **inner_mda_options: float | int | bool | str | None,
    ):
        """
        Args:
            inner_mda_name: The class name of the inner-MDA.
            n_processes: The maximum simultaneous number of threads,
                if ``use_threading`` is True, or processes otherwise,
                used to parallelize the execution.
            chain_linearize: Whether to linearize the chain of execution.
                Otherwise, linearize the overall MDA with base class method.
                This last option is preferred to minimize computations in adjoint mode,
                while in direct mode, linearizing the chain may be cheaper.
            sub_coupling_structures: The coupling structures to be used by the
                inner-MDAs.
                If None, they are created from the sub-disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the
                parallel tasks, if any.
            mdachain_parallel_options: The options of the MDOParallelChain
                instances, if any.
            **inner_mda_options: The options of the inner-MDAs.
        """
        self.n_processes = n_processes
        self.mdo_chain = None
        self._chain_linearize = chain_linearize
        self.inner_mdas = []

        # compute execution sequence of the disciplines
        super().__init__(
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

        if not self.coupling_structure.all_couplings and not self._chain_linearize:
            LOGGER.warning("No coupling in MDA, switching chain_linearize to True.")
            self._chain_linearize = True

        self._create_mdo_chain(
            disciplines,
            inner_mda_name=inner_mda_name,
            sub_coupling_structures=sub_coupling_structures,
            mdachain_parallelize_tasks=mdachain_parallelize_tasks,
            mdachain_parallel_options=mdachain_parallel_options,
            **inner_mda_options,
        )

        self.log_convergence = log_convergence

        self._initialize_grammars()
        self._check_consistency()
        self._set_default_inputs()
        self._compute_input_couplings()

        # cascade the tolerance
        for mda in self.inner_mdas:
            mda.tolerance = self.tolerance

    @MDA.log_convergence.setter
    def log_convergence(
        self,
        value: bool,
    ) -> None:
        self._log_convergence = value
        for mda in self.inner_mdas:
            mda.log_convergence = value

    def _create_mdo_chain(
        self,
        disciplines: Sequence[MDODiscipline],
        inner_mda_name: str = "MDAJacobi",
        sub_coupling_structures: Iterable[MDOCouplingStructure] | None = None,
        mdachain_parallelize_tasks: bool = False,
        mdachain_parallel_options: Mapping[str, int | bool | None] = None,
        **inner_mda_options: float | int | bool | str | None,
    ):
        """Create an MDO chain from the execution sequence of the disciplines.

        Args:
            inner_mda_name: The name of the class of the inner-MDAs.
            disciplines: The disciplines.
            sub_coupling_structures: The coupling structures to be used by the
                inner-MDAs.
                If None, they are created from the sub-disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the
                parallel tasks,
                if any.
            mdachain_parallel_options: The options of the MDOParallelChain instances,
                if any.
            **inner_mda_options: The options of the inner-MDAs.
        """
        if sub_coupling_structures is None:
            sub_coupling_structures = repeat(None)
        self.__sub_coupling_structures_iterator = iter(sub_coupling_structures)

        chained_disciplines = []
        for parallel_tasks in self.coupling_structure.sequence:
            process = self.__create_process_from_disciplines(
                disciplines,
                inner_mda_name,
                inner_mda_options,
                mdachain_parallel_options,
                parallel_tasks,
                mdachain_parallelize_tasks,
            )
            chained_disciplines.append(process)

        self.mdo_chain = MDOChain(
            chained_disciplines, name="MDA chain", grammar_type=self.grammar_type
        )

    def __create_process_from_disciplines(
        self,
        disciplines: Sequence[MDODiscipline],
        inner_mda_name: str,
        inner_mda_options: Mapping[str, str | float | int],
        mdachain_parallel_options: Mapping[str, str | float | int],
        parallel_tasks: list[tuple[MDODiscipline]],
        mdachain_parallelize_tasks: bool,
    ) -> MDODiscipline:
        """Create a process from disciplines.

        This method creates a process that will be appended to the main inner
        :class:`.MDOChain` of the :class:`.MDAChain`.
        Depending on the number and type of disciplines provided,
        as well as the options provided by the user,
        the process may be a sole discipline,
        a :class:`.MDA`,
        a :class:`MDOChain`,
        or a :class:`MDOParallelChain`.

        Args:
            disciplines: The disciplines.
            inner_mda_name: The inner :class:`.MDA` class name.
            inner_mda_options: The inner :class:`.MDA` options.
            mdachain_parallel_options: The :class:`MDOParallelChain` options.
            parallel_tasks: The parallel tasks to be processed.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks,
                if any.

        Returns:
            A process.
        """
        parallel_disciplines = self.__compute_parallel_disciplines(
            disciplines,
            parallel_tasks,
            inner_mda_name,
            inner_mda_options,
        )

        return self.__create_process_from_parallel_disciplines(
            parallel_disciplines,
            mdachain_parallelize_tasks,
            mdachain_parallel_options,
        )

    def __compute_parallel_disciplines(
        self,
        disciplines: Sequence[MDODiscipline],
        parallel_tasks: list[tuple[MDODiscipline]],
        inner_mda_name: str,
        inner_mda_options: dict[str, dict[str, int | float | str]],
    ) -> Sequence[MDODiscipline | MDA]:
        """Compute the parallel disciplines.

        This method computes the parallel disciplines,
        if any.
        If there is any coupled disciplines in a parallel task,
        a :class:`.MDA` is created,
        based on the :class:`.MDA` options provided.

        Args:
            disciplines: The disciplines.
            parallel_tasks: The parallel tasks.
            inner_mda_name: The inner :class:`.MDA` class name.
            inner_mda_options: The inner :class:`.MDA` options.

        Returns:
            The parallel disciplines.
        """
        parallel_disciplines = []
        for coupled_disciplines in parallel_tasks:
            is_one_discipline_self_coupled = self.__is_one_discipline_self_coupled(
                coupled_disciplines
            )
            if len(coupled_disciplines) > 1 or is_one_discipline_self_coupled:
                discipline = self.__create_inner_mda(
                    disciplines,
                    coupled_disciplines,
                    inner_mda_name,
                    inner_mda_options,
                )
                self.inner_mdas.append(discipline)
            else:
                discipline = coupled_disciplines[0]

            parallel_disciplines.append(discipline)
        return parallel_disciplines

    def __create_process_from_parallel_disciplines(
        self,
        parallel_disciplines: Sequence[MDODiscipline],
        mdachain_parallelize_tasks: bool,
        mdachain_parallel_options: dict[str, int | float | str] | None,
    ) -> MDODiscipline | MDOChain | MDOParallelChain:
        """Create a process from parallel disciplines.

        Depending on the number of disciplines and the options provided,
        the returned GEMSEO process can be a sole :class:`.MDODiscipline` instance,
        a :class:`.MDOChain` or a :class:`.MDOParallelChain`.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks.
            mdachain_parallel_options: The options of the :class:`.MDOParallelChain`.

        Returns:
            A GEMSEO process instance.
        """
        if len(parallel_disciplines) > 1:
            process = self.__create_sequential_or_parallel_chain(
                parallel_disciplines,
                mdachain_parallelize_tasks,
                mdachain_parallel_options,
            )
        else:
            process = parallel_disciplines[0]
        return process

    def __create_inner_mda(
        self,
        disciplines: Sequence[MDODiscipline],
        coupled_disciplines: Sequence[MDODiscipline],
        inner_mda_name: str,
        inner_mda_options: dict[str, dict[str, int | float | str]],
    ) -> MDA:
        """Create an inner MDA from the coupled disciplines and the MDA options.

        Args:
            disciplines: The disciplines.
            coupled_disciplines: The coupled disciplines.
            inner_mda_name: The inner :class:`.MDA` class name.
            inner_mda_options: The inner :class:`.MDA` options.

        Returns:
            The :class:`.MDA` instance.
        """
        inner_mda_disciplines = self.__get_coupled_disciplines_initial_order(
            coupled_disciplines, disciplines
        )
        mda = create_mda(
            inner_mda_name,
            inner_mda_disciplines,
            max_mda_iter=self.max_mda_iter,
            tolerance=self.tolerance,
            linear_solver_tolerance=self.linear_solver_tolerance,
            grammar_type=self.grammar_type,
            use_lu_fact=self.use_lu_fact,
            linear_solver=self.linear_solver,
            linear_solver_options=self.linear_solver_options,
            coupling_structure=next(self.__sub_coupling_structures_iterator),
            **inner_mda_options,
        )
        mda.n_processes = self.n_processes
        return mda

    def __is_one_discipline_self_coupled(
        self, disciplines: Sequence[MDODiscipline]
    ) -> bool:
        """Whether the disciplines contain only one self-coupled discipline which is also
        not a MDA.

        Args:
            disciplines: The disciplines.

        Returns:
            True if the sole discipline of coupled_disciplines is self-coupled
            and not a MDA.
        """
        first_discipline = disciplines[0]
        is_one_discipline_self_coupled = (
            len(disciplines) == 1
            and self.coupling_structure.is_self_coupled(first_discipline)
            and not isinstance(disciplines[0], MDA)
        )
        return is_one_discipline_self_coupled

    @staticmethod
    def __get_coupled_disciplines_initial_order(
        coupled_disciplines: Sequence[MDODiscipline],
        disciplines: Sequence[MDODiscipline],
    ) -> list[MDODiscipline]:
        """Get the coupled disciplines in the same order as initially given by the user.

        Args:
            coupled_disciplines: The coupled disciplines.
            disciplines: The disciplines.

        Returns:
            The ordered list of coupled disciplines.
        """
        return [disc for disc in disciplines if disc in coupled_disciplines]

    def __create_sequential_or_parallel_chain(
        self,
        parallel_disciplines: Sequence[MDODiscipline],
        mdachain_parallelize_tasks: bool,
        mdachain_parallel_options: dict[str, int | float | str] | None,
    ) -> MDOChain | MDOParallelChain:
        """Create a :class:`.MDOChain' or :class:`.MDOParallelChain`.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks,
                if any.
            mdachain_parallel_options: The :class:`MDOParallelChain options.

        Returns:
            Either a :class:`.MDOChain` or :class:`.MDOParallelChain instance.
        """
        if mdachain_parallelize_tasks:
            process = self.__create_mdo_parallel_chain(
                parallel_disciplines,
                mdachain_parallel_options,
            )
        else:
            process = MDOChain(parallel_disciplines, grammar_type=self.grammar_type)
        return process

    def __create_mdo_parallel_chain(
        self,
        parallel_disciplines: Sequence[MDODiscipline],
        mdachain_parallel_options: dict[str, int | float | str] | None,
    ) -> MDOParallelChain:
        """Create a :class:`.MDOParallelChain` with the provided disciplines and options.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallel_options: The :class:`.MDOParallelChain` options.

        Returns:
            A :class:`.MDOParallelChain` instance.
        """
        if mdachain_parallel_options is None:
            mdachain_parallel_options = {}
        return MDOParallelChain(
            parallel_disciplines,
            grammar_type=self.grammar_type,
            **mdachain_parallel_options,
        )

    def _initialize_grammars(self) -> None:
        """Define all inputs and outputs of the chain."""
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        self.input_grammar.update(self.mdo_chain.input_grammar)
        self.output_grammar.update(self.mdo_chain.output_grammar)
        self._add_residuals_norm_to_output_grammar()

    def _check_consistency(self):
        """Check if there is no more than 1 equation per variable.

        For instance if a strong coupling is not also a self coupling.
        """
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        super()._check_consistency()

    def _run(self) -> None:
        if self.warm_start:
            self._couplings_warm_start()
        self.local_data = self.mdo_chain.execute(self.local_data)

        res_sum = 0.0
        for mda in self.inner_mdas:
            res_local = mda.local_data.get(self.RESIDUALS_NORM)
            if res_local is not None:
                res_sum += res_local[-1] ** 2
        self.local_data[self.RESIDUALS_NORM] = array([res_sum**0.5])
        return self.local_data

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)
            self.mdo_chain.add_differentiated_outputs(outputs)
            # the Jacobian of the MDA chain is the Jacobian of the MDO chain
            self.mdo_chain.linearize(self.get_input_data())
            self.jac = self.mdo_chain.jac
        else:
            super()._compute_jacobian(inputs, outputs)

    def add_differentiated_inputs(
        self,
        inputs: Iterable[str] | None = None,
    ) -> None:
        MDA.add_differentiated_inputs(self, inputs)
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)

    def add_differentiated_outputs(
        self,
        outputs: Iterable[str] | None = None,
    ) -> None:  # noqa: D102
        MDA.add_differentiated_outputs(self, outputs=outputs)
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_outputs(outputs)

    @property
    def normed_residual(self) -> float:
        """The normed_residuals, computed from the sub-MDAs residuals."""
        return sum(mda.normed_residual**2 for mda in self.inner_mdas) ** 0.5

    @normed_residual.setter
    def normed_residual(
        self,
        normed_residual: float,
    ) -> None:
        """Set the normed_residual.

        Has no effect,
        since the normed residuals are defined by inner-MDAs residuals
        (see associated property).

        Here for compatibility with mother class.
        """

    def get_expected_dataflow(
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.mdo_chain.get_expected_dataflow()

    def get_expected_workflow(self) -> SerialExecSequence:
        exec_s = SerialExecSequence(self)
        workflow = self.mdo_chain.get_expected_workflow()
        exec_s.extend(workflow)
        return exec_s

    def reset_statuses_for_run(self) -> None:
        super().reset_statuses_for_run()
        self.mdo_chain.reset_statuses_for_run()

    def plot_residual_history(
        self,
        show: bool = False,
        save: bool = True,
        n_iterations: int | None = None,
        logscale: tuple[int, int] | None = None,
        filename: str | None = None,
        fig_size: tuple[float, float] = (50.0, 10.0),
    ) -> None:
        for mda in self.inner_mdas:
            if filename is not None:
                s_filename = split(filename)
                filename = join(
                    s_filename[0],
                    f"{mda.__class__.__name__}_{s_filename[1]}",
                )
            mda.plot_residual_history(
                show, save, n_iterations, logscale, filename, fig_size
            )
