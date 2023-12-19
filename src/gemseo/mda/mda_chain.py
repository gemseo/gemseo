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
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import array

from gemseo import create_mda
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import SerialExecSequence
from gemseo.mda.initialization_chain import MDOInitializationChain
from gemseo.mda.mda import MDA

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.coupling_structure import MDOCouplingStructure
    from gemseo.core.discipline_data import DisciplineData
    from gemseo.utils.matplotlib_figure import FigSizeType

LOGGER = logging.getLogger(__name__)

N_CPUS = cpu_count()


class MDAChain(MDA):
    """A chain of MDAs.

    The execution sequence is provided by the :class:`.DependencyGraph`.
    """

    inner_mdas: list[MDA]
    """The ordered MDAs."""

    __initialize_defaults: bool
    """Whether to compute the eventually missing :attr:`.default_inputs`."""

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
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        coupling_structure: MDOCouplingStructure | None = None,
        sub_coupling_structures: Iterable[MDOCouplingStructure | None] | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
        mdachain_parallelize_tasks: bool = False,
        mdachain_parallel_options: Mapping[str, int | bool] | None = None,
        initialize_defaults: bool = False,
        **inner_mda_options: float | int | bool | str | None,
    ) -> None:
        """
        Args:
            inner_mda_name: The class name of the inner-MDA.
            n_processes: The maximum simultaneous number of threads if ``use_threading``
                is set to True, otherwise processes, used to parallelize the execution.
            chain_linearize: Whether to linearize the chain of execution. Otherwise,
                linearize the overall MDA with base class method. This last option is
                preferred to minimize computations in adjoint mode, while in direct
                mode, linearizing the chain may be cheaper.
            sub_coupling_structures: The coupling structures to be used by the
                inner-MDAs. If ``None``, they are created from the sub-disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks, if
                any.
            mdachain_parallel_options: The options of the MDOParallelChain instances, if
                any.
            initialize_defaults: Whether to create a :class:`.MDOInitializationChain`
                to compute the eventually missing :attr:`.default_inputs` at the first
                execution.
            **inner_mda_options: The options of the inner-MDAs.
        """  # noqa:D205 D212 D415
        self.n_processes = n_processes
        self.mdo_chain = None
        self._chain_linearize = chain_linearize
        self.inner_mdas = []

        # compute execution sequence of the disciplines
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            use_lu_fact=use_lu_fact,
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
        self.__initialize_defaults = initialize_defaults
        self._initialize_grammars()
        self._check_consistency()
        self._compute_input_couplings()

        # cascade the tolerance
        for mda in self.inner_mdas:
            mda.tolerance = self.tolerance

    @property
    def max_mda_iter(self) -> int:
        """The maximum iterations number of each of the inner MDA algorithms."""
        return super().max_mda_iter

    @max_mda_iter.setter
    def max_mda_iter(self, max_mda_iter: int) -> None:  # noqa: D102
        self._max_mda_iter = max_mda_iter
        for mda in self.inner_mdas:
            mda.max_mda_iter = max_mda_iter

    @MDA.log_convergence.setter
    def log_convergence(  # noqa: D102
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
        sub_coupling_structures: Iterable[MDOCouplingStructure | None] | None = None,
        mdachain_parallelize_tasks: bool = False,
        mdachain_parallel_options: Mapping[str, int | bool] | None = None,
        **inner_mda_options: float | int | bool | str | None,
    ) -> None:
        """Create an MDO chain from the execution sequence of the disciplines.

        Args:
            disciplines: The disciplines.
            inner_mda_name: The name of the class of the inner-MDAs.
            acceleration: The acceleration method to be used to improve the convergence
                rate of the fixed point iteration method.
            over_relax_factor: The over-relaxation factor.
            sub_coupling_structures: The coupling structures to be used by the inner
                MDAs. If ``None``, they are created from the sub-disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the
                parallel tasks, if any.
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
                parallel_tasks,
                mdachain_parallelize_tasks,
                mdachain_parallel_options,
                inner_mda_options,
            )
            chained_disciplines.append(process)

        self.mdo_chain = MDOChain(
            chained_disciplines, name="MDA chain", grammar_type=self.grammar_type
        )

    def __create_process_from_disciplines(
        self,
        disciplines: Sequence[MDODiscipline],
        inner_mda_name: str,
        parallel_tasks: list[tuple[MDODiscipline]],
        mdachain_parallelize_tasks: bool,
        mdachain_parallel_options: Mapping[str, int | bool] | None,
        inner_mda_options: Mapping[str, float | int | bool | str | None],
    ) -> MDODiscipline:
        """Create a process from disciplines.

        This method creates a process that will be appended to the main inner
        :class:`.MDOChain` of the :class:`.MDAChain`. Depending on the number and type
        of disciplines, as well as the options provided by the user, the process may be
        a sole discipline, a :class:`.MDA`, a :class:`MDOChain`, or a
        :class:`MDOParallelChain`.

        Args:
            disciplines: The disciplines.
            inner_mda_name: The inner :class:`.MDA` class name.
            acceleration: The acceleration method to be used to improve the convergence
                rate of the fixed point iteration method.
            over_relax_factor: The over-relaxation factor.
            parallel_tasks: The parallel tasks to be processed.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks,
                if any.
            mdachain_parallel_options: The :class:`MDOParallelChain` options.
            inner_mda_options: The inner :class:`.MDA` options.

        Returns:
            A process.
        """
        parallel_disciplines = self.__compute_parallel_disciplines(
            disciplines,
            inner_mda_name,
            parallel_tasks,
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
        inner_mda_name: str,
        parallel_tasks: list[tuple[MDODiscipline]],
        inner_mda_options: Mapping[str, float | int | bool | str | None],
    ) -> Sequence[MDODiscipline | MDA]:
        """Compute the parallel disciplines.

        This method computes the parallel disciplines,
        if any.
        If there is any coupled disciplines in a parallel task,
        an :class:`.MDA` is created,
        based on the :class:`.MDA` options provided.

        Args:
            disciplines: The disciplines.
            inner_mda_name: The inner :class:`.MDA` class name.
            acceleration: The acceleration method to be used to improve the convergence
                rate of the fixed point iteration method.
            over_relax_factor: The over-relaxation factor.
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
        mdachain_parallel_options: Mapping[str, int | bool] | None,
    ) -> MDODiscipline | MDOChain | MDOParallelChain:
        """Create a process from parallel disciplines.

        Depending on the number of disciplines and the options provided,
        the returned GEMSEO process can be a sole :class:`.MDODiscipline` instance,
        an :class:`.MDOChain` or an :class:`.MDOParallelChain`.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks.
            mdachain_parallel_options: The options of the :class:`.MDOParallelChain`.

        Returns:
            A GEMSEO process instance.
        """
        if len(parallel_disciplines) == 1:
            return parallel_disciplines[0]

        return self.__create_sequential_or_parallel_chain(
            parallel_disciplines,
            mdachain_parallelize_tasks,
            mdachain_parallel_options,
        )

    def __create_inner_mda(
        self,
        disciplines: Sequence[MDODiscipline],
        coupled_disciplines: Sequence[MDODiscipline],
        inner_mda_name: str,
        inner_mda_options: Mapping[str, float | int | bool | str | None],
    ) -> MDA:
        """Create an inner MDA from the coupled disciplines and the MDA options.

        Args:
            disciplines: The disciplines.
            coupled_disciplines: The coupled disciplines.
            inner_mda_name: The inner :class:`.MDA` class name.
            inner_mda_options: The inner :class:`.MDA` options.
            acceleration: The acceleration method to be used to improve the convergence
                rate of the fixed point iteration method.
            over_relax_factor: The over-relaxation factor.

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
        """Return whether only one self-coupled discipline which is also not an MDA.

        Args:
            disciplines: The disciplines.

        Returns:
            True if the sole discipline of coupled_disciplines is self-coupled
            and not an MDA.
        """
        first_discipline = disciplines[0]
        return (
            len(disciplines) == 1
            and self.coupling_structure.is_self_coupled(first_discipline)
            and not isinstance(disciplines[0], MDA)
        )

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
        mdachain_parallel_options: Mapping[str, int | bool] | None,
    ) -> MDOChain | MDOParallelChain:
        """Create an :class:`.MDOChain` or :class:`.MDOParallelChain`.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallelize_tasks: Whether to parallelize the parallel tasks,
                if any.
            mdachain_parallel_options: The :class:`MDOParallelChain options.

        Returns:
            Either an :class:`.MDOChain` or :class:`.MDOParallelChain instance.
        """
        if mdachain_parallelize_tasks:
            return self.__create_mdo_parallel_chain(
                parallel_disciplines,
                mdachain_parallel_options,
            )
        return MDOChain(parallel_disciplines, grammar_type=self.grammar_type)

    def __create_mdo_parallel_chain(
        self,
        parallel_disciplines: Sequence[MDODiscipline],
        mdachain_parallel_options: Mapping[str, int | bool] | None,
    ) -> MDOParallelChain:
        """Create an :class:`.MDOParallelChain`.

        Args:
            parallel_disciplines: The parallel disciplines.
            mdachain_parallel_options: The :class:`.MDOParallelChain` options.

        Returns:
            an :class:`.MDOParallelChain` instance.
        """
        if mdachain_parallel_options is None:
            mdachain_parallel_options = {}

        return MDOParallelChain(
            parallel_disciplines,
            grammar_type=self.grammar_type,
            name=None,
            **mdachain_parallel_options,
        )

    def _initialize_grammars(self) -> None:
        """Define all inputs and outputs of the chain."""
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        self.input_grammar = self.mdo_chain.input_grammar.copy()
        self.output_grammar = self.mdo_chain.output_grammar.copy()

    def _check_consistency(self) -> None:
        """Check if there is no more than 1 equation per variable.

        For instance if a strong coupling is not also a self coupling.
        """
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        super()._check_consistency()

    def execute(  # noqa:D102
        self, input_data: Mapping[str, Any] | None = None
    ) -> DisciplineData:
        if self.__initialize_defaults:
            init_chain = MDOInitializationChain(
                self.disciplines, available_data_names=input_data or ()
            )
            self.default_inputs.update(init_chain.execute(input_data))
            self.__initialize_defaults = False
        return super().execute(input_data=input_data)

    def _run(self) -> None:
        super()._run()

        self.local_data = self.mdo_chain.execute(self.local_data)

        res_sum = 0.0
        for mda in self.inner_mdas:
            res_local = mda.local_data.get(self.RESIDUALS_NORM)
            if res_local is not None:
                res_sum += res_local[-1] ** 2
        self.local_data[self.RESIDUALS_NORM] = array([res_sum**0.5])

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

    def add_differentiated_inputs(  # noqa:D102
        self,
        inputs: Iterable[str] | None = None,
    ) -> None:
        MDA.add_differentiated_inputs(self, inputs)
        if self._chain_linearize:
            self.mdo_chain.add_differentiated_inputs(inputs)

    def add_differentiated_outputs(  # noqa: D102
        self,
        outputs: Iterable[str] | None = None,
    ) -> None:
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

        Has no effect, since the normed residuals are defined by inner-MDAs residuals
        (see associated property).

        Here for compatibility with mother class.
        """

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        return self.mdo_chain.get_expected_dataflow()

    def get_expected_workflow(self) -> SerialExecSequence:  # noqa:D102
        exec_s = SerialExecSequence(self)
        workflow = self.mdo_chain.get_expected_workflow()
        exec_s.extend(workflow)
        return exec_s

    def reset_statuses_for_run(self) -> None:  # noqa:D102
        super().reset_statuses_for_run()
        self.mdo_chain.reset_statuses_for_run()

    def plot_residual_history(  # noqa: D102
        self,
        show: bool = False,
        save: bool = True,
        n_iterations: int | None = None,
        logscale: tuple[int, int] | None = None,
        filename: Path | str = "",
        fig_size: FigSizeType = (50.0, 10.0),
    ) -> None:
        if filename:
            file_path = Path(filename)
        for mda in self.inner_mdas:
            if filename:
                path = file_path.parent / f"{mda.__class__.__name__}_{file_path.name}"
            else:
                path = filename
            mda.plot_residual_history(
                show, save, n_iterations, logscale, path, fig_size
            )
