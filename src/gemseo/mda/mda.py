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
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Base class for all Multi-disciplinary Design Analyses (MDA)."""

from __future__ import annotations

import logging
from abc import abstractmethod
from enum import auto
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from strenum import LowercaseStrEnum

from gemseo.algos.sequence_transformer.acceleration import AccelerationMethod
from gemseo.algos.sequence_transformer.composite.relaxation_acceleration import (
    RelaxationAcceleration,
)
from gemseo.caches.simple_cache import SimpleCache
from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import ExecutionSequenceFactory
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Mapping
    from collections.abc import Sequence
    from pathlib import Path
    from typing import Any
    from typing import ClassVar

    from matplotlib.figure import Figure
    from numpy import ndarray
    from numpy.typing import NDArray

    from gemseo.core.discipline_data import DisciplineData
    from gemseo.core.execution_sequence import LoopExecSequence
    from gemseo.utils.matplotlib_figure import FigSizeType

LOGGER = logging.getLogger(__name__)


# TODO: API: rename to BaseMDA.
class MDA(MDODiscipline, metaclass=ABCGoogleDocstringInheritanceMeta):
    """An MDA analysis."""

    RESIDUALS_NORM: ClassVar[str] = "MDA residuals norm"

    activate_cache = True

    tolerance: float
    """The tolerance of the iterative direct coupling solver."""

    linear_solver: str
    """The name of the linear solver."""

    linear_solver_tolerance: float
    """The tolerance of the linear solver in the adjoint equation."""

    linear_solver_options: Mapping[str, Any]
    """The options of the linear solver."""

    _max_mda_iter: int
    """The maximum iterations number for the MDA algorithm."""

    coupling_structure: MDOCouplingStructure
    """The coupling structure to be used by the MDA."""

    assembly: JacobianAssembly

    residual_history: list[float]
    """The history of the MDA residuals."""

    reset_history_each_run: bool
    """Whether to reset the history of MDA residuals before each run."""

    warm_start: bool
    """Whether the second iteration and ongoing start from the previous solution."""

    scaling: ResidualScaling
    """The scaling method applied to MDA residuals for convergence monitoring."""

    _scaling_data: float | list[tuple[slice, float]] | NDArray[float] | None
    """The data required to perform the scaling of the MDA residuals."""

    norm0: float | None
    """The reference residual, if any."""

    normed_residual: float
    """The normed residual."""

    strong_couplings: list[str]
    """The names of the strong coupling variables."""

    all_couplings: list[str]
    """The names of all the coupling variables."""

    matrix_type: JacobianAssembly.JacobianType
    """The type of the matrix."""

    use_lu_fact: bool
    """Whether to store a LU factorization of the matrix."""

    lin_cache_tol_fact: float
    """The tolerance factor to cache the Jacobian."""

    _starting_indices: list[int]
    """The indices of the residual history where a new execution starts."""

    _sequence_transformer: RelaxationAcceleration
    """The sequence transformer aimed at improving the convergence rate.

    The transformation applies a relaxation followed by an acceleration.
    """

    class ResidualScaling(LowercaseStrEnum):
        """The scaling method applied to MDA residuals for convergence monitoring."""

        NO_SCALING = auto()
        r"""The residual vector is not scaled. The MDA is considered converged when its
        Euclidean norm satisfies,

        .. math::

            \|R_k\|_2 \leq \text{tol}.
        """

        INITIAL_RESIDUAL_NORM = auto()
        r"""The :math:k`-th residual vector is scaled by the Euclidean norm of the
        initial residual (if not null, else it is not scaled). The MDA is considered
        converged when its Euclidean norm satisfies,

        .. math::

            \frac{ \|R_k\|_2 }{ \|R_0\|_2 } \leq \text{tol}.
        """

        INITIAL_SUBRESIDUAL_NORM = auto()
        r"""The :math:k`-th residual vector is scaled discipline-wise. The sub-residual
        associated wich each discipline is scaled by the Euclidean norm of the initial
        sub-residual (if not null, else it is not scaled). The MDA is considered
        converged when the Euclidean norm of each sub-residual satisfies,

        .. math::

            \max_i \left| \frac{\|r^i_k\|_2}{\|r^i_0\|_2} \right| \leq \text{tol}.
        """

        N_COUPLING_VARIABLES = auto()
        r"""The :math:k`-th residual vector is scaled using the number of coupling
        variables. The MDA is considered converged when its Euclidean norm satisfies,
        .. math::

            \frac{ \|R_k\|_2 }{ \sqrt{n_\text{coupl.}} } \leq \text{tol}.
        """

        INITIAL_RESIDUAL_COMPONENT = auto()
        r"""The :math:k`-th residual is scaled component-wise. Each component is scaled
        by the corresponding component of the initial residual (if not null, else it is
        not scaled). The MDA is considered converged when each component satisfies,

        .. math::

            \max_i \left| \frac{(R_k)_i}{(R_0)_i} \right| \leq \text{tol}.
        """

        SCALED_INITIAL_RESIDUAL_COMPONENT = auto()
        r"""The :math:k`-th residual vector is scaled component-wise and by the number
        coupling variables. If :math:`\div` denotes the component-wise division between
        two vectors, then the MDA is considered converged when the residual vector
        satisfies,

        .. math::

            \frac{1}{\sqrt{n_\text{coupl.}}} \| R_k \div R_0 \|_2 \leq \text{tol}.
        """

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
        acceleration_method: AccelerationMethod = AccelerationMethod.NONE,
        over_relaxation_factor: float = 1.0,
    ) -> None:
        """
        Args:
            disciplines: The disciplines from which to compute the MDA.
            max_mda_iter: The maximum iterations number for the MDA algorithm.
            name: The name to be given to the MDA.
                If ``None``, use the name of the class.
            grammar_type: The type of the input and output grammars.
            tolerance: The tolerance of the iterative direct coupling solver;
                the norm of the current residuals divided by initial residuals norm
                shall be lower than the tolerance to stop iterating.
            linear_solver_tolerance: The tolerance of the linear solver
                in the adjoint equation.
            warm_start: Whether the second iteration and ongoing start
                from the previous coupling solution.
            use_lu_fact: Whether to store a LU factorization of the matrix
                when using adjoint/forward differentiation.
                to solve faster multiple RHS problem.
            coupling_structure: The coupling structure to be used by the MDA.
                If ``None``, it is created from `disciplines`.
            log_convergence: Whether to log the MDA convergence,
                expressed in terms of normed residuals.
            linear_solver: The name of the linear solver.
            linear_solver_options: The options passed to the linear solver factory.
            acceleration_method: The acceleration method to be used to improve the
                convergence rate of the fixed point iteration method.
            over_relaxation_factor: The over-relaxation factor.
        """  # noqa:D205 D212 D415
        super().__init__(name, grammar_type=grammar_type)
        self.tolerance = tolerance
        self.linear_solver = linear_solver
        self.linear_solver_tolerance = linear_solver_tolerance
        self.linear_solver_options = linear_solver_options or {}
        self.max_mda_iter = max_mda_iter
        self._disciplines = disciplines
        if coupling_structure is None:
            self.coupling_structure = MDOCouplingStructure(disciplines)
        else:
            self.coupling_structure = coupling_structure
        self.assembly = JacobianAssembly(self.coupling_structure)
        self.residual_history = []
        self._starting_indices = []
        self.reset_history_each_run = False
        self.warm_start = warm_start

        self._sequence_transformer = RelaxationAcceleration(
            over_relaxation_factor, acceleration_method
        )

        self.scaling = self.ResidualScaling.INITIAL_RESIDUAL_NORM
        self._scaling_data = None

        # Don't erase coupling values before calling _compute_jacobian

        self._linearize_on_last_state = True
        self.norm0 = None
        self._current_iter = 0
        self.normed_residual = 1.0
        self.strong_couplings = self.coupling_structure.strong_couplings
        self.all_couplings = self.coupling_structure.all_couplings
        self._input_couplings = []
        self.matrix_type = JacobianAssembly.JacobianType.MATRIX
        self.use_lu_fact = use_lu_fact
        # By default don't use an approximate cache for linearization
        self.lin_cache_tol_fact = 0.0

        self._initialize_grammars()
        self.output_grammar.update_from_names([self.RESIDUALS_NORM])
        self._check_consistency()
        self.__check_linear_solver_options()
        self._check_coupling_types()
        self._log_convergence = log_convergence

    @property
    def acceleration_method(self) -> AccelerationMethod:
        """The acceleration method."""
        return self._sequence_transformer.acceleration_method

    @acceleration_method.setter
    def acceleration_method(self, acceleration_method: AccelerationMethod) -> None:
        self._sequence_transformer.acceleration_method = acceleration_method

    @property
    def over_relaxation_factor(self) -> float:
        """The over-relaxation factor."""
        return self._sequence_transformer.over_relaxation_factor

    @over_relaxation_factor.setter
    def over_relaxation_factor(self, over_relaxation_factor: float) -> None:
        self._sequence_transformer.over_relaxation_factor = over_relaxation_factor

    # TODO: API: this property is useless, either remove it or at least check it is
    # positive in the setter.
    @property
    def max_mda_iter(self) -> int:
        """The maximum iterations number of the MDA algorithm."""
        return self._max_mda_iter

    @max_mda_iter.setter
    def max_mda_iter(self, max_mda_iter: int) -> None:
        self._max_mda_iter = max_mda_iter

    def _initialize_grammars(self) -> None:
        """Define all the inputs and outputs of the MDA.

        Add all the outputs of all the disciplines to the outputs.
        """
        for discipline in self.disciplines:
            self.input_grammar.update(discipline.input_grammar)
            self.output_grammar.update(discipline.output_grammar)

    # TODO: API: this property is useless, remove it?
    @property
    def log_convergence(self) -> bool:
        """Whether to log the MDA convergence."""
        return self._log_convergence

    @log_convergence.setter
    def log_convergence(
        self,
        value: bool,
    ) -> None:
        self._log_convergence = value

    def __check_linear_solver_options(self) -> None:
        """Check the linear solver options.

        The linear solver tolerance cannot be set
        using the linear solver option dictionary,
        as it is set using the linear_solver_tolerance keyword argument.

        Raises:
            ValueError: If the ``tol`` keyword is in :attr:`.linear_solver_options`.
        """
        if "tol" in self.linear_solver_options:
            msg = (
                "The linear solver tolerance shall be set"
                " using the linear_solver_tolerance argument."
            )
            raise ValueError(msg)

    def _check_consistency(self) -> None:
        """Check if there are not more than one equation per variable.

        For instance if a strong coupling is not also a self coupling, or if outputs are
        defined multiple times.
        """
        strong_c_disc = self.coupling_structure.get_strongly_coupled_disciplines(
            add_self_coupled=False
        )
        also_strong = [
            disc
            for disc in strong_c_disc
            if self.coupling_structure.is_self_coupled(disc)
        ]
        if also_strong:
            for disc in also_strong:
                in_outs = sorted(
                    set(disc.get_input_data_names()) & set(disc.get_output_data_names())
                )
                LOGGER.warning(
                    "Self coupling variables in discipline %s are: %s.",
                    disc.name,
                    in_outs,
                )

            also_strong_n = sorted(disc.name for disc in also_strong)
            LOGGER.warning(
                "The following disciplines contain self-couplings and strong couplings:"
                " %s. This is not a problem as long as their self-coupling variables "
                "are not strongly coupled to another discipline.",
                also_strong_n,
            )

        all_outs = {}
        multiple_outs = []
        for disc in self.disciplines:
            for out in disc.get_output_data_names():
                if out in all_outs:
                    multiple_outs.append(out)
                all_outs[out] = disc

        if multiple_outs:
            LOGGER.warning(
                "The following outputs are defined multiple times: %s.",
                sorted(multiple_outs),
            )

    # TODO: API: better naming: _compute_input_coupling_names
    def _compute_input_couplings(self) -> None:
        """Compute the strong couplings that are inputs of the MDA."""
        self._input_couplings = sorted(
            set(self.strong_couplings).intersection(self.get_input_data_names())
        )

    def _retrieve_diff_inouts(
        self, compute_all_jacobians: bool = False
    ) -> tuple[set[str] | list[str], set[str] | list[str]]:
        if compute_all_jacobians:
            strong_cpl = set(self.strong_couplings)
            inputs = set(self.get_input_data_names())
            outputs = self.get_output_data_names()
            # Don't linearize wrt
            inputs -= strong_cpl & inputs
            # Don't do this with output couplings because
            # their derivatives wrt design variables may be needed
            # outputs = outputs - (strong_cpl & outputs)
        else:
            inputs, outputs = MDODiscipline._retrieve_diff_inouts(self)

        if self.RESIDUALS_NORM in outputs:
            outputs = list(outputs)
            outputs.remove(self.RESIDUALS_NORM)

        return inputs, outputs

    def _check_coupling_types(self) -> None:
        """Check that the coupling variables are of type array in the grammars.

        Raises:
            TypeError: When at least one of the coupling variables is not an array.
        """
        not_arrays = set()
        for coupling_name in self.all_couplings:
            for discipline in self.disciplines:
                for grammar in (discipline.input_grammar, discipline.output_grammar):
                    if (
                        coupling_name in grammar
                        and not grammar.data_converter.is_numeric(coupling_name)
                    ):
                        not_arrays.add(coupling_name)
                        break

        if not_arrays:
            raise TypeError(
                f"The coupling variables {sorted(not_arrays)} must be numeric."
            )

    def reset_disciplines_statuses(self) -> None:
        """Reset all the statuses of the disciplines."""
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()

    def reset_statuses_for_run(self) -> None:  # noqa:D102
        super().reset_statuses_for_run()
        self.reset_disciplines_statuses()

    def get_expected_workflow(self) -> LoopExecSequence:  # noqa:D102
        disc_exec_seq = ExecutionSequenceFactory.serial()
        for disc in self.disciplines:
            disc_exec_seq.extend(disc.get_expected_workflow())
        return ExecutionSequenceFactory.loop(self, disc_exec_seq)

    def get_expected_dataflow(  # noqa:D102
        self,
    ) -> list[tuple[MDODiscipline, MDODiscipline, list[str]]]:
        all_disc = [self, *self.disciplines]
        graph = DependencyGraph(all_disc)
        res = graph.get_disciplines_couplings()
        for discipline in self.disciplines:
            res.extend(discipline.get_expected_dataflow())
        return res

    def _compute_jacobian(
        self,
        inputs: Collection[str] | None = None,
        outputs: Collection[str] | None = None,
    ) -> None:
        # Do not re-execute disciplines if inputs error is beyond self tol
        # Apply a safety factor on this (mda is a loop, inputs
        # of first discipline
        # have changed at convergence, therefore the cache is not exactly
        # the same as the current value
        exec_cache_tol = self.lin_cache_tol_fact * self.tolerance
        self.__check_linear_solver_options()
        residual_variables = {}
        for disc in self.disciplines:
            residual_variables.update(disc.residual_variables)

        couplings_adjoint = sorted(
            set(self.all_couplings)
            - residual_variables.keys()
            - set(residual_variables.values())
        )

        self.jac = self.assembly.total_derivatives(
            self.local_data,
            outputs,
            inputs,
            couplings_adjoint,
            tol=self.linear_solver_tolerance,
            mode=self.linearization_mode,
            matrix_type=self.matrix_type,
            use_lu_fact=self.use_lu_fact,
            exec_cache_tol=exec_cache_tol,
            execute=exec_cache_tol == 0.0,
            linear_solver=self.linear_solver,
            residual_variables=residual_variables,
            **self.linear_solver_options,
        )

    def check_jacobian(
        self,
        input_data: Mapping[str, ndarray] | None = None,
        derr_approx: MDODiscipline.ApproximationMode = MDODiscipline.ApproximationMode.FINITE_DIFFERENCES,  # noqa:E501
        step: float = 1e-7,
        threshold: float = 1e-8,
        linearization_mode: str = "auto",
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
        parallel: bool = False,
        n_processes: int = MDODiscipline.N_CPUS,
        use_threading: bool = False,
        wait_time_between_fork: int = 0,
        auto_set_step: bool = False,
        plot_result: bool = False,
        file_path: str | Path = "jacobian_errors.pdf",
        show: bool = False,
        fig_size_x: float = 10,
        fig_size_y: float = 10,
        reference_jacobian_path: None | Path | str = None,
        save_reference_jacobian: bool = False,
        indices: Iterable[int] | None = None,
    ) -> bool:
        """Check if the analytical Jacobian is correct with respect to a reference one.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `True`,
        compute the reference Jacobian with the approximation method
        and save it in `reference_jacobian_path`.

        If `reference_jacobian_path` is not `None`
        and `save_reference_jacobian` is `False`,
        do not compute the reference Jacobian
        but read it from `reference_jacobian_path`.

        If `reference_jacobian_path` is `None`,
        compute the reference Jacobian without saving it.

        Args:
            input_data: The input values.
                If ``None``, use the default input values.
            derr_approx: The derivative approximation method.
            threshold: The acceptance threshold for the Jacobian error.
            linearization_mode: The mode of linearization,
                either "direct", "adjoint" or "auto" switch
                depending on dimensions of inputs and outputs.
            inputs: The names of the inputs with respect to which to differentiate.
                If ``None``, use the inputs of the MDA.
            outputs: The outputs to differentiate.
                If ``None``, use all the outputs of the MDA.
            step: The step
                for finite differences or complex step differentiation methods.
            parallel: Whether to execute the MDA in parallel.
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
            wait_time_between_fork: The time waited between two forks
                of the process / thread.
            auto_set_step: Whether to compute the optimal step
                for a forward first order finite differences gradient approximation.
            plot_result: Whether to plot the result of the validation
                comparing the exact and approximated Jacobians.
            file_path: The path to the output file if `plot_result` is `True`.
            show: Whether to open the figure.
            fig_size_x: The *x* size of the figure in inches.
            fig_size_y: The *y* size of the figure in inches.
            reference_jacobian_path: The path of the reference Jacobian file.
            save_reference_jacobian: Whether to save the reference Jacobian.
            indices: The indices of the inputs and outputs
                for the different sub-Jacobian matrices,
                formatted as ``{variable_name: variable_components}``
                where ``variable_components`` can be either
                an integer, e.g. `2`
                a sequence of integers, e.g. `[0, 3]`,
                a slice, e.g. `slice(0,3)`,
                the ellipsis symbol (`...`)
                or `None`, which is the same as ellipsis.
                If a variable name is missing, consider all its components.
                If ``None``,
                consider all the components of all the ``inputs`` and ``outputs``.

        Returns:
            Whether the passed Jacobian is correct.
        """
        # Strong couplings are not linearized
        if inputs is None:
            inputs = self.get_input_data_names()
        if outputs is None:
            outputs = self.get_output_data_names()

        inputs = list(inputs)
        outputs = list(outputs)

        for coupling in self.all_couplings:
            if coupling in outputs:
                outputs.remove(coupling)
            if coupling in inputs:
                inputs.remove(coupling)

        if self.RESIDUALS_NORM in outputs:
            outputs.remove(self.RESIDUALS_NORM)

        return super().check_jacobian(
            input_data=input_data,
            derr_approx=derr_approx,
            step=step,
            threshold=threshold,
            linearization_mode=linearization_mode,
            inputs=inputs,
            outputs=outputs,
            parallel=parallel,
            n_processes=n_processes,
            use_threading=use_threading,
            wait_time_between_fork=wait_time_between_fork,
            auto_set_step=auto_set_step,
            plot_result=plot_result,
            file_path=file_path,
            show=show,
            fig_size_x=fig_size_x,
            fig_size_y=fig_size_y,
            reference_jacobian_path=reference_jacobian_path,
            save_reference_jacobian=save_reference_jacobian,
            indices=indices,
        )

    def execute(  # noqa:D102
        self, input_data: Mapping[str, Any] | None = None
    ) -> DisciplineData:
        self._current_iter = 0
        return super().execute(input_data=input_data)

    def _set_cache_tol(
        self,
        cache_tol: float,
    ) -> None:
        """Set to the cache input tolerance.

        To be overloaded by subclasses.

        Args:
            cache_tol: The cache tolerance.
        """
        super()._set_cache_tol(cache_tol)
        for disc in self.disciplines:
            disc.cache_tol = cache_tol or 0.0

    def plot_residual_history(
        self,
        show: bool = False,
        save: bool = True,
        n_iterations: int | None = None,
        logscale: tuple[int, int] | None = None,
        filename: Path | str = "",
        fig_size: FigSizeType | None = None,
    ) -> Figure:
        """Generate a plot of the residual history.

        The first iteration of each new execution is marked with a red dot.

        Args:
            show: Whether to display the plot on screen.
            save: Whether to save the plot as a PDF file.
            n_iterations: The number of iterations on the *x* axis.
                If ``None``, use all the iterations.
            logscale: The limits of the *y* axis.
                If ``None``, do not change the limits of the *y* axis.
            filename: The name of the file to save the figure.
                If empty, use "{mda.name}_residual_history.pdf".
            fig_size: The width and height of the figure in inches, e.g. `(w, h)`.

        Returns:
            The figure, to be customized if not closed.
        """
        fig = plt.figure()
        fig_ax = fig.add_subplot(1, 1, 1)

        history_length = len(self.residual_history)
        n_iterations = n_iterations or history_length

        if n_iterations > history_length:
            msg = (
                "Requested %s iterations but the residual history contains only %s, "
                "plotting all the residual history."
            )
            LOGGER.info(msg, n_iterations, history_length)
            n_iterations = history_length

        # red dot for first iteration
        colors = ["black"] * n_iterations
        for index in self._starting_indices:
            colors[index] = "red"

        fig_ax.scatter(
            list(range(n_iterations)),
            self.residual_history[:n_iterations],
            s=20,
            color=colors,
            zorder=2,
        )
        fig_ax.plot(
            self.residual_history[:n_iterations], linestyle="-", c="k", zorder=1
        )
        fig_ax.axhline(y=self.tolerance, c="blue", linewidth=0.5, zorder=0)
        fig_ax.set_title(f"{self.name}: residual plot")

        fig_ax.set_yscale("log")
        fig_ax.set_xlabel(r"iterations", fontsize=14)
        fig_ax.set_xlim([-1, n_iterations])
        fig_ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        fig_ax.set_ylabel(r"$\log(||residuals||/||y_0||)$", fontsize=14)
        if logscale is not None:
            fig_ax.set_ylim(logscale)

        if save and not filename:
            filename = f"{self.name}_residual_history.pdf"

        save_show_figure(fig, show, filename, fig_size=fig_size)

        return fig

    # TODO: API: better naming: _prepare_warm_start
    def _couplings_warm_start(self) -> None:
        """Load the previous couplings values to local data."""
        cached_outputs = self.cache.last_entry.outputs

        if not cached_outputs:
            return

        # Non simple caches require NumPy arrays.
        if not isinstance(self.cache, SimpleCache):
            to_value = self.input_grammar.data_converter.convert_array_to_value
            for input_name, input_value in self.__get_cached_outputs(cached_outputs):
                self.local_data[input_name] = to_value(input_name, input_value)
        else:
            self.local_data.update(dict(self.__get_cached_outputs(cached_outputs)))

    def __get_cached_outputs(self, cached_outputs) -> Iterator[Any]:
        """Return an iterator over the input couplings names and value in cache.

        Args:
            cached_outputs: The cached outputs.

        Returns:
            The names and value of the input couplings in cache.
        """
        for input_name in self._input_couplings:
            input_value = cached_outputs.get(input_name)
            if input_value is not None:
                yield input_name, input_value

    @abstractmethod
    def _run(self) -> None:  # noqa:D103
        if self.warm_start:
            self._couplings_warm_start()
