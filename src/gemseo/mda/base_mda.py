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
#
# Copyright 2024 Capgemini
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
from typing import Any
from typing import ClassVar
from typing import Final

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from strenum import LowercaseStrEnum

from gemseo.caches.simple_cache import SimpleCache
from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.coupling_structure import DependencyGraph
from gemseo.core.derivatives.jacobian_assembly import JacobianAssembly
from gemseo.core.discipline import Discipline
from gemseo.core.process_discipline import ProcessDiscipline
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.matplotlib_figure import save_show_figure
from gemseo.utils.pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Iterator
    from collections.abc import Sequence
    from pathlib import Path

    from matplotlib.figure import Figure
    from numpy.typing import NDArray

    from gemseo.core.discipline.base_discipline import _CacheType
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.mda.base_mda_settings import BaseMDASettings
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.matplotlib_figure import FigSizeType


LOGGER = logging.getLogger(__name__)


class _BaseMDAProcessFlow(BaseProcessFlow):
    """The process data and execution flow."""

    def get_disciplines_in_data_flow(self) -> list[Discipline]:
        return [self._node]

    def get_execution_flow(self) -> LoopExecSequence:  # noqa:D102
        return LoopExecSequence(self._node, super().get_execution_flow())

    def get_data_flow(  # noqa:D102
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        disciplines = super().get_disciplines_in_data_flow()
        graph = DependencyGraph([self._node, *disciplines])
        res = self._get_disciplines_couplings(graph)
        for discipline in self._node.disciplines:
            res.extend(discipline.get_process_flow().get_data_flow())
        return res

    def _get_disciplines_couplings(
        self, graph: DependencyGraph
    ) -> list[tuple[str, str, list[str]]]:
        """Return the couplings between disciplines.

        Args:
            graph: The dependency graph of the disciplines.

        Returns:
            The couplings between disciplines associated to the edges of the graph.
            For each edge,
            the first component corresponds to the *from* discipline,
            the second component corresponds to the *to* discipline,
            the third component corresponds to the list of coupling variables.
        """
        return graph.get_disciplines_couplings()


class BaseMDA(ProcessDiscipline):
    """A base class for multidisciplinary analysis (MDA)."""

    NORMALIZED_RESIDUAL_NORM: Final[str] = "MDA residuals norm"

    default_cache_type: ClassVar[_CacheType] = ProcessDiscipline.CacheType.SIMPLE

    _linearize_on_last_state: ClassVar[bool] = True
    """Whether to update the local data from the input data before linearizing."""

    # TODO: use generics to handle the type of the settings
    Settings: ClassVar[type[BaseMDASettings]]
    """The Pydantic model for the settings."""

    settings: BaseMDASettings
    """The settings of the MDA."""

    assembly: JacobianAssembly
    """The Jacobian assembly."""

    coupling_structure: CouplingStructure
    """The coupling structure to be used by the MDA."""

    residual_history: list[float]
    """The history of the MDA residuals."""

    reset_history_each_run: bool
    """Whether to reset the history of MDA residuals before each run."""

    _scaling: ResidualScaling
    """The scaling method applied to MDA residuals for convergence monitoring."""

    _scaling_data: float | list[tuple[slice, float]] | NDArray[float] | None
    """The data required to perform the scaling of the MDA residuals."""

    # TODO: API: remove
    norm0: float | None
    """The reference residual, if any."""

    # TODO: API: change to normalized_residual_norm
    normed_residual: float
    """The normed residual."""

    matrix_type: JacobianAssembly.JacobianType
    """The type of the matrix."""

    lin_cache_tol_fact: float
    """The tolerance factor to cache the Jacobian."""

    _starting_indices: list[int]
    """The indices of the residual history where a new execution starts."""

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
        disciplines: Sequence[Discipline],
        settings_model: BaseMDASettings | None = None,
        **settings: Any,
    ) -> None:
        """
        Args:
            disciplines: The disciplines from which to compute the MDA.
            settings_model: The MDA settings as a Pydantic model.
                If ``None``, use ``**settings``.
            **settings: The MDA settings.
                These arguments are ignored when ``settings_model`` is not ``None``.
        """  # noqa:D205 D212 D415
        self.settings = create_model(
            self.Settings,
            settings_model=settings_model,
            **settings,
        )

        super().__init__(disciplines, name=self.settings.name)

        self.coupling_structure = (
            CouplingStructure(disciplines)
            if self.settings.coupling_structure is None
            else self.settings.coupling_structure
        )

        self.assembly = JacobianAssembly(self.coupling_structure)
        self.residual_history = []
        self._starting_indices = []
        self.reset_history_each_run = False

        self._scaling = self.ResidualScaling.INITIAL_RESIDUAL_NORM
        self._scaling_data = None

        # Don't erase coupling values before calling _compute_jacobian

        self.norm0 = None
        self._current_iter = 0
        self.normed_residual = 1.0
        self._input_couplings = []
        self._non_numeric_array_variables = []
        self.matrix_type = JacobianAssembly.JacobianType.MATRIX
        # By default don't use an approximate cache for linearization
        self.lin_cache_tol_fact = 0.0

        self._initialize_grammars()
        self.io.output_grammar.update_from_names([self.NORMALIZED_RESIDUAL_NORM])
        self._check_consistency()
        self.__check_linear_solver_settings()
        self._check_coupling_types()

    @property
    def scaling(self) -> ResidualScaling:
        """The scaling method applied to MDA residuals for convergence monitoring."""
        return self._scaling

    @scaling.setter
    def scaling(self, scaling: ResidualScaling) -> None:
        # This setter will be overloaded in certain child classes.
        self._scaling = scaling

    def _initialize_grammars(self) -> None:
        """Define the grammars as the union of the disciplines' grammars."""
        for discipline in self._disciplines:
            self.io.input_grammar.update(discipline.io.input_grammar)
            self.io.output_grammar.update(discipline.io.output_grammar)

    def __check_linear_solver_settings(self) -> None:
        """Check the linear solver options.

        The linear solver tolerance cannot be set
        using the linear solver option dictionary,
        as it is set using the linear_solver_tolerance keyword argument.

        Raises:
            ValueError: If the ``rtol`` keyword is in :attr:`.linear_solver_settings`.
        """
        if "rtol" in self.settings.linear_solver_settings:
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
                    set(disc.io.input_grammar) & set(disc.io.output_grammar)
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
        for disc in self._disciplines:
            for out in disc.io.output_grammar:
                if out in all_outs:
                    multiple_outs.append(out)
                all_outs[out] = disc

        if multiple_outs:
            LOGGER.warning(
                "The following outputs are defined multiple times: %s.",
                sorted(multiple_outs),
            )

    def _compute_input_coupling_names(self) -> None:
        """Compute the strong couplings that are inputs of the MDA."""
        self._input_couplings = sorted(
            set(self.coupling_structure.strong_couplings).intersection(
                self.io.input_grammar
            )
        )

    def _get_differentiated_io(
        self,
        compute_all_jacobians: bool = False,
    ) -> tuple[set[str] | list[str], set[str] | list[str]]:
        if compute_all_jacobians:
            strong_cpl = set(self.coupling_structure.strong_couplings)
            inputs = set(self.io.input_grammar)
            outputs = self.io.output_grammar
            # Don't linearize wrt
            inputs -= strong_cpl & inputs
            # Don't do this with output couplings because
            # their derivatives wrt design variables may be needed
            # outputs = outputs - (strong_cpl & outputs)
        else:
            inputs, outputs = Discipline._get_differentiated_io(self)

        if self.NORMALIZED_RESIDUAL_NORM in outputs:
            outputs = list(outputs)
            outputs.remove(self.NORMALIZED_RESIDUAL_NORM)

        # Filter the non-numeric arrays
        inputs = [
            input_
            for input_ in inputs
            if self.io.input_grammar.data_converter.is_numeric(input_)
        ]
        outputs = [
            output
            for output in outputs
            if self.io.output_grammar.data_converter.is_numeric(output)
        ]

        return inputs, outputs

    def _check_coupling_types(self) -> None:
        """Check that the coupling variables are numeric.

        If non-numeric array coupling variables are present, they will be filtered and
        not taken into account in the MDA residual. Yet, this method warns the user that
        some of the coupling variables are non-numeric arrays, in case of this event
        follows an improper setup.
        """
        not_arrays = set()
        for coupling_name in self.coupling_structure.all_couplings:
            for discipline in self._disciplines:
                for grammar in (
                    discipline.io.input_grammar,
                    discipline.io.output_grammar,
                ):
                    if (
                        coupling_name in grammar
                        and not grammar.data_converter.is_numeric(coupling_name)
                    ):
                        not_arrays.add(coupling_name)
                        break

        self._non_numeric_array_variables = not_arrays
        if not_arrays:
            LOGGER.debug(
                "The coupling variable(s) %s is/are not an array of numeric values.",
                not_arrays,
            )

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        # Do not re-execute disciplines if inputs error is beyond self tol
        # Apply a safety factor on this (mda is a loop, inputs
        # of first discipline
        # have changed at convergence, therefore the cache is not exactly
        # the same as the current value
        exec_cache_tol = self.lin_cache_tol_fact * self.settings.tolerance
        self.__check_linear_solver_settings()
        residual_variables = {}
        for disc in self._disciplines:
            residual_variables.update(disc.io.residual_to_state_variable)

        couplings_adjoint = sorted(
            set(self.coupling_structure.all_couplings).difference(
                self._non_numeric_array_variables
            )
            - residual_variables.keys()
            - set(residual_variables.values())
        )

        output_names = list(
            set(output_names).difference(self._non_numeric_array_variables)
        )
        input_names = list(
            set(input_names).difference(self._non_numeric_array_variables)
        )

        self.jac = self.assembly.total_derivatives(
            self.io.data,
            output_names,
            input_names,
            couplings_adjoint,
            rtol=self.settings.linear_solver_tolerance,
            mode=self.linearization_mode,
            matrix_type=self.matrix_type,
            use_lu_fact=self.settings.use_lu_fact,
            exec_cache_tol=exec_cache_tol,
            execute=exec_cache_tol == 0.0,
            linear_solver=self.settings.linear_solver,
            residual_variables=residual_variables,
            **self.settings.linear_solver_settings,
        )

    def _prepare_io_for_check_jacobian(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> tuple[Iterable[str], Iterable[str]]:
        # Strong couplings are not linearized.
        input_names, output_names = super()._prepare_io_for_check_jacobian(
            input_names, output_names
        )

        input_names = list(input_names)
        output_names = list(output_names)

        for coupling in self.coupling_structure.all_couplings:
            if coupling in output_names:
                output_names.remove(coupling)
            if coupling in input_names:
                input_names.remove(coupling)

        if self.NORMALIZED_RESIDUAL_NORM in output_names:
            output_names.remove(self.NORMALIZED_RESIDUAL_NORM)

        # Remove non-numeric arrays that cannot be differentiated.
        # TODO: use self._non_numeric_array_variables,
        #       also factorize with _compute_jacobian?
        input_names = [
            input_
            for input_ in input_names
            if self.io.input_grammar.data_converter.is_numeric(input_)
        ]
        output_names = [
            output
            for output in output_names
            if self.io.output_grammar.data_converter.is_numeric(output)
        ]
        return input_names, output_names

    def execute(  # noqa:D102
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> DisciplineData:
        self._current_iter = 0
        return super().execute(input_data=input_data)

    def plot_residual_history(
        self,
        show: bool = False,
        save: bool = True,
        n_iterations: int | None = None,
        logscale: tuple[float, float] = (),
        filename: Path | str = "",
        fig_size: FigSizeType = (),
    ) -> Figure:
        """Generate a plot of the residual history.

        The first iteration of each new execution is marked with a red dot.

        Args:
            show: Whether to display the plot on screen.
            save: Whether to save the plot as a PDF file.
            n_iterations: The number of iterations on the *x* axis.
                If ``None``, use all the iterations.
            logscale: The limits of the *y* axis.
                If empty, do not change the limits of the *y* axis.
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
        colors = [
            "red" if index in self._starting_indices else "black"
            for index in range(n_iterations)
        ]

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
        fig_ax.axhline(y=self.settings.tolerance, c="blue", linewidth=0.5, zorder=0)
        fig_ax.set_title(f"{self.name}: residual plot")

        fig_ax.set_yscale("log")
        fig_ax.set_xlabel(r"iterations", fontsize=14)
        fig_ax.set_xlim([-1, n_iterations])
        fig_ax.get_xaxis().set_major_locator(MaxNLocator(integer=True))
        fig_ax.set_ylabel(r"$\log(||residuals||/||y_0||)$", fontsize=14)
        if logscale:
            fig_ax.set_ylim(logscale)

        if save and not filename:
            filename = f"{self.name}_residual_history.pdf"

        save_show_figure(fig, show, filename, fig_size=fig_size)

        return fig

    def _prepare_warm_start(self) -> None:
        """Load the previous couplings values to local data."""
        cached_outputs = self.cache.last_entry.outputs

        if not cached_outputs:
            return

        # Non simple caches require NumPy arrays.
        if not isinstance(self.cache, SimpleCache):
            to_value = self.io.input_grammar.data_converter.convert_array_to_value
            for input_name, input_value in self.__get_cached_outputs(cached_outputs):
                self.io.update_output_data({
                    input_name: to_value(input_name, input_value)
                })
        else:
            self.io.data.update(dict(self.__get_cached_outputs(cached_outputs)))

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
    def _execute(self) -> None:  # noqa:D103
        if self.settings.warm_start:
            self._prepare_warm_start()

    def _execute_disciplines_and_update_local_data(
        self, input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT
    ) -> None:
        """Execute the disciplines and update the local data with their output data.

        Args:
            input_data: The input data to execute the disciplines.
                If empty, use the local data.
        """
