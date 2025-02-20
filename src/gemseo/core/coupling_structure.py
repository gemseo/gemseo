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
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#          Arthur Piat: greatly improve the N2 layout
"""Graph-based analysis of the weak and strong couplings between several disciplines."""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Literal
from typing import overload

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.dependency_graph import ExecutionSequence
from gemseo.utils.discipline import check_disciplines_consistency
from gemseo.utils.n2d3.n2_html import N2HTML

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    from gemseo.core.discipline import Discipline
    from gemseo.utils.matplotlib_figure import FigSizeType

NodeType = tuple[list[str], list[str]]
EdgesType = dict[int, dict[int, list[str]]]
GraphType = dict[int, set[int]]
ComponentType = tuple[int]


class CouplingStructure:
    """Structure of the couplings between several disciplines.

    The methods of this class include the computation of weak, strong or all couplings.
    """

    disciplines: Sequence[Discipline]
    """The disciplines."""

    graph: DependencyGraph
    """The dependency graph of the disciplines."""

    sequence: ExecutionSequence
    """The sequence of execution of the disciplines."""

    _strongly_coupled_disc: list[Discipline] | None
    _weakly_coupled_disc: list[Discipline] | None
    _all_couplings: list[str] | None
    _weak_couplings: list[str] | None
    _strong_couplings: list[str] | None

    def __init__(
        self,
        disciplines: Sequence[Discipline],
    ) -> None:
        """
        Args:
            disciplines: The disciplines that possibly exchange coupling variables.
        """  # noqa: D205, D212, D415
        check_disciplines_consistency(disciplines, True, False)
        self.disciplines = disciplines
        self.graph = DependencyGraph(disciplines)
        self.sequence = self.graph.get_execution_sequence()

        self._weak_couplings = None
        self._weakly_coupled_disc = None
        self._strong_couplings = None
        self._strongly_coupled_disc = None
        self._all_couplings = None

    @staticmethod
    def is_self_coupled(
        discipline: Discipline,
    ) -> bool:
        """Test if the discipline is self-coupled.

        Self-coupling means that one of its outputs is also an input.

        Args:
            discipline: The discipline.

        Returns:
            Whether the discipline is self-coupled.
        """
        self_c_vars = set(discipline.io.input_grammar) & set(
            discipline.io.output_grammar
        )

        if discipline.io.residual_to_state_variable:
            states = discipline.io.residual_to_state_variable.values()
            self_c_vars -= set(states)
        return len(self_c_vars) > 0

    @property
    def strongly_coupled_disciplines(self) -> list[Discipline]:
        """The disciplines that are strongly coupled.

        The disciplines that lie in cycles in the coupling graphs.
        """
        if self._strongly_coupled_disc is None:
            self._strongly_coupled_disc = self.get_strongly_coupled_disciplines()
        return self._strongly_coupled_disc

    @overload
    def get_strongly_coupled_disciplines(
        self,
        add_self_coupled: bool = ...,
        by_group: Literal[False] = ...,
    ) -> list[Discipline]: ...

    @overload
    def get_strongly_coupled_disciplines(
        self,
        add_self_coupled: bool = ...,
        by_group: Literal[True] = ...,
    ) -> list[tuple[Discipline, ...]]: ...

    @overload
    def get_strongly_coupled_disciplines(
        self,
        add_self_coupled: bool = ...,
        by_group: bool = ...,
    ) -> list[Discipline] | list[tuple[Discipline, ...]]: ...

    # TODO: API: this method is rarely used, remove the kwargs and fix the overloads.
    def get_strongly_coupled_disciplines(
        self,
        add_self_coupled: bool = True,
        by_group: bool = False,
    ) -> list[Discipline] | list[tuple[Discipline, ...]]:
        """Determines the strongly coupled disciplines.

        That is the disciplines that occur in (possibly different) MDAs.

        Args:
            add_self_coupled: Whether to add the disciplines that are self-coupled
                to the list of strongly coupled disciplines.
            by_group: If ``True``,
                returns a list of lists of strongly coupled disciplines
                where the sublist contains the groups of disciplines that
                are strongly coupled together.
                If ``False``, returns a single list.

        Returns:
            The coupled disciplines.
        """
        strong_disciplines: list[tuple[Discipline, ...]] | list[Discipline] = []
        if by_group:
            strong_disc_update = strong_disciplines.append
        else:
            strong_disc_update = strong_disciplines.extend  # type:ignore[assignment] # mypy is confused with this perf trick

        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find MDAs
                if len(component) > 1:
                    strong_disc_update(component)  # type:ignore[arg-type] # mypy is confused with this perf trick
                elif add_self_coupled:
                    for discipline in component:
                        if self.is_self_coupled(discipline):
                            strong_disc_update([discipline])  # type:ignore[arg-type] # mypy is confused with this perf trick

        return strong_disciplines

    @property
    def weakly_coupled_disciplines(self) -> list[Discipline]:
        """The disciplines that do not appear in cycles in the coupling graph."""
        if self._weakly_coupled_disc is None:
            self._compute_weakly_coupled()
        return self._weakly_coupled_disc  # type: ignore[return-value]

    def _compute_weakly_coupled(self) -> None:
        """Determine the weakly coupled disciplines.

        These are the disciplines that do not appear in cycles in the coupling graph.

        Returns:
            The weakly coupled disciplines.
        """
        weak_disciplines = []
        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find single disciplines
                if len(component) == 1 and not self.is_self_coupled(component[0]):
                    weak_disciplines.append(component[0])  # noqa: PERF401
        self._weakly_coupled_disc = weak_disciplines

    @property
    def strong_couplings(self) -> list[str]:
        """The outputs of the strongly coupled disciplines.

        They are also inputs of a strongly coupled discipline.
        """
        if self._strong_couplings is None:
            self._compute_strong_couplings()
        return self._strong_couplings  # type: ignore[return-value]

    def _compute_strong_couplings(self) -> None:
        """Determine the strong couplings.

        These are the outputs of the strongly coupled disciplines that are also inputs
        of the strongly coupled disciplines.
        """
        # determine strong couplings = the outputs of the strongly coupled
        # disciplines that are inputs of any other discipline
        strong_couplings = set()

        for group in self.get_strongly_coupled_disciplines(by_group=True):
            inputs = itertools.chain(*(disc.io.input_grammar for disc in group))
            outputs = itertools.chain(*(disc.io.output_grammar for disc in group))
            strong_couplings.update(set(inputs) & set(outputs))

        self._strong_couplings = sorted(strong_couplings)

    def _compute_weak_couplings(self) -> None:
        """Determine the weak couplings.

        These are the outputs of the weakly coupled disciplines.

        Returns:
            The names of the weakly coupling variables.
        """
        # determine strong couplings = the outputs of the weakly coupled
        # disciplines that are inputs of any other discipline
        weak_couplings: set[str] = set()
        for weak_discipline in self.weakly_coupled_disciplines:
            weak_couplings.update(weak_discipline.io.output_grammar)
        self._weak_couplings = sorted(weak_couplings)

    @property
    def weak_couplings(self) -> list[str]:
        """The outputs of the weakly coupled disciplines."""
        if self._weak_couplings is None:
            self._compute_weak_couplings()
        return self._weak_couplings  # type: ignore[return-value]

    @property
    def all_couplings(self) -> list[str]:
        """The inputs of disciplines that are also outputs of other disciplines."""
        if self._all_couplings is None:
            self._compute_all_couplings()
        return self._all_couplings  # type: ignore[return-value]

    def _compute_all_couplings(self) -> None:
        """Compute the disciplines couplings.

        These are the inputs of disciplines that are also outputs of other disciplines.
        """
        inputs: set[str] = set()
        outputs: set[str] = set()
        for discipline in self.disciplines:
            inputs.update(discipline.io.input_grammar)
            outputs.update(discipline.io.output_grammar)
        self._all_couplings = sorted(inputs & outputs)

    def get_output_couplings(
        self,
        discipline: Discipline,
        strong: bool = True,
    ) -> list[str]:
        """Compute the output coupling variables of a discipline, either strong or weak.

        Args:
            discipline: The discipline.
            strong: If ``True``,
                consider the strong couplings. Otherwise, the weak ones.

        Returns:
            The names of the output coupling variables.
        """
        output_names = discipline.io.output_grammar
        couplings = self.strong_couplings if strong else self.all_couplings
        return sorted(name for name in output_names if name in couplings)

    def get_input_couplings(
        self,
        discipline: Discipline,
        strong: bool = True,
    ) -> list[str]:
        """Compute all the input coupling variables of a discipline.

        Args:
            discipline: The discipline.
            strong: If ``True``,
                consider the strong couplings. Otherwise, the weak ones.

        Returns:
            The names of the input coupling variables.
        """
        input_names = discipline.io.input_grammar
        couplings = self.strong_couplings if strong else self.all_couplings
        return sorted(name for name in input_names if name in couplings)

    def find_discipline(
        self,
        output: str,
    ) -> Discipline:
        """Find which discipline produces a given output.

        Args:
            output: The name of an output.

        Returns:
            The discipline producing this output, if it exists.

        Raises:
            ValueError: If the output is not an output of the discipline.
        """
        for discipline in self.disciplines:
            if output in discipline.io.output_grammar:
                return discipline

        msg = f"{output} is not the output of a discipline."
        raise ValueError(msg)

    def __draw_n2_chart(
        self,
        file_path: str | Path,
        show_data_names: bool,
        save: bool,
        show: bool,
        fig_size: FigSizeType,
    ) -> None:
        """Draw the N2 chart for the disciplines.

        Args:
            file_path: The name of the file path of the figure.
            show_data_names: Whether to show the names of the coupling data;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            save: Whether to save the figure to file_path.
            show: Whether to display the static N2 chart on screen.
            fig_size: The width and height of the figure in inches.
        """
        fig = plt.figure(figsize=fig_size)
        plt.grid(True)
        ax = plt.gca()
        ax.grid(True, linestyle="-", color="black", lw=1)
        n_disciplines = len(self.disciplines)
        ax_ticks = list(range(n_disciplines + 1))
        ax.xaxis.set_ticks(ax_ticks)
        ax.yaxis.set_ticks(ax_ticks)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set(xlim=(0, ax_ticks[-1]), ylim=(0, ax_ticks[-1]))
        ax.tick_params(axis="x", direction="in")
        ax.tick_params(axis="y", direction="in")
        fig.tight_layout()

        self_coupling = {}
        for discipline in self.disciplines:
            coupling_names = set(discipline.io.input_grammar).intersection(
                discipline.io.output_grammar
            )
            if coupling_names:
                self_coupling[discipline] = sorted(coupling_names)

        for discipline_index, discipline in enumerate(self.disciplines):
            x_1 = discipline_index
            x_2 = discipline_index + 1
            y_1 = n_disciplines - discipline_index
            y_2 = n_disciplines - discipline_index - 1
            color = "limegreen"
            self_coupling_variables = self_coupling.get(discipline, [])
            if self_coupling_variables:
                color = "royalblue"

            if not show_data_names:
                self_coupling_variables = []

            plt.fill([x_1, x_1, x_2, x_2], [y_1, y_2, y_2, y_1], color, alpha=0.45)
            discipline_name = plt.text(
                discipline_index + 0.5,
                n_disciplines - discipline_index - 0.5,
                "\n\n".join([discipline.name, *self_coupling_variables]),
                verticalalignment="center",
                horizontalalignment="center",
            )
            self._check_size_text(discipline_name, fig, n_disciplines)

        couplings = self.graph.get_disciplines_couplings()

        if couplings:
            self.__add_couplings(couplings, show_data_names, n_disciplines, fig, ax)

        if save:
            plt.savefig(str(file_path))

        if show:
            plt.show()

    def plot_n2_chart(
        self,
        file_path: str | Path = "n2.pdf",
        show_data_names: bool = True,
        save: bool = True,
        show: bool = False,
        fig_size: FigSizeType = (15.0, 10.0),
        show_html: bool = False,
    ) -> None:
        """Generate a dynamic N2 chart for the disciplines, and possibly a static one.

        A static N2 chart is a figure generated with the matplotlib library
        that can be saved to ``file_path``, displayed on screen or both;
        the extension of ``file_path`` must be recognized by matplotlib.

        A dynamic N2 chart is an HTML file with interactive features such as
        reordering the disciplines,
        expanding or collapsing the groups of strongly coupled disciplines
        and
        displaying information on disciplines or couplings.

        The disciplines are located on the diagonal of the chart
        while the coupling variables are situated on the other blocks
        of the matrix view.
        A coupling variable is outputted by a discipline horizontally
        and enters another vertically.
        On the static chart,
        a blue diagonal block represent a self-coupled discipline,
        i.e. a discipline having some of its outputs as inputs.

        Args:
            file_path: The file path to save the static N2 chart.
            show_data_names: Whether to show the names of the coupling variables
                between two disciplines;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            save: Whether to save the static N2 chart.
            show: Whether to display the static N2 chart on screen.
            fig_size: The width and height of the static N2 chart in inches.
            show_html: Whether to display the interactive N2 chart in a browser.

        Raises:
            ValueError: When there is less than two disciplines.
        """
        if len(self.disciplines) < 2:
            msg = "N2 diagrams need at least two disciplines."
            raise ValueError(msg)

        html_file_path = Path(file_path).parent / "n2.html"
        self_coupled_discipline = [
            discipline.name
            for discipline in self.disciplines
            if self.is_self_coupled(discipline)
        ]
        N2HTML(html_file_path, show_html).from_graph(
            self.graph, self_coupled_discipline
        )

        if save or show:
            self.__draw_n2_chart(file_path, show_data_names, save, show, fig_size)

    @staticmethod
    def _check_size_text(
        text: Text,
        figure: Figure,
        n_disciplines: int,
    ) -> None:
        """Adapt the size of the figure based on the size of the text to display.

        Args:
            text: The text shown in the N2 matrix.
            figure: The figure of the N2 matrix.
            n_disciplines: The number of disciplines to be visible.
        """
        renderer = figure.canvas.get_renderer()  # type: ignore[attr-defined]
        bbox = text.get_window_extent(renderer=renderer)
        width = bbox.width
        height = bbox.height
        size_max_box = figure.get_size_inches() * figure.dpi / n_disciplines
        inches = figure.get_size_inches()
        if width > size_max_box[0]:
            width_l = round(width * n_disciplines / figure.dpi) + 1
            figure.set_size_inches(width_l, inches[1])
        if height > size_max_box[1]:
            length_l = round(height * n_disciplines / figure.dpi) + 1
            figure.set_size_inches(inches[0], length_l)

    def __add_couplings(
        self,
        couplings: Sequence[tuple[Discipline, Discipline, list[str]]],
        show_data_names: bool,
        n_disciplines: int,
        fig: Figure,
        ax: Axes,
    ) -> None:
        """Add the existing couplings to the N2 chart.

        Args:
            couplings: The discipline couplings.
            show_data_names: If ``True``, show the names of the coupling data ;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            n_disciplines: The number of disciplines being considered.
            fig: The figure on which to plot.
            ax: The :class:`~matplotlib.axes.Axes` object associated with ``fig``
                on which to plot.
        """
        max_coupling_size = max(len(variables) for _, _, variables in couplings)

        for source, destination, variables in couplings:
            source_position = self.disciplines.index(source)
            destination_position = self.disciplines.index(destination)
            if show_data_names:
                variable_names = plt.text(
                    destination_position + 0.5,
                    n_disciplines - source_position - 0.5,
                    "\n".join(variables),
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                self._check_size_text(variable_names, fig, n_disciplines)
            else:
                circle = Circle(
                    (
                        0.5 + destination_position,
                        n_disciplines - 0.5 - source_position,
                    ),
                    len(variables) / (3.0 * max_coupling_size),
                    color="royalblue",
                )
                ax.add_artist(circle)
