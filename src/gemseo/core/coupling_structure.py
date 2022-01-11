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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#          Arthur Piat: greatly improve the N2 layout
"""Graph-based analysis of the weak and strong couplings between several disciplines."""

from __future__ import unicode_literals

import itertools
import logging
from typing import Dict, List, Sequence, Set, Tuple, Union

import matplotlib.pyplot as plt
from matplotlib.figure import Axes, Figure
from matplotlib.text import Text
from pylab import gca

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.discipline import MDODiscipline
from gemseo.utils.n2d3.n2_html import N2HTML
from gemseo.utils.py23_compat import Path, string_types

LOGGER = logging.getLogger(__name__)

NodeType = Tuple[List[str], List[str]]
EdgesType = Dict[int, Dict[int, List[str]]]
GraphType = Dict[int, Set[int]]
ComponentType = Tuple[int]


class MDOCouplingStructure(object):
    """Structure of the couplings between several disciplines.

    The methods of this class include the computation of weak, strong or all couplings.

    Attributes:
        disciplines (Sequence[MDODiscipline]): The disciplines.
        graph (DependencyGraph): The directed graph of the disciplines.
        sequence (List[List[Tuple[MDODiscipline]]]): The sequence of execution
            of the disciplines.
    """

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
    ):  # type: (...) -> None
        """
        Args:
            disciplines: The disciplines that possibly exchange coupling variables.
        """
        self.disciplines = disciplines
        self.graph = DependencyGraph(disciplines)
        self.sequence = self.graph.get_execution_sequence()

    @staticmethod
    def is_self_coupled(
        discipline,  # type: MDODiscipline
    ):  # type: (...) -> bool
        """Test if the discipline is self-coupled.

        Self-coupling means that one of its outputs is also an input.

        Args:
            discipline: The discipline.

        Returns:
            Whether the discipline is self-coupled.
        """
        return (
            len(
                set(discipline.get_input_data_names())
                & set(discipline.get_output_data_names())
            )
            > 0
        )

    # methods that determine strong/weak/all couplings
    def strongly_coupled_disciplines(
        self,
        add_self_coupled=True,  # type: bool
        by_group=False,  # type: bool
    ):  # type: (...) -> Union[List[MDODiscipline],List[List[MDODiscipline]]]
        """Determines the strongly coupled disciplines, that is the disciplines that
        occur in (possibly different) MDAs.

        Args:
            add_self_coupled: if True, adds the disciplines that are self-coupled
                to the list of strongly coupled disciplines
            by_group: if True, returns a list of list of strongly coupled diciplines
                where the sublists contains the groups of disciplines that
                are strongly coupled together.
                if False, returns a single list

        Returns:
            The coupled disciplines list or list of list
        """
        strong_disciplines = []
        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find MDAs
                if len(component) > 1:
                    if by_group:
                        strong_disciplines.append(component)
                    else:
                        strong_disciplines += component
                elif add_self_coupled:
                    for discipline in component:
                        if self.is_self_coupled(discipline):
                            if by_group:
                                strong_disciplines.append([discipline])
                            else:
                                strong_disciplines.append(discipline)
        return strong_disciplines

    def weakly_coupled_disciplines(self):  # type: (...) -> List[MDODiscipline]
        """Determine the weakly coupled disciplines.

        These are the disciplines that do not occur in MDAs.

        Returns:
            The weakly coupled disciplines.
        """
        weak_disciplines = []
        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find single disciplines
                if len(component) == 1 and not self.is_self_coupled(component[0]):
                    weak_disciplines.append(component[0])
        return weak_disciplines

    def strong_couplings(self):  # type: (...) -> List[str]
        """Determine the strong couplings.

        These are the outputs of the strongly coupled disciplines
        that are also inputs of the strongly coupled disciplines.

        Returns:
            The names of the strongly coupling variables.
        """
        strong_disciplines = self.strongly_coupled_disciplines(
            add_self_coupled=True, by_group=True
        )
        # determine strong couplings = the outputs of the strongly coupled
        # disciplines that are inputs of any other discipline
        strong_couplings = set()

        for group in strong_disciplines:
            inputs = itertools.chain(*[disc.get_input_data_names() for disc in group])
            outputs = itertools.chain(*[disc.get_output_data_names() for disc in group])
            strong_couplings.update(set(inputs) & set(outputs))

        return sorted(list(strong_couplings))

    def weak_couplings(self):  # type: (...) -> List[str]
        """Determine the weak couplings.

        These are the outputs of the weakly coupled disciplines.

        Returns:
            The names of the weakly coupling variables.
        """
        weak_disciplines = self.weakly_coupled_disciplines()
        # determine strong couplings = the outputs of the weakly coupled
        # disciplines that are inputs of any other discipline
        weak_couplings = set()
        for weak_discipline in weak_disciplines:
            weak_couplings.update(weak_discipline.get_output_data_names())
        return sorted(list(weak_couplings))

    def input_couplings(
        self,
        discipline,  # type: MDODiscipline
    ):  # type: (...) -> List[str]
        """Compute all the input coupling variables of a discipline.

        Args:
            discipline: The discipline.

        Returns:
            The names of the input coupling variables.
        """
        input_names = discipline.get_input_data_names()
        strong_couplings = self.strong_couplings()
        return sorted([name for name in input_names if name in strong_couplings])

    def get_all_couplings(self):  # type: (...) -> List[str]
        """Compute all the weak and strong coupling variables.

        Returns:
            The names of the coupling variables.
        """
        inputs = []
        outputs = []
        for discipline in self.disciplines:
            inputs += discipline.get_input_data_names()
            outputs += discipline.get_output_data_names()
        return sorted(list(set(inputs) & set(outputs)))

    def output_couplings(
        self,
        discipline,  # type: MDODiscipline
        strong=True,  # type: bool
    ):  # type: (...) -> List[str]
        """Compute the output coupling variables of a discipline, either strong or weak.

        Args:
            discipline: The discipline.
            strong: If True, consider the strong couplings. Otherwise, the weak ones.

        Returns:
            The names of the output coupling variables.
        """
        output_names = discipline.get_output_data_names()
        if strong:
            couplings = self.strong_couplings()
        else:
            couplings = self.get_all_couplings()

        return sorted([name for name in output_names if name in couplings])

    def find_discipline(
        self,
        output,  # type: str
    ):  # type: (...) -> MDODiscipline
        """Find which discipline produces a given output.

        Args:
            output: The name of an output.

        Returns:
            The discipline producing this output, if it exists.

        Raises:
            TypeError: If the name of the output is not a string.
            ValueError: If the output is not an output of the discipline.
        """
        if not isinstance(output, string_types):
            raise TypeError("Output shall be a string")

        for discipline in self.disciplines:
            if discipline.is_output_existing(output):
                return discipline

        raise ValueError("{} is not the output of a discipline".format(output))

    def __draw_n2_chart(
        self,
        file_path,  # type: Union[str,Path]
        show_data_names,  # type: True
        save,  # type:bool
        show,  # type: bool
        figsize,  # type: Tuple[int]
    ):  # type: (...) -> None
        """Draw the N2 chart for the disciplines.

        Args:
            file_path: The name of the file path of the figure.
            show_data_names: Whether to show the names of the coupling data;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            save: Whether to save the figure to file_path.
            show: Whether to display the static N2 chart on screen.
            figsize: The width and height of the figure in inches.
        """
        fig = plt.figure(figsize=figsize)
        plt.grid(True)
        axe = gca()
        axe.grid(True, linestyle="-", color="black", lw=1)
        n_disciplines = len(self.disciplines)
        ax_ticks = list(range(n_disciplines + 1))
        axe.xaxis.set_ticks(ax_ticks)
        axe.yaxis.set_ticks(ax_ticks)
        axe.xaxis.set_ticklabels([])
        axe.yaxis.set_ticklabels([])
        axe.set(xlim=(0, ax_ticks[-1]), ylim=(0, ax_ticks[-1]))
        axe.tick_params(axis="x", direction="in")
        axe.tick_params(axis="y", direction="in")
        fig.tight_layout()

        for discipline_index, discipline in enumerate(self.disciplines):
            x_1 = discipline_index
            x_2 = discipline_index + 1
            y_1 = n_disciplines - discipline_index
            y_2 = n_disciplines - discipline_index - 1
            plt.fill(
                [x_1, x_1, x_2, x_2], [y_1, y_2, y_2, y_1], "limegreen", alpha=0.45
            )
            discipline_name = plt.text(
                discipline_index + 0.5,
                n_disciplines - discipline_index - 0.5,
                discipline.name,
                verticalalignment="center",
                horizontalalignment="center",
            )
            self._check_size_text(discipline_name, fig, n_disciplines)

        couplings = self.graph.get_disciplines_couplings()

        if couplings:
            self.__add_couplings(couplings, show_data_names, n_disciplines, fig, axe)

        if save:
            plt.savefig(str(file_path))

        if show:
            plt.show()

    def plot_n2_chart(
        self,
        file_path="n2.pdf",  # type: Union[str,Path]
        show_data_names=True,  # type:bool
        save=True,  # type:bool
        show=False,  # type: bool
        figsize=(15, 10),  # type: Tuple[int]
        open_browser=False,  # type:bool
    ):  # type: (...) -> None
        """Generate a dynamic N2 chart for the disciplines, and possibly a static one.

        A static N2 chart is a figure generated with the matplotlib library
        that can be saved to ``file_path``, displayed on screen or both;
        the extension of ``file_path`` must be recognized by matplotlib.

        A dynamic N2 chart is a HTML file with interactive features such as
        reordering the disciplines,
        expanding or collapsing the groups of strongly coupled disciplines
        and
        displaying information on disciplines or couplings.

        Args:
            file_path: The file path to save the static N2 chart.
            show_data_names: Whether to show the names of the coupling data ;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            save: Whether to save the static N2 chart.
            show: Whether to display the static N2 chart on screen.
            figsize: The width and height of the static N2 chart in inches.
            open_browser: Whether to display the interactive N2 chart in a browser.

        Raises:
            ValueError: When there is less than two disciplines.
        """
        if len(self.disciplines) < 2:
            raise ValueError("N2 diagrams need at least two disciplines.")

        html_file_path = Path(file_path).parent / "n2.html"
        N2HTML(html_file_path, open_browser).from_graph(self.graph)

        if save or show:
            self.__draw_n2_chart(file_path, show_data_names, save, show, figsize)

    @staticmethod
    def _check_size_text(
        text,  # type: Text
        figure,  # type: Figure
        n_disciplines,  # type: int
    ):  # type: (...)-> None
        """Adapt the size of the figure based on the size of the text to display.

        Args:
            text: The text shown in the N2 matrix.
            figure: The figure of the N2 matrix.
            n_disciplines: The number of disciplines to be visible.
        """
        renderer = figure.canvas.get_renderer()
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
        couplings,  # type: Sequence[Tuple[MDODiscipline, MDODiscipline, List[str]]]
        show_data_names,  # type: bool
        n_disciplines,  # type: int
        fig,  # type: Figure
        axe,  # type: Axes
    ):  # type: (...) -> None
        """Add the existing couplings to the N2 chart.

        Args:
            couplings: The discipline couplings.
            show_data_names: If ``True``, show the names of the coupling data ;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            n_disciplines: The number of disciplines being considered.
            fig: The figure where the couplings will be added.
            axe: The axes of the figure.
        """

        max_coupling_size = max([len(variables) for _, _, variables in couplings])

        for source, destination, variables in couplings:
            source_position = self.disciplines.index(source)
            destination_position = self.disciplines.index(destination)
            if show_data_names:
                variables_names = plt.text(
                    destination_position + 0.5,
                    n_disciplines - source_position - 0.5,
                    "\n".join(variables),
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                self._check_size_text(variables_names, fig, n_disciplines)
            else:
                circle = plt.Circle(
                    (
                        0.5 + destination_position,
                        n_disciplines - 0.5 - source_position,
                    ),
                    len(variables) / (3.0 * max_coupling_size),
                    color="blue",
                )
                axe.add_artist(circle)
