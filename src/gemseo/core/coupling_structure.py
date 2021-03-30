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
"""
Coupled problem analysis, weak/strong coupling computation using graphs
***********************************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import os
from shutil import move

import matplotlib.pyplot as plt
from future import standard_library
from graphviz import Digraph
from numpy import array
from pylab import gca

from gemseo.utils.py23_compat import string_types

standard_library.install_aliases()


from gemseo import LOGGER


class MDOCouplingStructure(object):
    """Structure of the couplings between disciplines
    The methods of this class include the computation of weak,
    strong or all couplings."""

    def __init__(self, disciplines):
        """
        Constructor

        :param disciplines: list of MDO disciplines that possibly
            exchange coupling variables
        """
        self.disciplines = disciplines
        # generate the directed graph of the disciplines and the
        # resulting execution sequence
        self.graph = DependencyGraph(disciplines)
        self.sequence = self._compute_execution_sequence()

    def _compute_execution_sequence(self):
        """Generates the execution sequence of the disciplines.
        Transforms the sequence of indices into a sequence of
        disciplines."""
        sequence = []
        for parallel_tasks in self.graph.execution_sequence:
            parallel_tasks_disc = []
            for component in parallel_tasks:
                # replace index by corresponding discipline
                component_disc = tuple(self.disciplines[index] for index in component)
                parallel_tasks_disc.append(component_disc)
            sequence.append(parallel_tasks_disc)
        return sequence

    @staticmethod
    def is_self_coupled(discipline):
        """
        Tests if the discipline is self coupled
        ie if one of its outputs is also an input

        :param discipline: the discipline
        :returns: a boolean
        """
        return (
            len(
                set(discipline.get_input_data_names())
                & set(discipline.get_output_data_names())
            )
            > 0
        )

    # methods that determine strong/weak/all couplings

    def strongly_coupled_disciplines(self):
        """Determines the strongly coupled disciplines, that is
        the disciplines that occur in (possibly different) MDAs."""
        strong_disciplines = []
        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find MDAs
                if len(component) > 1:
                    for discipline in component:
                        strong_disciplines.append(discipline)
                else:
                    for discipline in component:
                        if self.is_self_coupled(discipline):
                            strong_disciplines.append(discipline)
        return strong_disciplines

    def weakly_coupled_disciplines(self):
        """Determines the weakly coupled disciplines, that is
        the disciplines that do not occur in MDAs."""
        weak_disciplines = []
        for parallel_tasks in self.sequence:
            for component in parallel_tasks:
                # find single disciplines
                if len(component) == 1 and not self.is_self_coupled(component[0]):
                    weak_disciplines.append(component[0])
        return weak_disciplines

    def strong_couplings(self):
        """Determines the strong couplings = outputs of the strongly
        coupled disciplines that are also inputs of the strongly
        coupled disciplines."""
        strong_disciplines = self.strongly_coupled_disciplines()
        # determine strong couplings = the outputs of the strongly coupled
        # disciplines that are inputs of any other discipline
        strong_couplings = set()
        for strong_discipline in strong_disciplines:
            strong_couplings.update(strong_discipline.get_output_data_names())
        inputs = set()
        for discipline in self.disciplines:
            inputs.update(discipline.get_input_data_names())
        return sorted(list(strong_couplings & inputs))

    def weak_couplings(self):
        """Determines the weak couplings = outputs of the weakly
        coupled disciplines."""
        weak_disciplines = self.weakly_coupled_disciplines()
        # determine strong couplings = the outputs of the weakly coupled
        # disciplines that are inputs of any other discipline
        weak_couplings = set()
        for weak_discipline in weak_disciplines:
            weak_couplings.update(weak_discipline.get_output_data_names())
        return sorted(list(weak_couplings))

    def input_couplings(self, discipline):
        """Computes all input coupling variables of a discipline.

        :param discipline: the discipline
        :returns: the list of input coupling variables names
        """
        input_names = discipline.get_input_data_names()
        strong_couplings = self.strong_couplings()
        return sorted([name for name in input_names if name in strong_couplings])

    def get_all_couplings(self):
        """
        Computes all coupling variables, weak or strong
        :returns: the list of coupling variables names
        """
        inputs = []
        outputs = []
        for discipline in self.disciplines:
            inputs += discipline.get_input_data_names()
            outputs += discipline.get_output_data_names()
        return sorted(list(set(inputs) & set(outputs)))

    def output_couplings(self, discipline, strong=True):
        """Computes the output coupling variables of a discipline.

        :param discipline: the discipline
        :returns: the list of output coupling variables names
        """
        output_names = discipline.get_output_data_names()
        if strong:
            couplings = self.strong_couplings()
        else:
            couplings = self.get_all_couplings()

        return sorted([name for name in output_names if name in couplings])

    def find_discipline(self, output):
        """Finds which discipline produces a given output.

        :param output: the name of the output
        :returns: the discipline if it is found, otherwise raise
            an exception

        """
        if not isinstance(output, string_types):
            raise TypeError("Output shall be a string")
        for discipline in self.disciplines:
            if discipline.is_output_existing(output):
                return discipline
        raise ValueError(output + " is not the output " + "of a discipline")

    def plot_n2_chart(
        self,
        file_path="n2.pdf",
        show_data_names=True,
        save=True,
        show=False,
        figsize=(15, 10),
    ):
        """
        Generates a N2 plot for the disciplines list.

        :param file_path: file path of the figure
        :param show_data_names: if true, the names of the
            coupling data is shown
            otherwise, circles are drawn, which size depend on the
            number of coupling names
        :param save: if True, saved the figure to file_path
        :param show: if True, shows the plot
        """

        fig = plt.figure(figsize=figsize)
        plt.rc("grid", linestyle="-", color="black", lw=1)
        plt.grid(True)
        axe = gca()
        n_disc = len(self.disciplines)
        ax_ticks = list(range(n_disc + 1))
        axe.xaxis.set_ticks(ax_ticks)
        axe.yaxis.set_ticks(ax_ticks)
        axe.xaxis.set_ticklabels([])
        axe.yaxis.set_ticklabels([])
        fig.tight_layout()

        for i, disc_i in enumerate(self.disciplines):
            x_1 = i
            x_2 = i + 1
            y_1 = n_disc - i
            y_2 = n_disc - i - 1
            plt.fill(
                [x_1, x_1, x_2, x_2], [y_1, y_2, y_2, y_1], "limegreen", alpha=0.45
            )
            text = plt.text(
                i + 0.5,
                n_disc - i - 0.5,
                disc_i.name,
                verticalalignment="center",
                horizontalalignment="center",
            )
            self._check_size_text(text, fig, n_disc)

        coupl_tuples = self.graph.get_disciplines_couplings()

        max_cpls = array([len(tpl[2]) for tpl in coupl_tuples]).max()

        for disc1, disc2, c_vars in coupl_tuples:
            i = self.disciplines.index(disc1)
            j = self.disciplines.index(disc2)
            if show_data_names:
                text = plt.text(
                    j + 0.5,
                    n_disc - i - 0.5,
                    "\n".join(c_vars),
                    verticalalignment="center",
                    horizontalalignment="center",
                )
                self._check_size_text(text, fig, n_disc)
            else:
                circle = plt.Circle(
                    (0.5 + j, n_disc - 0.5 - i),
                    len(c_vars) / (3.0 * max_cpls),
                    color="blue",
                )
                axe.add_artist(circle)

        if save:
            plt.savefig(file_path)
        if show:
            plt.show()

    @staticmethod
    def _check_size_text(text, figure, n_disc):
        """
        check the size of the text plotted in the N2 matrix and adapt
        the fig size according to the text shown

        :param text: text shown in the N2 matrix
        :param figure: figure of the n2 matrix
        :param n_disc: number of disciplines to be visible
        """
        renderer = figure.canvas.get_renderer()
        bbox = text.get_window_extent(renderer=renderer)
        width = bbox.width
        height = bbox.height
        size_max_box = figure.get_size_inches() * figure.dpi / n_disc
        inches = figure.get_size_inches()
        if width > size_max_box[0]:
            width_l = round(width * n_disc / figure.dpi) + 1
            figure.set_size_inches(width_l, inches[1])
        if height > size_max_box[1]:
            length_l = round(height * n_disc / figure.dpi) + 1
            figure.set_size_inches(inches[0], length_l)


class DependencyGraph(object):
    """Constructs a graph of dependency between the disciplines, and
    generate a sequence of execution (including strongly coupled
    disciplines, sequential tasks, parallel tasks)."""

    def __init__(self, disciplines):
        """
        Constructor

        :param disciplines: list of disciplines
        """
        self.disciplines = disciplines
        # initial graph
        initial_nodes = self._create_initial_nodes()
        (self.initial_graph, self.initial_edges) = self._compute_graph(initial_nodes)
        # strongly connected components
        self.components = self._strongly_connected_components()
        # resulting reduced graph
        reduced_nodes = self._create_component_nodes(initial_nodes)
        (self.reduced_graph, self.reduced_edges) = self._compute_graph(reduced_nodes)
        self.execution_sequence = self._topological_sort()

    def _create_initial_nodes(self):
        """Creates a list of (input, output) coupling variables
        for each discipline.
        The resulting list has the same order as the disciplines."""
        nodes = []
        for discipline in self.disciplines:
            nodes.append(
                (discipline.get_input_data_names(), discipline.get_output_data_names())
            )
        return nodes

    def _create_component_nodes(self, initial_nodes):
        """Creates a list of (input, output) coupling variables
        for each strongly connected component.
        The resulting list has the same order as the strongly connected
        components.

        :param initial_nodes: nodes of the initial graph
        """
        nodes = []
        for component in self.components:
            component_inputs = set()
            component_outputs = set()
            for node_index in component:
                (inputs, outputs) = initial_nodes[node_index]
                component_inputs.update(inputs)
                component_outputs.update(outputs)
            nodes.append((list(component_inputs), list(component_outputs)))
        return nodes

    @staticmethod
    def _compute_graph(nodes):
        """Computes the successors_i of each node and the edges between
        the nodes.

        :param nodes: the nodes of the graph
        """
        graph = {}
        disc_i = 0
        edges = {}

        for (_, outputs_i) in nodes:
            successors_i = set()
            # find out in which discipline(s) the outputs_i are used
            for output_i in outputs_i:
                disc_j = 0
                for (inputs_j, _) in nodes:
                    if disc_i != disc_j and output_i in inputs_j:
                        successors_i.add(disc_j)
                        # add the edge disc_i -> disc_j with
                        # label output_i
                        if disc_i not in edges:
                            edges[disc_i] = {}
                        if disc_j not in edges[disc_i]:
                            edges[disc_i][disc_j] = []
                        edges[disc_i][disc_j].append(output_i)
                    disc_j += 1

            graph[disc_i] = successors_i
            disc_i += 1
        return graph, edges

    def _strongly_connected_components(self):
        """Tarjan's algorithm determines the strongly connected components of a
        directed initial_graph.
        Within a component, there exists a path between each pair of nodes.

        Based on:
        http://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        """
        stack = []
        comp_inds = {}
        node_indices = {}
        component_list = []  # result

        def strong_connect(node_index, current_index):
            """Browses the initial_graph from a node_index.

            :param node_index: current node index
            :param current_index:
            """
            node_indices[node_index] = current_index
            comp_inds[node_index] = current_index
            current_index += 1
            stack.append(node_index)

            # Consider successors of node_index
            successors = self.initial_graph.get(node_index, [])
            for successor in successors:
                if successor not in comp_inds:
                    # Successor has not yet been visited; recurse on it
                    current_index = strong_connect(successor, current_index)
                    comp_inds[node_index] = min(
                        comp_inds[node_index], comp_inds[successor]
                    )
                elif successor in stack:
                    # the successor is in the stack, therefore in the current
                    # strongly connected component
                    comp_inds[node_index] = min(
                        comp_inds[node_index], node_indices[successor]
                    )

            # If node_index is a root node_index, pop the stack and generate an
            # SCC
            if comp_inds[node_index] == node_indices[node_index]:
                # build the component by popping the stack
                connected_component = self._unstack_component(node_index, stack)
                component_list.append(tuple(connected_component))

            return current_index

        current_index = 0
        for node in self.initial_graph:
            if node not in comp_inds:
                current_index = strong_connect(node, current_index)

        return component_list

    @staticmethod
    def _unstack_component(node, stack):
        """Builds a set of nodes corresponding to a strongly connected
        component.

        :param node: current node
        :param stack: current stack
        """
        # build the component by popping the stack
        connected_component = []
        while True:
            successor = stack.pop()
            connected_component.append(successor)
            if successor == node:
                break
        return connected_component

    # TOPOLOGICAL SORT
    def _topological_sort(self):
        """Computes a topological sort of a directed graph.
        Determines if some nodes may be run in parallel."""
        current_graph = self.reduced_graph.copy()
        result = []
        while True:
            # find the leaves of the initial_graph
            leaves = set()
            for (node_index, successors) in current_graph.items():
                if not successors:
                    leaves.add(node_index)

            if not leaves:
                break
            # all leaves are parallelizable
            parallel_tasks = set(
                self.components[component_index] for component_index in leaves
            )
            result.append(sorted(parallel_tasks))
            # update the current initial_graph by removing the leaves and
            # the initial_edges pointing toward them
            current_graph = {
                node_index: (successors - leaves)
                for (node_index, successors) in current_graph.items()
                if node_index not in leaves
            }
        return result[::-1]

    def get_disciplines_couplings(self):
        """Returns couplings between disciplines as a list of
        3-uples (from_disc, to_disc, variables names set).
        """
        couplings = []
        for from_disc in self.initial_edges:
            for to_disc in self.initial_edges[from_disc]:
                couplings.append(
                    (
                        self.disciplines[from_disc],
                        self.disciplines[to_disc],
                        sorted(self.initial_edges[from_disc][to_disc]),
                    )
                )
        return couplings

    # EXPORT METHODS
    def export_initial_graph(self, file_path):
        """Exports a visualization of the initial graph.

        :param file_path: file path of the generated file
        """
        file_name, file_extension = os.path.splitext(file_path)
        dot = Digraph(comment="Dependency graph", format=file_extension[1:])
        # add the disciplines as nodes
        disc_dict = {}
        for i, disc in enumerate(self.disciplines):
            disc_id = str(i)
            dot.node(disc_id, disc.name)
            disc_dict[disc] = disc_id

        # add the coupling as initial_edges
        for disc_from in self.initial_edges:
            for disc_to in self.initial_edges[disc_from]:
                outputs = self.initial_edges[disc_from][disc_to]
                dot.edge(str(disc_from), str(disc_to), label=",".join(sorted(outputs)))
                # outputs of the last node

        last_tasks = self.execution_sequence[-1]
        for k, last_task in enumerate(last_tasks):
            last_taskind = last_task[0]
            i_str = "-" + str(k)
            disc = self.disciplines[last_taskind]
            last_outputs = disc.get_output_data_names()
            if last_outputs != []:
                # create an edge to an invisible node
                dot.node(i_str, style="invis", shape="point")
                label = ",".join(last_outputs)
                dot.edge(disc_dict[disc], i_str, label=label)

        dot.render(file_name, view=False)
        move(file_name, "{}.gv".format(file_name))

    def export_reduced_graph(self, file_path):
        """Exports a visualization of the reduced graph.

        :param file_path: the file_path of the generated file
        """
        file_name, file_extension = os.path.splitext(file_path)
        dot = Digraph(comment="Dependency graph", format=file_extension[1:])
        # add the disciplines as nodes
        for parallel_tasks in self.execution_sequence:
            for component in parallel_tasks:
                # if MDA, aggregate the names of the disciplines
                if len(component) > 1:
                    disc_names = [self.disciplines[disc].name for disc in component]
                    component_name = "MDA of " + ", ".join(disc_names)
                else:
                    component_name = self.disciplines[component[0]].name
                dot.node(str(component), component_name)

        # outputs of the last nodes

        last_tasks = self.execution_sequence[-1]
        for i, last_task in enumerate(last_tasks):
            last_taskind = last_task[0]
            i_str = "-" + str(i)
            last_outputs = self.disciplines[last_taskind].get_output_data_names()
            if last_outputs != []:
                # create an edge to an invisible node
                dot.node(i_str, style="invis", shape="point")
                label = ",".join(last_outputs)
                dot.edge(str(last_task), i_str, label=label)

        # add the coupling as initial_edges
        for disc_from in self.reduced_edges:
            for disc_to in self.reduced_edges[disc_from]:
                outputs = self.reduced_edges[disc_from][disc_to]
                dot.edge(
                    str(self.components[disc_from]),
                    str(self.components[disc_to]),
                    label=",".join(sorted(outputs)),
                )
        dot.render(file_name, view=False)
