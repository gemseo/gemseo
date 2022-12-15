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
"""Generator of the JSON file defining the coupling structure used by the N2 chart."""
from __future__ import annotations

import json
from typing import Iterable
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING

from jinja2 import Template

if TYPE_CHECKING:
    from gemseo.core.coupling_structure import DependencyGraph
    from gemseo.core.discipline import MDODiscipline


class N2JSON:
    """The JSON structure to be used by the D3.js-based N2 chart."""

    _DEFAULT_GROUP_TEMPLATE = "Group {}"
    _DEFAULT_WEAKLY_COUPLED_DISCIPLINES = "Weakly coupled disciplines"
    __NA = "n/a"

    def __init__(
        self,
        graph: DependencyGraph,
        self_coupled_disciplines: Sequence[str] | None = None,
    ) -> None:
        """
        Args:
            graph: The dependency graph.
            self_coupled_disciplines: The names of the self-coupled disciplines, if any.
        """
        self._graph = graph
        self.__disciplines = list(graph.disciplines)

        data = {}

        self.__disciplines_names = self._get_disciplines_names()

        couplings = self._graph.get_disciplines_couplings()

        groups, n_groups, data["children"] = self._compute_groups(
            self.__disciplines_names
        )

        variables_sizes = self._compute_variables_sizes()

        data["nodes"], data["groups"] = self._create_nodes(
            groups,
            variables_sizes,
            self.__disciplines_names,
            n_groups,
            data["children"],
        )

        data["links"] = self._create_links(
            couplings,
            len(data["nodes"]),
            variables_sizes,
            self.__disciplines_names,
            n_groups,
        )

        data["disciplines"] = self._create_groups_menu_html(
            self.__disciplines_names, data["children"], data["groups"]
        )

        data["self_coupled_disciplines"] = self_coupled_disciplines or []
        self.__json = json.dumps(data, sort_keys=True)

    def __str__(self):
        return self.__json

    @staticmethod
    def _create_variables_html(
        names: Iterable[str],
        variables_sizes: Mapping[str, int] | None = None,
    ) -> str:
        """Generate the HTML representation of variables from their names and sizes.

        Args:
            names: The names of the variables.
            variables_sizes: The sizes of the variables.
                If None, display only the names.

        Return:
            The HTML representation of the sorted variables.
        """
        variables = [
            {
                "name": name,
                "size": None
                if variables_sizes is None
                else variables_sizes.get(name, 1),
            }
            for name in sorted(names)
        ]

        return Template(
            "<div align='center'>"
            "    <table>"
            "    {%- for variable in variables %}"
            "        <tr>"
            "            <td>{{ variable.name }}</td>"
            "            {%- if variable.size is not none -%}"
            "            <td>({{ variable.size }})</td>"
            "            {%- endif %}"
            "        </tr>"
            "    {%- endfor %}"
            "    </table>"
            "</div>"
        ).render(variables=variables)

    @classmethod
    def _create_coupling_html(
        cls,
        source: str,
        destination: str,
        coupling_names: Iterable[str],
        variables_sizes: Mapping[str, int],
    ) -> str:
        """Generate the HTML representation of a bi-disciplinary coupling.

        Args:
            source: The name of the source discipline.
            destination: The name of the destination discipline.
            coupling_names: The names of the coupling variables.
            variables_sizes: The sizes of the variables.

        Returns:
            The HTML block describing this bi-disciplinary coupling.
        """
        return Template(
            "The coupling variables "
            "from <b>{{ source }}</b> "
            "to <b>{{ destination }}</b>:"
            "{{ coupling_variables }}"
        ).render(
            source=source,
            destination=destination,
            coupling_variables=cls._create_variables_html(
                coupling_names, variables_sizes
            ),
        )

    @classmethod
    def _create_discipline_html(
        cls,
        discipline: MDODiscipline,
        variables_sizes: Mapping[str, int],
    ) -> str:
        """Generate the HTML representation of a discipline.

        Args:
            discipline: The discipline.
            variables_sizes: The sizes of the variables.

        Returns:
            The HTML block describing the discipline.
        """
        html_inputs_names = cls._create_variables_html(
            discipline.get_input_data_names(), variables_sizes
        )
        html_outputs_names = cls._create_variables_html(
            discipline.get_output_data_names(), variables_sizes
        )
        return Template(
            "The inputs of <b>{{ discipline }}</b>:"
            "{{ inputs }}"
            "The outputs of <b>{{ discipline }}</b>:"
            "{{ outputs }}"
        ).render(
            discipline=discipline.name,
            inputs=html_inputs_names,
            outputs=html_outputs_names,
        )

    @classmethod
    def _create_group_html(
        cls,
        group: int,
        disciplines: Sequence[str],
        n_groups: int,
        children: Sequence[Sequence[int]],
    ) -> str:
        """Generate the HTML representation of a group of disciplines.

        Args:
            group: The index of the group of disciplines.
            disciplines: The names of the disciplines.
            n_groups: The number of groups.
            children: The indices of the disciplines for the different groups.

        Returns:
            The HTML block describing the group of disciplines.
        """
        disciplines = [disciplines[child - n_groups] for child in children[group]]
        return Template("The disciplines of <b>{{group}}</b>:{{ disciplines }}").render(
            group=cls._DEFAULT_GROUP_TEMPLATE.format(group),
            disciplines=cls._create_variables_html(disciplines),
        )

    @staticmethod
    def _create_groups_menu_html(
        disciplines: Sequence[str],
        children: Sequence[Sequence[int]],
        groups: Sequence[str],
    ) -> str:
        """Generate the HTML representation of the right menu related to the groups.

        Args:
            disciplines: The names of the disciplines.
            children: The indices of the disciplines for the different groups.
            groups: The names of the groups.

        Returns:
            The HTML block used to collapse, expand and visualize groups.
        """
        data = []
        n_groups = len(groups)
        for group_index, disciplines_indices in enumerate(children):
            disciplines_names = [
                disciplines[discipline_index - n_groups]
                for discipline_index in disciplines_indices
            ]
            data.append(
                {
                    "disciplines": sorted(disciplines_names),
                    "group_index": group_index,
                    "group_name": groups[group_index],
                }
            )
        return Template(
            "    <ul class='collapsible'>"
            "        <li>"
            "           <div class='switch'>"
            "              <label>"
            "              <input "
            "type='checkbox' "
            "onclick='expand_collapse_all(json.groups.length,svg);' id='check_all'/>"
            "              <span class='lever'></span>"
            "              </label>All groups"
            "           </div>"
            "       </li>"
            "        {%- for datum in data %}"
            "        <li>"
            "            <div class='collapsible-header'>"
            "               {%- if datum.group_index != 0 %}"
            "               <div class='switch'>"
            "                   <label>"
            "                   <input "
            "type='checkbox' id='check_{{ datum.group_index }}' "
            "onclick='expand_collapse_group({{ datum.group_index }},svg)'/>"
            "                   <span class='lever'></span>"
            "                   </label>"
            "               </div>"
            "               {%- endif %}"
            "               <span id='group_name_{{ datum.group_index }}' "
            "contenteditable='true' "
            "class='group' "
            "onblur='change_group_name(this,{{ datum.group_index }});'>"
            "{{ datum.group_name }}"
            "               </span>"
            "            </div>"
            "            <div class='collapsible-body'>"
            "               {%- for discipline in datum.disciplines %}"
            "               {{ discipline }}{% if not loop.last %},{% endif %}"
            "               {%- endfor %}"
            "            </div>"
            "        </li>"
            "        {%- endfor %}"
            "    </ul>"
        ).render(data=data)

    def _create_links(
        self,
        couplings: Iterable[tuple[MDODiscipline, MDODiscipline, Sequence[str]]],
        n_nodes: int,
        variables_sizes: Mapping[str, int],
        disciplines: Sequence[str],
        n_groups: int,
    ) -> list[dict[str, int | str]]:
        """Create the links.

        Args:
            couplings: The couplings.
            n_nodes: The number of nodes.
            variables_sizes: The sizes of the variables.
            disciplines: The names of the disciplines.
            n_groups: The number of groups.

        Returns:
            The existing links between disciplines,
            defined by a source discipline
            a target discipline,
            a value quantifying the degree of this relationship,
            and an HTML representation.
        """

        links = []
        for link in couplings:
            source, target, variables = link
            if variables:
                source_index = n_groups + disciplines.index(source.name)
                target_index = n_groups + disciplines.index(target.name)
                links.append(
                    {
                        "source": source_index,
                        "target": target_index,
                        "value": len(variables),
                        "description": self._create_coupling_html(
                            source.name, target.name, variables, variables_sizes
                        ),
                    }
                )

        for index in range(n_nodes):
            links.append(
                {
                    "source": index,
                    "target": index,
                    "value": 1,
                    "description": "",
                }
            )

        return links

    def _create_nodes(
        self,
        group: Mapping[str, int],
        variables_sizes: Mapping[str, int],
        disciplines: Sequence[str],
        n_groups: int,
        children: Sequence[Sequence[int]],
    ) -> tuple[list[dict[str, int | str | bool]], list[str]]:
        """Create the nodes representing either a discipline or a disciplines group.

        Args:
            group: The indices of the groups to which the disciplines belong.
            variables_sizes: The sizes of the variables.
            disciplines: The names of the disciplines.
            n_groups: The number of groups.
            children: The indices of the disciplines for the different groups.

        Returns:
            The existing nodes,
            defined by a name,
            a group index,
            an HTML representation
            a target discipline,
            a value quantifying the degree of this relationship,
            an HTML representation,
            and whether the node represents a group.
        """
        disciplines_nodes = [
            {
                "name": discipline.name,
                "group": group[discipline.name],
                "description": self._create_discipline_html(
                    discipline, variables_sizes
                ),
                "is_group": False,
            }
            for discipline in self.__disciplines
        ]

        groups_nodes = [
            {
                "name": self._DEFAULT_GROUP_TEMPLATE.format(group_index),
                "group": group_index,
                "description": self._create_group_html(
                    group_index, disciplines, n_groups, children
                ),
                "is_group": True,
            }
            for group_index in range(n_groups)
        ]

        groups_names = [
            self._DEFAULT_GROUP_TEMPLATE.format(group_index)
            for group_index in range(1, n_groups)
        ]
        groups_names = [self._DEFAULT_WEAKLY_COUPLED_DISCIPLINES] + groups_names

        return groups_nodes + disciplines_nodes, groups_names

    def _compute_groups(
        self,
        disciplines: Sequence[str],
    ) -> tuple[dict[str, int], int, list[list[int]]]:
        """Compute the groups and the children.

        Args:
            disciplines: The names of the disciplines

        Returns:
            The groups to which the disciplines belong,
            the number of groups,
            and the indices of the disciplines for the different groups.
        """
        children = [[]]
        n_groups = 1
        groups = {}
        for parallel_tasks in self._graph.get_execution_sequence():
            for components in parallel_tasks:
                if len(components) > 1:
                    n_groups += 1
                    children.append([])
                    for component in components:
                        index = disciplines.index(component.name)
                        children[-1].append(index)
                        groups[component.name] = n_groups - 1
                else:
                    index = disciplines.index(components[0].name)
                    groups[components[0].name] = 0
                    children[0].append(index)
        indices = []
        new_children = []
        index = 0
        for group in children:
            new_children.append([])
            for child in group:
                new_children[-1].append(index)
                index += 1
                indices.append(child)
        self.__disciplines = [self.__disciplines[index] for index in indices]
        self.__disciplines_names = [
            self.__disciplines_names[index] for index in indices
        ]

        new_children = [[child + n_groups for child in group] for group in new_children]
        return groups, n_groups, new_children

    def _compute_variables_sizes(self) -> dict[str, int]:
        """Compute the sizes of the coupling variables.

        Returns:
            The names of the coupling variables bound to their sizes.
        """
        variables_sizes = {}
        for discipline in self.__disciplines:
            for name in discipline.get_input_data_names():
                if name not in variables_sizes or variables_sizes[name] == self.__NA:
                    default_value = discipline.default_inputs.get(name)
                    if default_value is not None:
                        size = default_value.size
                    else:
                        size = self.__NA
                    variables_sizes[name] = size

        return variables_sizes

    def _get_disciplines_names(self) -> list[str]:
        """Return the names of the disciplines."""
        return [discipline.name for discipline in self.__disciplines]
