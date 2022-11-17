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
#       :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Provide routines for XDSM and tikz."""
from __future__ import annotations

from pathlib import Path

from pyxdsm.XDSM import XDSM


class XDSMToPDFConverter:
    """Convert an XDSM to a PDF file with tikz and latex."""

    def __init__(self):  # noqa: D107
        self.__xdsm = XDSM()

    def convert(self, xdsm_data, directory_path, filename_without_ext, scenario):
        """Convert a dictionary representation of a XDSM into a pdf.

        Args:
            xdsm_data: XDSM dictionary representation.
            directory_path (str): output directory.
            filename_without_ext (str): output file name, default is 'xdsm'.
            scenario (str): scenario name, default is 'root'.
            quiet (bool): set to True to suppress output from pdflatex.
        """
        workflow = xdsm_data[scenario]["workflow"][1]
        numbers = {}

        self.__get_numbers(numbers, workflow)
        self.__add_nodes(numbers, xdsm_data[scenario]["nodes"])
        self.__add_edges(xdsm_data[scenario]["edges"])
        self.__add_processes(workflow)

        # workaround xdsm not well handling file path: latex expects posix path
        file_path = str(directory_path / filename_without_ext).replace("\\", "/")

        self.__xdsm.write(file_path)

        if not Path(filename_without_ext).with_suffix(".pdf").exists():
            raise RuntimeError(
                "Something went wrong during the Latex compilation,"
                " as xdsm.pdf has not been generated. Please have a look at the"
                " Latex log file to investigate the root cause of the error."
            )

        # Build XDSM for sub-scenarios
        if scenario == "root":
            subscenarios = [key for key in xdsm_data.keys() if "scn-" in key]
            for subscenario in subscenarios:
                self.convert(
                    xdsm_data,
                    directory_path,
                    filename_without_ext=filename_without_ext + "_" + subscenario,
                    scenario=subscenario,
                )

    def __add_processes(self, workflow, prev=None):
        """Create the pyXDSM processes from a given workflow in a recursive way.

        Args:
            workflow (list): workflow component of the dictionary storing the XDSM.
            prev (str): name of the previous node.
        """
        systems = []
        last_node = None
        for idx, system in enumerate(workflow):
            if isinstance(system, str):
                # system is a node
                systems.append(system)
                last_node = system
            elif isinstance(system, list):
                # system is a group of nodes
                if isinstance(system[0], str):
                    # system[0] is a node
                    if last_node is not None:
                        self.__xdsm.add_process([last_node, system[0]])
                    self.__add_processes(system, workflow[idx - 1])
                elif isinstance(system[0], dict):
                    # system[0] is a parallel sequence
                    for sub_sys in system[0]["parallel"]:
                        # add a sub-process for each parallel element
                        sub_workflow = [last_node, sub_sys]
                        self.__add_processes(sub_workflow, last_node)
        if prev is not None:
            systems.append(prev)
        self.__xdsm.add_process(systems)

    def __get_numbers(self, numbers, nodes, current=0, end=0, following=1):
        """Give number to the different nodes in a recursive way.

        Args:
            numbers: dictionary.
            nodes (list): workflow component of the dictionary storing the XDSM.
            current (int): current step index.
            end (int): end-loop step index.
            following (int): following step index.

        Returns:
            The following step index.
        """
        prev_node = "undefined"
        for node in nodes:
            if isinstance(node, str):
                current = following
                following = current + 1
                end = current
                current_l = [current]
                if node in list(numbers.keys()):
                    current_l = numbers[node]["current"] + current_l
                numbers[node] = {"current": current_l, "next": following, "end": end}
                prev_node = node
            elif isinstance(node, list):
                following = self.__get_numbers(numbers, node, current, end, following)
                numbers[prev_node]["end"] = following
                following += 1
            elif isinstance(node, dict):
                for sub_node in node["parallel"]:
                    if not isinstance(sub_node, list):
                        following = self.__get_numbers(
                            numbers, [sub_node], current, end, following
                        )
                    else:
                        following = self.__get_numbers(
                            numbers, sub_node, current, end, following
                        )
        return following

    def __add_nodes(self, numbers, nodes):
        """Add the different nodes, called 'systems', in the XDSM."""

        for node in nodes:
            name = ",".join(
                [str(current) for current in numbers[node["id"]]["current"]]
            )

            name_1 = name + ",{}-{}:".format(
                str(numbers[node["id"]]["end"]), str(numbers[node["id"]]["next"])
            )
            name_2 = name + ":"

            if node["type"] == "optimization":
                node_type = "Optimization"
                name = name_1
            elif node["type"] == "lp_optimization":
                node_type = "LP_Optimization"
                name = name_1
            if node["type"] == "doe":
                node_type = "DOE"
                name = name_1
            elif node["type"] == "mda":
                node_type = "MDA"
                name = name_1
            elif node["type"] == "mdo":
                node_type = "MDO"
                name = name_2
            elif node["type"] == "analysis":
                node_type = "Function"
                name = name_2
            elif node["type"] == "function":
                node_type = "Function"
                name = name_2
            elif node["type"] == "metamodel":
                node_type = "Metamodel"
                name = name_2

            node_replaced = node["name"]
            escaped_characters = ["_", "$", "&", "{", "}", "%"]
            for char in escaped_characters:
                node_replaced = node_replaced.replace(char, rf"\{char}")
            name = name + node_replaced

            self.__xdsm.add_system(node["id"], node_type, r"\text{" + name + "}")

    def __add_edges(self, edges):
        """Add the edges called connections, inputs, outputs to the XDSM."""

        for edge in edges:
            old_names = edge["name"].split(",")
            names = []

            for name in old_names:
                sub = sup = None
                s_name = name.split("^")
                if len(s_name) == 2:
                    name = s_name[0]
                    suffix = s_name[1]
                    s_name = suffix.split("_")
                    if len(s_name) == 1:
                        s_name = name.split("_")
                        if len(s_name) > 1:
                            sup = suffix
                            sub = s_name[-1]
                            name = "".join(s_name[:-1])
                    else:
                        sub = s_name[0]
                        sup = s_name[1]
                else:
                    s_name = name.split("_")
                    if len(s_name) > 1:
                        sub = s_name[-1]
                        name = "".join(s_name[:-1])

                sup = "" if sup is None else "^{" + sup + "}"
                sub = "" if sub is None else "_{" + sub + "}"
                names.append(name + sub + sup)

            names = ", ".join(names)

            if edge["to"] == "_U_":
                self.__xdsm.add_output(edge["from"], r"" + names + "")
            elif edge["from"] != "_U_":
                self.__xdsm.connect(edge["from"], edge["to"], r"" + names + "")
            elif edge["from"] == "_U_":
                self.__xdsm.add_input("Opt", r"" + names + "")


def xdsm_data_to_pdf(
    xdsm_data, directory_path, filename_without_ext="xdsm", scenario="root"
):
    """Convert a dictionary representation of a XDSM to a pdf.

    Args:
        xdsm_data: XDSM dictionary representation.
        directory_path (str): output directory.
        filename_without_ext (str): output file name, default is 'xdsm'.
        scenario (str): scenario name, default is 'root'.
    """
    converter = XDSMToPDFConverter()
    converter.convert(xdsm_data, directory_path, filename_without_ext, scenario)
