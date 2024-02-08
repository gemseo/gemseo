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
from typing import TYPE_CHECKING
from typing import Any

from pyxdsm.XDSM import XDSM

if TYPE_CHECKING:
    from collections.abc import Sequence


class XDSMToPDFConverter:
    """Convert an XDSM to a PDF file with tikz and latex."""

    def __init__(self) -> None:  # noqa: D107
        self.__xdsm = XDSM()

    def convert(
        self,
        xdsm_data: dict[str, Any],
        directory_path: str | Path,
        file_name: str,
        scenario: str,
        build: bool = True,
        cleanup: bool = True,
        batchmode: bool = True,
    ) -> None:
        """Convert a dictionary representation of a XDSM into a pdf.

        Args:
            xdsm_data: The XDSM representation.
            directory_path: The path to the output directory where the pdf is generated.
            file_name: The name of the output file.
            scenario: The name of the scenario.
            build: Whether the standalone pdf of the XDSM will be built.
            cleanup: Whether pdflatex built files will be cleaned up
                after build is complete.
            batchmode: Whether pdflatex is run in `batchmode`.
        """
        workflow = xdsm_data[scenario]["workflow"][1]
        numbers = {}

        self.__get_numbers(numbers, workflow)
        self.__add_nodes(numbers, xdsm_data[scenario]["nodes"])
        self.__add_edges(xdsm_data[scenario]["edges"])
        self.__add_processes(workflow)

        self.__xdsm.write(
            file_name,
            outdir=str(directory_path),
            build=build,
            cleanup=cleanup,
            quiet=batchmode,
        )

        if build and (
            not (Path(directory_path) / file_name).with_suffix(".pdf").exists()
        ):
            msg = (
                "Something went wrong during the Latex compilation,"
                " as xdsm.pdf has not been generated. Please have a look at the"
                " Latex log file to investigate the root cause of the error."
            )
            raise RuntimeError(msg)

        # Build XDSM for sub-scenarios
        if scenario == "root":
            subscenarios = [key for key in xdsm_data if "scn-" in key]
            for subscenario in subscenarios:
                sub_xdsm = XDSMToPDFConverter()
                sub_xdsm.convert(
                    xdsm_data,
                    directory_path,
                    file_name=f"{file_name}_{subscenario}",
                    scenario=subscenario,
                    build=build,
                    cleanup=cleanup,
                    batchmode=batchmode,
                )

    def __add_processes(
        self, workflow: list[Any], prev: str | list[str] | None = None
    ) -> None:
        """Create the pyXDSM processes from a given workflow in a recursive way.

        Args:
            workflow: The workflow component of the dictionary storing the XDSM.
            prev: The name of the previous node.
        """
        last_node = prev
        for system in workflow:
            if isinstance(system, str):  # system is a node
                if isinstance(last_node, list):  # case of previous parallel nodes
                    for node in last_node:
                        self.__xdsm.add_process([node, system])
                else:
                    if last_node:
                        self.__xdsm.add_process([last_node, system])
                last_node = system
            elif isinstance(system, list):  # system is a group of nodes (MDA, chain...)
                self.__add_processes(system, last_node)
            elif isinstance(system, dict):  # system is parallel
                last_nodes = []
                for sub_sys in system["parallel"]:
                    # add a sub-process for each parallel element
                    if isinstance(sub_sys, str):
                        sub_workflow = [last_node, sub_sys]
                    elif isinstance(sub_sys, list):
                        sub_workflow = [last_node, *sub_sys]
                    self.__add_processes(sub_workflow)

                    if isinstance(sub_sys, list):
                        # take the last node when it is a chain (e.g. [d1, d2, d2...]),
                        # but take the first node when it is an iterative struct
                        # (e.g. MDA, [d1, [d2, d3]])
                        last_nodes.append(
                            sub_sys[0] if isinstance(sub_sys[-1], list) else sub_sys[-1]
                        )
                    else:
                        last_nodes.append(sub_sys)
                last_node = last_nodes

        # return loop form last node to previous node
        if prev:
            last_node = [last_node] if not isinstance(last_node, list) else last_node
            for node in last_node:
                self.__xdsm.add_process([node, prev])

    def __get_numbers(
        self,
        numbers: Sequence[int],
        nodes,
        current: int = 0,
        end: int = 0,
        following: int = 1,
    ):
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
                init_following = following
                followings = []
                for sub_node in node["parallel"]:
                    if not isinstance(sub_node, list):
                        followings.append(
                            self.__get_numbers(
                                numbers, [sub_node], current, end, init_following
                            )
                        )
                    else:
                        followings.append(
                            self.__get_numbers(
                                numbers, sub_node, current, end, init_following
                            )
                        )
                following = max(followings)
        return following

    def __add_nodes(self, numbers: Sequence[int], nodes) -> None:
        """Add the different nodes, called 'systems', in the XDSM."""
        for node in nodes:
            name = ",".join([
                str(current) for current in numbers[node["id"]]["current"]
            ])

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
            elif node["type"] == "doe":
                node_type = "DOE"
                name = name_1
            elif node["type"] == "mda":
                node_type = "MDA"
                name = name_1
            elif node["type"] == "mdo":
                node_type = "SubOptimization"
                name = name_2
            elif node["type"] in {"analysis", "function"}:
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

    def __add_edges(self, edges) -> None:
        """Add the edges called connections, inputs, outputs to the XDSM."""
        for edge in edges:
            old_names = edge["name"].split(",")
            names = [
                name.replace("_", r"\_").replace("(0)", "{(0)}") for name in old_names
            ]

            if len(names) > 2:
                names = f"{', '.join(names[:2])}..., ({len(names)})"
            else:
                names = ", ".join(names)

            if edge["to"] == "_U_":
                self.__xdsm.add_output(edge["from"], r"" + names + "")
            elif edge["from"] != "_U_":
                self.__xdsm.connect(edge["from"], edge["to"], r"" + names + "")
            elif edge["from"] == "_U_":
                self.__xdsm.add_input("Opt", r"" + names + "")


def xdsm_data_to_pdf(
    xdsm_data: dict[str, Any],
    directory_path: Path | str,
    file_name: str = "xdsm",
    scenario: str = "root",
    pdf_build: bool = True,
    pdf_cleanup: bool = True,
    pdf_batchmode: bool = True,
) -> None:
    """Convert a dictionary representation of a XDSM to a pdf.

    Args:
        xdsm_data: The XDSM representation.
        directory_path: The output directory where the pdf is generated.
        file_name: The output file name (without extension).
        scenario: The name of the scenario name.
        pdf_build: Whether the standalone pdf of the XDSM will be built.
        pdf_cleanup: Whether pdflatex built files will be cleaned up
            after build is complete.
        pdf_batchmode: Whether pdflatex is run in `batchmode`.
    """
    converter = XDSMToPDFConverter()
    converter.convert(
        xdsm_data,
        str(directory_path),
        file_name,
        scenario,
        pdf_build,
        pdf_cleanup,
        pdf_batchmode,
    )
