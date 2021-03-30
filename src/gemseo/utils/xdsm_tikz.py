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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Provide routines for XDSM and tikz."""

from __future__ import absolute_import, division, unicode_literals

import os
from pathlib import Path

from future import standard_library

from gemseo.third_party.pyxdsm.XDSM import XDSM
from gemseo.utils.py23_compat import string_types

standard_library.install_aliases()


def __add_processes(xdsm, workflow, prev=None):
    """Create the pyXDSM processes from a given workflow
    in a recursive way.

    :param XDSM xdsm: XDSM object
    :param list workflow: workflow component of the dictionary
        storing the XDSM
    :param str prev: name of the previous node
    """
    systems = []
    last_node = None
    for idx, system in enumerate(workflow):
        if isinstance(system, string_types):
            # system is a node
            systems.append(system)
            last_node = system
        elif isinstance(system, list):
            # system is a group of nodes
            if isinstance(system[0], string_types):
                # system[0] is a node
                if last_node is not None:
                    xdsm.add_process([last_node, system[0]], arrow=True)
                __add_processes(xdsm, system, workflow[idx - 1])
            elif isinstance(system[0], dict):
                # system[0] is a parallel sequence
                for sub_sys in system[0]["parallel"]:
                    # add a sub-process for each parallel element
                    sub_workflow = [last_node, sub_sys]
                    __add_processes(xdsm, sub_workflow, last_node)
    if prev is not None:
        systems.append(prev)
    xdsm.add_process(systems, arrow=True)


def __get_numbers(numbers, nodes, current=0, end=0, following=1, prev_node=None):
    """Give number to the different nodes in a recursive way.

    :param numbers: dictionary
    :param list nodes: workflow component of the dictionary
        storing the XDSM
    :param int current: current step index
    :param int end: end-loop step index
    :param int following: following step index
    """
    for node in nodes:
        if isinstance(node, string_types):
            current = following
            following = current + 1
            end = current
            current_l = [current]
            if node in list(numbers.keys()):
                current_l = numbers[node]["current"] + current_l
            numbers[node] = {"current": current_l, "next": following, "end": end}
            prev_node = node
        elif isinstance(node, list):
            following = __get_numbers(numbers, node, current, end, following, prev_node)
            numbers[prev_node]["end"] = following
            following += 1
        elif isinstance(node, dict):
            for sub_nodes in node["parallel"]:
                if not isinstance(sub_nodes, list):
                    following = __get_numbers(
                        numbers, [sub_nodes], current, end, following
                    )
                else:
                    following = __get_numbers(
                        numbers, sub_nodes, current, end, following
                    )
                following -= 1
            following += 1
    return following


def __add_nodes(xdsm, numbers, nodes):
    """Add the different nodes, called 'systems', in the XDSM."""
    for node in nodes:
        name = ",".join([str(current) for current in numbers[node["id"]]["current"]])

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
            if node["name"] == "MDAChain":
                name = name_2
            else:
                name = name_1
        elif node["type"] == "mdo":
            node_type = "MDO"
            name = name_2
        elif node["type"] == "analysis":
            node_type = "Analysis"
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
            node_replaced = node_replaced.replace(char, r"\{}".format(char))
        name = name + node_replaced
        xdsm.add_system(node["id"], node_type, r"\text{" + name + "}")


def __add_edges(xdsm, edges):
    """Add the edges called connections, inputs, outputs to the XDSM."""
    for edge in edges:
        old_names = edge["name"].split(",")
        names = []

        for name in old_names:
            name = name.replace("(0)", "{(0)}")
            if "*" in name:
                if "_" in name:
                    name = name.replace("^*", "")
                    name = name.replace("_", "^{*}_{")
            elif "(0)" in name:
                if "_" in name:
                    name = name.replace("(0)", "")
                    name = name.replace("_", "^{(0)}_{")
                else:
                    name = name.replace("(0)", "{(0)}")
            else:
                name = name.replace("_", "_{")
            name += "}" * name.count("_")
            names.append(name)

        names = ", ".join(names)

        if edge["to"] == "_U_":
            xdsm.add_output(edge["from"], r"" + names + "", side="left")
        elif edge["from"] != "_U_":
            xdsm.connect(edge["from"], edge["to"], r"" + names + "")
        elif edge["from"] == "_U_":
            xdsm.add_input("Opt", r"" + names + "")


def xdsm_dict2tex(dict_xdsm, out_dir, out_filename="xdsm", scenario="root", quiet=True):
    """Convert a dictionary representation of a XDSM
    into a pdf or tikz representation.

    :param dict xdsm: XDSM dictionary representation
    :param str out_dir: output directory
    :param str out_filename: output file name (default: 'xdsm')
    :param str scenario: scenario name (default: 'root')
    :param bool quiet: set to True to suppress output from pdflatex.
    """
    workflow = dict_xdsm[scenario]["workflow"][1]

    numbers = {}
    __get_numbers(numbers, workflow)

    xdsm = XDSM()

    __add_nodes(xdsm, numbers, dict_xdsm[scenario]["nodes"])
    __add_edges(xdsm, dict_xdsm[scenario]["edges"])
    __add_processes(xdsm, workflow)

    out_texfile = os.path.join(out_dir, out_filename)
    out_xdsm = os.path.join(out_dir, out_filename) + ".pdf"
    xdsm.write(out_texfile, quiet=quiet)

    if not Path(out_xdsm).exists():
        raise RuntimeError(
            "Something went wrong during the Latex compilation,"
            " as xdsm.pdf has not been generated. Please have a look at the Latex log file"
            " to investigate the root cause of the error."
        )

    # Build XDSM for sub-scenarios
    if scenario == "root":
        subscenarios = [key for key in dict_xdsm.keys() if key.startswith("scn-")]
        for subscenario in subscenarios:
            xdsm_dict2tex(
                dict_xdsm,
                out_dir,
                out_filename=out_filename + "_" + subscenario,
                scenario=subscenario,
                quiet=quiet,
            )
    return numbers
