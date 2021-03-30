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
#       :author: Remi Lafage
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Generate XDSMjs input json file
*******************************
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import webbrowser
from builtins import open, str
from json import dumps
from multiprocessing import RLock
from os import getcwd
from os.path import abspath, basename, join, splitext
from tempfile import mkdtemp

from future import standard_library

from gemseo.core.discipline import MDODiscipline
from gemseo.core.execution_sequence import (
    AtomicExecSequence,
    LoopExecSequence,
    ParallelExecSequence,
    SerialExecSequence,
)
from gemseo.core.mdo_scenario import MDOScenarioAdapter
from gemseo.core.monitoring import Monitoring
from gemseo.mda.mda import MDA
from gemseo.utils.locks import synchronized
from gemseo.utils.show_utils import generate_xdsm_html

from .xdsm_tikz import xdsm_dict2tex

standard_library.install_aliases()


from gemseo import LOGGER

OPT_NAME = OPT_ID = "Opt"
USER_NAME = USER_ID = "_U_"


class XDSMizer(object):
    """Class to build the XDSM diagram of a scenario

    Generates input json for XDSMjs javascript library
    https://github.com/OneraHub/XDSMjs

    See :
    Martins, Joaquim RRA, and Andrew B. Lambe.
    "Multidisciplinary design optimization: a survey of architectures."
    AIAA journal 51.9 (2013): 2049-2075.
    """

    def __init__(self, scenario, hashref="root", level=0, expected_workflow=None):
        """
        Constructor

        :param scenario: the MDO scenario to be represented as a XDSM diagram
        :param hashref: key used in the final XDSM json structure to reference
                        {nodes, edges, workflow, optpb} data structure.
                        Default to 'root'
        :param level: level number corresponding to scenario depth.
                      Root scenario is level 0.
        """
        self.scenario = scenario
        self.level = level
        self.hashref = hashref
        self.lock = RLock()
        self._monitor = None
        self.outdir = "."
        self.outfilename = "xdsm.json"
        self.to_hashref = {}
        self.to_id = {}  # dictionary to map AtomicExecSequence to XDSM id
        self.initialize(expected_workflow)
        self.print_statuses = False  # Prints the statuses in the console
        self.latex_output = False

    def initialize(self, workflow=None):
        """
        Initialize from a given workflow or use self.scenario' one,
        create sub XDSM diagram accordingly.

        :param workflow: composite execution sequence
        """
        self.sub_xdsmizers = []
        # Find disciplines from workflow structure
        if workflow:
            self.workflow = workflow
        else:
            self.workflow = self.scenario.get_expected_workflow()
        self.atoms = XDSMizer._get_single_level_atoms(self.workflow)

        self.to_hashref = {}
        level = self.level + 1
        num = 1
        for atom in self.atoms:
            if atom.discipline.is_scenario():
                if atom.discipline == self.scenario:
                    self.to_hashref[atom] = "root"
                    self.root_atom = atom
                else:  # sub-scenario
                    self.to_hashref[atom] = "scn-" + str(level) + "-" + str(num)
                    sub_workflow = XDSMizer._find_sub_workflow(self.workflow, atom)
                    self.sub_xdsmizers.append(
                        XDSMizer(
                            atom.discipline, self.to_hashref[atom], level, sub_workflow
                        )
                    )
                    num += 1

    def monitor(
        self,
        outdir=".",
        outfilename="xdsm.json",
        print_statuses=False,
        latex_output=False,
    ):
        """Monitors |g| discipline execution by generating XDSM json file on
        discipline status update.

        :param str outdir: the directory where XDSM json file is generated
        :param str outfilename: the file name of the generated XDSM json file
            (default: xdsm.json)
        :param bool print_statuses: print the statuses in the console
            at each update (default: False)
        :param bool latex_output: generate tikz, tex and pdf output
            (default: False)
        """
        self._monitor = Monitoring(self.scenario)
        self._monitor.add_observer(self)
        # have to reinitialize with monitored workflow
        self.initialize(self._monitor.workflow)
        self.outdir = outdir
        self.outfilename = outfilename
        self.print_statuses = print_statuses
        self.latex_output = latex_output

    def update(self, atom):  # pylint: disable=unused-argument
        """Callback function that generate new XDSM regarding the given
        atom status update
        :param atom: discipline which status has been updated
        """
        self.run(
            outdir=self.outdir,
            outfilename=self.outfilename,
            latex_output=self.latex_output,
        )
        if self.print_statuses:
            LOGGER.info(str(self._monitor))

    def run(
        self,
        outdir=None,
        latex_output=False,
        outfilename="xdsm.html",
        html_output=True,
        json_output=False,
        open_browser=False,
    ):
        """Generates a XDSM diagram from the process.
        By default, a self contined HTML file is generated, that can be
        viewed in a browser.


        :param outdir: the directory where XDSM json file is generated.
            if None, current working dir is used.
            If open_browser is True and outdir is None, generates a
            temporary directory to store the file
        :param outfilename: the file name of the generated XDSM json file
            (default: xdsm.json).
        :param latex_output: produces .tex, .tikz and .tex files if True
            If ``outdir`` is not set the XDSM json is printed
            on the standard output.
        :param open_browser: if True, opens the web browser with the XDSM
        :param html_output: if True, outputs a self contained HTML file
        :param json_output: if True, outputs a JSON file for XDSMjs

        :returns: XDSM json either in a file when ``outdir`` is set,
                  ouput on the console otherwise.
        """
        xdsm = self.xdsmize()
        xdsm_json = dumps(xdsm, indent=2, ensure_ascii=False)
        base = basename(outfilename)
        outfile_basename = splitext(base)[0]

        no_html_loc = False
        if outdir is None:
            outdir = getcwd()
            no_html_loc = True

        if json_output:
            with open(join(outdir, outfile_basename + ".json"), "w") as out:
                out.write(xdsm_json)
        if latex_output:
            xdsm_dict2tex(xdsm, outdir, outfile_basename)

        if html_output or open_browser:
            if no_html_loc:
                outdir = mkdtemp(suffix="", prefix="tmp", dir=None)
            out_file_path = join(outdir, outfile_basename + ".html")
            LOGGER.info("Generating HTML XDSM file in : %s", out_file_path)
            generate_xdsm_html(xdsm, out_file_path)
            if open_browser:
                url = "file://" + abspath(out_file_path)
                webbrowser.open(url, new=2)  # open in new tab
            return out_file_path

        return xdsm

    def get_all_sub_xdsmizers(self):
        """Retrieves all sub xdsmizers corresponding to sub Scenario objects

        :returns: the array of XDSMizer objects
        """
        result = []
        for sub in self.sub_xdsmizers:
            result.append(sub)
            result.extend(sub.get_all_sub_xdsmizers())
        return result

    @synchronized
    def xdsmize(self, algoname="Optimizer"):
        """Builds the Python data structure to be used to generate JSON
        format compatible with XDSMjs viewer.

        :param algoname: Default value = "Optimizer")
        :returns: the Python object containing relevant information
                    for XDSM representation
        """
        nodes = self._create_nodes(algoname)
        edges = self._create_edges()
        workflow = self._create_workflow()
        optpb = str(self.scenario.formulation.opt_problem)

        if self.level == 0:
            res = {
                self.hashref: {
                    "nodes": nodes,
                    "edges": edges,
                    "workflow": workflow,
                    "optpb": optpb,
                }
            }
            for sub_xdsmizer in self.get_all_sub_xdsmizers():
                res[sub_xdsmizer.hashref] = sub_xdsmizer.xdsmize()
            return res
        return {"nodes": nodes, "edges": edges, "workflow": workflow, "optpb": optpb}

    def _create_nodes(self, algoname):  # pylint: disable=too-many-branches
        """Manages XDSM diagram nodes creation from optimization
        algorithm and disciplines

        :param algoname: name of the algorithm
        """
        nodes = []
        self.to_id = {}

        statuses = self.workflow.get_state_dict()

        # Optimization
        self.to_id[self.root_atom] = OPT_ID
        opt_node = {"id": OPT_ID, "name": algoname, "type": "optimization"}
        if statuses[self.root_atom.uuid]:
            opt_node["status"] = statuses[self.root_atom.uuid]

        nodes.append(opt_node)

        # Disciplines
        for atom_id, atom in enumerate(
            self.atoms
        ):  # pylint: disable=too-many-nested-blocks
            # if a node already created from an atom with same discipline
            # at one level just reference the same node
            for ref_atom in self.to_id:
                if atom.discipline == ref_atom.discipline:
                    self.to_id[atom] = self.to_id[ref_atom]

                    if (
                        atom.status
                        and atom.parent.status is MDODiscipline.STATUS_RUNNING
                    ):

                        node = None
                        for a_node in nodes:
                            if a_node["id"] == self.to_id[atom]:
                                node = a_node
                                break

                        if not node:
                            # TODO: add specific exception?
                            raise "Node " + self.to_id[
                                atom
                            ] + " not found in " + nodes  # pragma: no cover

                        node["status"] = atom.status

                    break

            if atom in self.to_id:
                continue

            self.to_id[atom] = "Dis" + str(atom_id)
            node = {"id": self.to_id[atom], "name": atom.discipline.name}

            # node type
            if isinstance(atom.discipline, MDA):
                node["type"] = "mda"
            elif atom.discipline.is_scenario():
                node["type"] = "mdo"
                node["subxdsm"] = self.to_hashref[atom]
                node["name"] = atom.discipline.name + "_" + self.to_hashref[atom]
            else:
                node["type"] = "analysis"

            if statuses[atom.uuid]:
                node["status"] = statuses[atom.uuid]

            nodes.append(node)

        return nodes

    def _create_edges(self):
        """Manage XDSM edges creation from scenario dataflow."""
        edges = []
        # convenient method to factorize code for creating and appending edges

        def add_edge(from_edge, to_edge, varnames):
            """Adds an edge

            :param from_edge: param to_edge:
            :param varnames:
            :param to_edge:

            """
            edge = {"from": from_edge, "to": to_edge, "name": ", ".join(varnames)}
            edges.append(edge)

        # For User to/from optimization
        opt_pb = self.scenario.formulation.opt_problem

        # fct names such as -y4
        functions_names = opt_pb.get_all_functions_names()

        # output variables used by the fonction (eg y4)
        fct_varnames = [f.outvars for f in opt_pb.get_all_functions()]
        function_varnames = []
        for fvars in fct_varnames:
            function_varnames.extend(fvars)

        to_user = functions_names
        to_opt = self.scenario.get_optim_variables_names()

        add_edge(USER_ID, OPT_ID, [x + "^(0)" for x in to_opt])
        add_edge(OPT_ID, USER_ID, [x + "^*" for x in to_user])

        # Disciplines to/from optimization
        for atom in self.atoms:
            if atom is not self.root_atom:
                varnames = set(atom.discipline.get_input_data_names()) & set(
                    self.scenario.get_optim_variables_names()
                )
                if varnames:
                    add_edge(OPT_ID, self.to_id[atom], varnames)

                varnames = set(atom.discipline.get_output_data_names()) & set(
                    function_varnames
                )
                # print set(disc.get_output_data_names()), set(functions_names)
                if varnames:
                    add_edge(self.to_id[atom], OPT_ID, varnames)

        # Disciplines to User/Optimization (from User is already handled at
        # optimizer level)
        disc_to_opt = function_varnames
        for atom in self.atoms:
            if atom is not self.root_atom:
                # special case MDA : skipped
                if isinstance(atom.discipline, MDA):
                    continue
                out_to_user = [
                    o
                    for o in atom.discipline.get_output_data_names()
                    if o not in disc_to_opt
                ]
                out_to_opt = [
                    o
                    for o in atom.discipline.get_output_data_names()
                    if o in disc_to_opt
                ]
                if out_to_user:
                    add_edge(self.to_id[atom], USER_ID, [x + "^*" for x in out_to_user])
                if out_to_opt:
                    add_edge(self.to_id[atom], OPT_ID, out_to_opt)

        # Disciplines to/from disciplines
        for coupling in self.scenario.get_expected_dataflow():
            (disc1, disc2, varnames) = coupling
            add_edge(
                self.to_id[self._find_atom(disc1)],
                self.to_id[self._find_atom(disc2)],
                varnames,
            )

        return edges

    @staticmethod
    def _get_single_level_atoms(workflow):
        """
        Retrieves the list of atoms of the given workflow without
        looking into loop execution sequences coming from Scenario.
        Thus retrieves the atoms for a one level XDSM diagram.

        :param workflow: execution sequence
        :returns: a list of atoms
        """
        atoms = []
        for seq in workflow.sequence_list:
            if isinstance(seq, LoopExecSequence):
                atoms.append(seq.atom_controller)
                if not seq.atom_controller.discipline.is_scenario():
                    atoms += XDSMizer._get_single_level_atoms(seq.iteration_sequence)
            elif isinstance(seq, AtomicExecSequence):
                atoms.append(seq)
            else:
                atoms += XDSMizer._get_single_level_atoms(seq)
        return atoms

    def _find_atom(self, discipline):
        """
        Find atom corresponding to the given discipline.
        Raise exception if not found

        :param disicpline: an MDODiscipline
        :returns: the corresponding atom
        """
        atom = None
        if isinstance(discipline, MDOScenarioAdapter):
            atom = self._find_atom(discipline.scenario)
        else:
            for atom_i in self.atoms:
                if discipline == atom_i.discipline:
                    atom = atom_i
        if atom is None:
            disciplines = [a.discipline for a in self.atoms]
            raise ValueError(
                "Discipline {} " "not found in {}".format(discipline, disciplines)
            )
        return atom

    @staticmethod
    def _find_sub_workflow(workflow, atom_controller):
        """
        Find loop execution sequence sub_workflow with the given
        atom as controller within the given workflow

        :param atom_controller: the AtomicExecSequence object that controls
        the LoopExecutionSequence object to find
        :returns: the subworkflow (LoopExecutionSequence), None if not found
        """
        sub_workflow = None
        for seq in workflow.sequence_list:
            if isinstance(seq, LoopExecSequence):
                if seq.atom_controller.uuid == atom_controller.uuid:
                    sub_workflow = seq
                    return sub_workflow

                sub_workflow = sub_workflow or XDSMizer._find_sub_workflow(
                    seq.iteration_sequence, atom_controller
                )
            elif not isinstance(seq, AtomicExecSequence):
                sub_workflow = sub_workflow or XDSMizer._find_sub_workflow(
                    seq, atom_controller
                )

        return sub_workflow

    def _create_workflow(self):
        """Manage XDSM workflow creation from formulation workflow"""
        workflow = [USER_ID, expand(self.workflow, self.to_id)]
        return workflow


def expand(wks, to_id):
    """Expands workflow structure as an ids structure using to_id mapping.
    The expansion preserve the structure while replacing the object by its id
    in all case except when a tuple is encountered as cdr then the expansion
    transforms loop[A, (B,C)] in [idA, {'parallel': [idB, idC]}]
    :param wks: the workflow structure
    :param to_id: the mapping dict from object to id
    :returns: the ids structure valid to be used as XDSM json chains
    """
    if isinstance(wks, SerialExecSequence):
        res = []
        for seq in wks.sequence_list:
            res += expand(seq, to_id)
        ids = res
    elif isinstance(wks, ParallelExecSequence):
        res = []
        for seq in wks.sequence_list:
            res += expand(seq, to_id)
        ids = [{"parallel": res}]
    elif isinstance(wks, LoopExecSequence):
        if (
            wks.atom_controller.discipline.is_scenario()
            and to_id[wks.atom_controller] != OPT_ID
        ):
            # sub-scnario consider only the controller
            ids = [to_id[wks.atom_controller]]
        else:
            ids = [to_id[wks.atom_controller], expand(wks.iteration_sequence, to_id)]
    elif isinstance(wks, AtomicExecSequence):
        ids = [to_id[wks]]
    else:
        raise Exception("Bad execution sequence : found " + str(wks))
    return ids
