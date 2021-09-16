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
"""Creation of a XDSM diagram from a scenario.

The :class:`.XDSMizer` generates a JSON file.

The latter is used by
the `XDSMjs javascript library <https://github.com/OneraHub/XDSMjs>`_
to produce an interactive web XDSM
and by the pyxdsm python library
to produce TIKZ and PDF versions of the XDSM.

For more information, see:
A. B. Lambe and J. R. R. A. Martins, “Extensions to the Design Structure Matrix for
the Description of Multidisciplinary Design, Analysis, and Optimization Processes”,
Structural and Multidisciplinary Optimization, vol. 46, no. 2, p. 273-284, 2012.
"""
from __future__ import division, unicode_literals

import logging
import webbrowser
from json import dumps
from multiprocessing import RLock
from os.path import basename, splitext
from tempfile import mkdtemp
from typing import Any, Dict, List, Mapping, Optional, Union

from gemseo.core.discipline import MDODiscipline
from gemseo.core.doe_scenario import DOEScenario
from gemseo.core.execution_sequence import (
    AtomicExecSequence,
    CompositeExecSequence,
    LoopExecSequence,
    ParallelExecSequence,
    SerialExecSequence,
)
from gemseo.core.mdo_scenario import MDOScenario, MDOScenarioAdapter
from gemseo.core.monitoring import Monitoring
from gemseo.core.scenario import Scenario
from gemseo.mda.mda import MDA
from gemseo.utils.locks import synchronized
from gemseo.utils.py23_compat import Path
from gemseo.utils.show_utils import generate_xdsm_html
from gemseo.utils.xdsm_to_pdf import xdsm_data_to_pdf

LOGGER = logging.getLogger(__name__)

OPT_NAME = OPT_ID = "Opt"
USER_NAME = USER_ID = "_U_"

EdgeType = Dict[str, Union[MDODiscipline, List[str]]]
NodeType = Dict[str, str]
IdsType = Any

XdsmType = Dict[str, Any]


class XDSMizer(object):
    """Build the XDSM diagram of a scenario as a JSON structure."""

    def __init__(
        self,
        scenario,  # type: Scenario
        hashref="root",  # type: str
        level=0,  # type: int
        expected_workflow=None,  # type: Optional[CompositeExecSequence]
    ):  # type: (...) -> None
        """

        Args:
            scenario: The scenario to be represented as an XDSM diagram.
            hashref: The keyword used in the JSON structure
                to reference the dictionary data structure
                whose keys are "nodes", "edges", "workflow" and "optpb".
            level: The depth of the scenario. Root scenario is level 0.
            expected_workflow: The expected workflow,
                describing the sequence of execution of the different disciplines
                (:class:`.MDODiscipline`, :class:`.Scenario`, :class:`.MDA`, etc.)
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

    def initialize(
        self,
        workflow=None,  # type: Optional[CompositeExecSequence]
    ):  # type: (...) -> None
        """Initialize the XDSM from a workflow.

        The initialization also creates sub-XDSM diagram accordingly.

        Args:
            workflow: The composite execution sequence.
                If None, use the scenario's one.
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
                    name = atom.discipline.name
                    self.to_hashref[atom] = "{}_scn-{}-{}".format(name, level, num)
                    sub_workflow = XDSMizer._find_sub_workflow(self.workflow, atom)
                    self.sub_xdsmizers.append(
                        XDSMizer(
                            atom.discipline, self.to_hashref[atom], level, sub_workflow
                        )
                    )
                    num += 1

    def monitor(
        self,
        outdir=".",  # type: Optional[str]
        outfilename="xdsm.json",  # type: str
        print_statuses=False,  # type: bool
        latex_output=False,  # type: bool
    ):  # type: (...) -> None
        """Monitor the discipline execution by generating XDSM json file on discipline
        status update.

        Args:
            outdir: The name of the directory to store the different files.
                If None, the current working directory is used.
            outfilename: The name of the JSON file.
            print_statuses: If True, print the statuses in the console at each update.
            latex_output: If True, save the XDSM to tikz, tex and pdf files.
        """
        self._monitor = Monitoring(self.scenario)
        self._monitor.add_observer(self)
        # have to reinitialize with monitored workflow
        self.initialize(self._monitor.workflow)
        self.outdir = outdir
        self.outfilename = outfilename
        self.print_statuses = print_statuses
        self.latex_output = latex_output

    def update(
        self,
        atom,  # type: AtomicExecSequence
    ):  # type: (...) -> None  # pylint: disable=unused-argument
        """Generate a new XDSM regarding the atom status update.

        Args:
            atom: The discipline which status is monitored.
        """
        self.run(
            output_directory_path=self.outdir,
            outfilename=self.outfilename,
            latex_output=self.latex_output,
        )
        if self.print_statuses:
            LOGGER.info(str(self._monitor))

    def run(
        self,
        output_directory_path=None,  # type: Optional[str]
        latex_output=False,  # type: bool
        outfilename="xdsm.html",  # type: str
        html_output=True,  # type: bool
        json_output=False,  # type: bool
        open_browser=False,  # type: bool
    ):  # type: (...) -> XdsmType
        """Generate a XDSM diagram from the process.

        By default,
        a self contained HTML file is generated,
        that can be viewed in a browser.

        Args:
            output_directory_path: The name of the directory to store the JSON file.
                If None, the current working directory is used.
                If open_browser is True and outdir is None,
                the file is stored in a temporary directory.
            outfilename: The name of the JSON file.
            latex_output: If True, save the XDSM to tikz, tex and pdf files.
            open_browser: If True, open the web browser and display the XDSM.
            html_output: If True, save the XDSM in a self-contained HTML file
            json_output: If True, save the JSON file.

        Returns:
            The XDSM structure expressed as a dictionary
            whose keys are "nodes", "edges", "workflow" and "optpb".
        """
        xdsm = self.xdsmize()
        xdsm_json = dumps(xdsm, indent=2, ensure_ascii=False)
        base = basename(outfilename)
        outfile_basename = splitext(base)[0]

        no_html_loc = False

        if output_directory_path is None:
            output_directory_path = Path.cwd()
            no_html_loc = True
        else:
            output_directory_path = Path(output_directory_path)

        if json_output:
            json_path = output_directory_path / "{}.json".format(outfile_basename)
            with json_path.open("w") as file_stream:
                file_stream.write(xdsm_json)

        if latex_output:
            xdsm_data_to_pdf(xdsm, output_directory_path, outfile_basename)

        if html_output or open_browser:
            if no_html_loc:
                output_directory_path = Path(mkdtemp(suffix="", prefix="tmp", dir=None))
            out_file_path = (output_directory_path / outfile_basename).with_suffix(
                ".html"
            )
            LOGGER.info("Generating HTML XDSM file in : %s", out_file_path)
            generate_xdsm_html(xdsm, out_file_path)
            if open_browser:
                url = "file://{}".format(out_file_path)
                webbrowser.open(url, new=2)  # open in new tab
            return out_file_path

        return xdsm

    def get_all_sub_xdsmizers(self):  # type: (...) -> List[XDSMizer]
        """Retrieve all the sub-xdsmizers corresponding to the sub-scenarios.

        Returns:
            The sub-xdsmizers.
        """
        result = []
        for sub in self.sub_xdsmizers:
            result.append(sub)
            result.extend(sub.get_all_sub_xdsmizers())
        return result

    @synchronized
    def xdsmize(
        self,
        algoname="Optimizer",  # type: str
    ):  # type: (...) -> Dict[str,Any]
        """Build the data structure to be used to generate the JSON file.

        Args:
            algoname: The name under which a scenario appears in an XDSM.

        Returns:
            The XDSM structure expressed as a dictionary
            whose keys are "nodes", "edges", "workflow" and "optpb".
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
                if sub_xdsmizer.scenario.name.endswith("ing"):
                    name = "{}er".format(sub_xdsmizer.scenario.name[:-3])
                elif sub_xdsmizer.scenario.name.endswith("Scenario"):
                    if isinstance(sub_xdsmizer.scenario, DOEScenario):
                        name = "Trade-Off"
                    elif isinstance(sub_xdsmizer.scenario, MDOScenario):
                        name = "Optimizer"
                    else:
                        name = sub_xdsmizer.scenario.name
                else:
                    name = sub_xdsmizer.scenario.name
                res[sub_xdsmizer.hashref] = sub_xdsmizer.xdsmize(name)
            return res
        return {"nodes": nodes, "edges": edges, "workflow": workflow, "optpb": optpb}

    def _create_nodes(
        self,
        algoname,  # type: str
    ):  # type: (...) ->  List[NodeType]# pylint: disable=too-many-branches
        """Create the nodes of the XDSM from the scenarios and the disciplines.

        Args:
            algoname: The name under which a scenario appears in an XDSM.
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
                node["name"] = self.to_hashref[atom]
            else:
                node["type"] = "analysis"

            if statuses[atom.uuid]:
                node["status"] = statuses[atom.uuid]

            nodes.append(node)

        return nodes

    def _create_edges(self):  # type: (...) -> List[EdgeType]
        """Create the edges of the XDSM from the dataflow of the scenario."""
        edges = []
        # convenient method to factorize code for creating and appending edges

        def add_edge(
            from_edge,  # type: MDODiscipline
            to_edge,  # type: MDODiscipline
            varnames,  # type: List[str]
        ):  # type: (...) -> None
            """Add an edge from a discipline to another with variables names as label.

            Args:
                from_edge: The starting discipline.
                to_edge: The end discipline.
                varnames: The names of the variables
                    going from the starting discipline to the end one.
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

        user_pattern = "L({})" if self.scenario.name == "Sampling" else "{}^(0)"
        opt_pattern = "{}^(1:N)" if self.scenario.name == "Sampling" else "{}^*"
        add_edge(USER_ID, OPT_ID, [user_pattern.format(x) for x in to_opt])
        add_edge(OPT_ID, USER_ID, [opt_pattern.format(x) for x in to_user])

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
    def _get_single_level_atoms(
        workflow,  # type: CompositeExecSequence
    ):  # type: (...) -> List[AtomicExecSequence]
        """Retrieve the list of atoms of the given workflow.

        This method does not look into the loop execution sequences
        coming from the scenario.
        Thus, it retrieves the atoms for a one level XDSM diagram.

        Args:
            The composite execution sequence.

        Returns:
            The atomic execution sequences.
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

    def _find_atom(
        self,
        discipline,  # type: MDODiscipline
    ):  # type: (...) -> AtomicExecSequence
        """Find the atomic sequence corresponding to a given discipline.

        Args:
            discipline: A discipline.

        Returns:
            The atomic sequence corresponding to the given discipline.

        Raises:
            ValueError: If the atomic sequence is not found.
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
                "Discipline {} not found in {}".format(discipline, disciplines)
            )
        return atom

    @staticmethod
    def _find_sub_workflow(
        workflow,  # type: CompositeExecSequence
        atom_controller,  # type: AtomicExecSequence
    ):  # type: (...) -> Optional[LoopExecSequence]
        """Find the sub-workflow from a workflow and controller atom in it.

        Args:
            workflow: The workflow from which to find a sub-workflow.
            atom_controller: The atomic execution sequence that controls
                the loop execution sequence to find.

        Returns:
            The sub-workflow.
            None if the list of execution sequences of the original workflow is empty.
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

    def _create_workflow(self):  # type: (...) -> List[str,IdsType]
        """Manage the creation of the XDSM workflow creation from a formulation one."""
        workflow = [USER_ID, expand(self.workflow, self.to_id)]
        return workflow


def expand(
    wks,  # type: CompositeExecSequence
    to_id,  # type: Mapping[str,str]
):  # type: (...) -> IdsType
    """Expand the workflow structure as an ids structure using to_id mapping.

    The expansion preserve the structure
    while replacing the object by its id in all case
    except when a tuple is encountered as cdr
    then the expansion transforms loop[A, (B,C)] in [idA, {'parallel': [idB, idC]}].

    Args:
        wks: The workflow structure.
        to_id: The mapping dict from object to id.

    Returns:
        The ids structure valid to be used as XDSM json chains.
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
        raise Exception("Bad execution sequence: found {}".format(wks))
    return ids
