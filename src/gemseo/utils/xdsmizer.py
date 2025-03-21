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

The latter is used by the
`XDSMjs javascript library <https://github.com/OneraHub/XDSMjs>`_
to produce an interactive web XDSM
and by the pyxdsm python library
to produce TIKZ and PDF versions of the XDSM.

For more information, see:
A. B. Lambe and J. R. R. A. Martins, “Extensions to the Design Structure Matrix for
the Description of Multidisciplinary Design, Analysis, and Optimization Processes”,
Structural and Multidisciplinary Optimization, vol. 46, no. 2, p. 273-284, 2012.
"""

from __future__ import annotations

import logging
from json import dumps
from multiprocessing import RLock
from pathlib import Path
from tempfile import mkdtemp
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from gemseo.algos.design_space import DesignSpace
from gemseo.core._process_flow.execution_sequences.execution_sequence import (
    ExecutionSequence,
)
from gemseo.core._process_flow.execution_sequences.loop import LoopExecSequence
from gemseo.core._process_flow.execution_sequences.parallel import ParallelExecSequence
from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.discipline import Discipline
from gemseo.core.execution_status import ExecutionStatus
from gemseo.core.monitoring import Monitoring
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter
from gemseo.mda.base_mda import BaseMDA
from gemseo.scenarios.base_scenario import BaseScenario
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.scenarios.mdo_scenario import MDOScenario
from gemseo.utils.locks import synchronized
from gemseo.utils.show_utils import generate_xdsm_html
from gemseo.utils.xdsm import XDSM
from gemseo.utils.xdsm_to_pdf import xdsm_data_to_pdf

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.core._process_flow.execution_sequences import BaseCompositeExecSequence

LOGGER = logging.getLogger(__name__)

OPT_NAME = OPT_ID = "Opt"
USER_NAME = USER_ID = "_U_"

EdgeType = dict[str, Union[Discipline, list[str]]]
NodeType = dict[str, str]
IdsType = Any


class XDSMizer:
    """Build the XDSM diagram of a discipline as a JSON structure."""

    _is_scenario: bool
    """Whether the object to XDSMize is a scenario."""

    _scenario_node_title: str
    """The title of the node representing the scenario."""

    def __init__(
        self,
        discipline: Discipline,
        hashref: str = "root",
        level: int = 0,
        expected_workflow: BaseCompositeExecSequence | None = None,
    ) -> None:
        """
        Args:
            discipline: The discipline to be represented as an XDSM diagram.
            hashref: The keyword used in the JSON structure
                to reference the dictionary data structure
                whose keys are "nodes", "edges", "workflow" and "optpb".
            level: The depth of the scenario. Root scenario is level 0.
            expected_workflow: The expected workflow,
                describing the sequence of execution of the different disciplines
                (:class:`.Discipline`, :class:`.Scenario`, :class:`.BaseMDA`, etc.)
        """  # noqa:D205 D212 D415
        if isinstance(discipline, BaseScenario):
            self._is_scenario = True
            if isinstance(discipline, MDOScenario):
                self._scenario_node_title = "Optimizer"
            elif isinstance(discipline, DOEScenario):
                self._scenario_node_title = "DOE"
            else:
                self._scenario_node_title = discipline.name
        else:
            self._is_scenario = False
            design_space = DesignSpace()
            for name in discipline.io.input_grammar:
                design_space.add_variable(name)
            output_names = iter(discipline.io.output_grammar)
            discipline = MDOScenario(
                [discipline],
                next(output_names),
                design_space,
                formulation_name="DisciplinaryOpt",
            )
            for output_name in output_names:
                discipline.add_observable(output_name)

            self._scenario_node_title = "Caller"

        self.scenario = discipline
        self.level = level
        self.hashref = hashref
        self.lock = RLock()
        self._monitor = None
        self.directory_path = "."
        self.json_file_name = "xdsm.json"
        self.to_hashref = {}
        self.to_id = {}  # dictionary to map ExecutionSequence to XDSM id
        self.initialize(expected_workflow)
        self.log_workflow_status = False
        self.save_pdf = False

    def initialize(
        self,
        workflow: BaseCompositeExecSequence | None = None,
    ) -> None:
        """Initialize the XDSM from a workflow.

        The initialization also creates sub-XDSM diagram accordingly.

        Args:
            workflow: The composite execution sequence.
                If ``None``, use the scenario's one.
        """
        self.sub_xdsmizers = []
        # Find disciplines from workflow structure
        if workflow:
            self.workflow = workflow
        else:
            self.workflow = self.scenario.get_process_flow().get_execution_flow()
        self.atoms = self._get_single_level_atoms(self.workflow)

        self.to_hashref = {}
        level = self.level + 1
        num = 1
        for atom in self.atoms:
            if isinstance(atom.process, BaseScenario):
                if atom.process == self.scenario:
                    self.to_hashref[atom] = "root"
                    self.root_atom = atom
                else:  # sub-scenario
                    name = atom.process.name
                    self.to_hashref[atom] = f"{name}_scn-{level}-{num}"
                    sub_workflow = self._find_sub_workflow(self.workflow, atom)
                    self.sub_xdsmizers.append(
                        XDSMizer(
                            atom.process, self.to_hashref[atom], level, sub_workflow
                        )
                    )
                    num += 1

    def monitor(
        self,
        directory_path: str | Path = ".",
        file_name: str = "xdsm",
        log_workflow_status: bool = False,
        save_pdf: bool = False,
    ) -> None:
        """Generate XDSM json file on process status update.

        Args:
            directory_path: The path of the directory to save the files.
            file_name: The file name to be suffixed by a file extension.
            log_workflow_status: Whether to log the evolution of the workflow's status.
            save_pdf: Whether to save the XDSM as a PDF file.
        """
        self._monitor = Monitoring(self.scenario)
        self._monitor.add_observer(self)
        # have to reinitialize with monitored workflow
        self.initialize(self._monitor.workflow)
        self.directory_path = directory_path
        self.json_file_name = f"{file_name}.json"
        self.log_workflow_status = log_workflow_status
        self.save_pdf = save_pdf

    def update(
        self,
        atom: ExecutionSequence,
    ) -> None:  # pylint: disable=unused-argument
        """Generate a new XDSM regarding the atom status update.

        Args:
            atom: The process which status is monitored.
        """
        self.run(
            directory_path=self.directory_path,
            file_name=self.json_file_name,
            save_pdf=self.save_pdf,
        )
        if self.log_workflow_status:
            LOGGER.info(str(self._monitor))

    def run(
        self,
        directory_path: str | Path = ".",
        file_name: str = "xdsm",
        show_html: bool = False,
        save_html: bool = True,
        save_json: bool = False,
        save_pdf: bool = False,
        pdf_build: bool = True,
        pdf_cleanup: bool = True,
        pdf_batchmode: bool = True,
    ) -> XDSM:
        """Generate a XDSM diagram of the :attr:`.scenario`.

        By default,
        a self-contained HTML file is generated,
        that can be viewed in a browser.

        Args:
            directory_path: The path of the directory to save the files.
            file_name: The file name to be suffixed by a file extension.
            show_html: Whether to open the web browser and display the XDSM.
            save_html: Whether to save the XDSM as a HTML file.
            save_json: Whether to save the XDSM as a JSON file.
            save_pdf: Whether to save the XDSM as a PDF file;
                use ``save_pdf=True`` and ``pdf_build=False``
                to generate the ``file_name.tex`` and ``file_name.tikz`` files
                without building the PDF file.
            pdf_build: Whether to generate the PDF file when ``save_pdf`` is ``True``.
            pdf_cleanup: Whether to clean up the intermediate files
                (``file_name.tex``, ``file_name.tikz`` and built files)
                used to build the PDF file.
            pdf_batchmode: Whether pdflatex is run in `batchmode`.

        Returns:
            A XDSM diagram.
        """
        xdsm = self.xdsmize(self._scenario_node_title)
        xdsm_json = dumps(xdsm, indent=2, ensure_ascii=False)

        directory_path = Path(directory_path)
        if not directory_path.exists():
            directory_path.mkdir()

        if save_json:
            with (directory_path / f"{file_name}.json").open("w") as file_stream:
                file_stream.write(xdsm_json)

        if save_pdf:
            xdsm_data_to_pdf(
                xdsm,
                directory_path,
                file_name,
                pdf_build=pdf_build,
                pdf_cleanup=pdf_cleanup,
                pdf_batchmode=pdf_batchmode,
            )

        html_file_path = None
        if save_html or show_html:
            if save_html:
                html_directory_path = directory_path
            else:
                html_directory_path = Path(mkdtemp(suffix="", prefix="tmp"))

            html_file_path = (html_directory_path / file_name).with_suffix(".html")
            generate_xdsm_html(xdsm, html_file_path)

        xdsm = XDSM(xdsm_json, html_file_path)
        if show_html:
            xdsm.visualize()

        return xdsm

    def get_all_sub_xdsmizers(self) -> list[XDSMizer]:
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
        algoname: str = "Optimizer",
    ) -> dict[str, Any]:
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
        optpb = (
            str(self.scenario.formulation.optimization_problem)
            if not self._is_scenario
            else ""
        )

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
                scenario_block_title = sub_xdsmizer._scenario_node_title
                res[sub_xdsmizer.hashref] = sub_xdsmizer.xdsmize(scenario_block_title)
            return res
        return {"nodes": nodes, "edges": edges, "workflow": workflow, "optpb": optpb}

    def _create_nodes(
        self,
        algoname: str,
    ) -> list[NodeType]:  # pylint: disable=too-many-branches
        """Create the nodes of the XDSM from the scenarios and the disciplines.

        Args:
            algoname: The name under which a scenario appears in an XDSM.
        """
        nodes = []
        self.to_id = {}

        statuses = self.workflow.get_statuses()

        # Optimization
        self.to_id[self.root_atom] = OPT_ID
        opt_node = {"id": OPT_ID, "name": algoname, "type": "optimization"}
        if statuses[self.root_atom.uuid]:
            opt_node["status"] = str(statuses[self.root_atom.uuid])

        nodes.append(opt_node)

        # Disciplines
        for atom_id, atom in enumerate(self.atoms):  # pylint: disable=too-many-nested-blocks
            # if a node already created from an atom with same process
            # at one level just reference the same node
            for ref_atom in tuple(self.to_id):
                if atom.process == ref_atom.process:
                    self.to_id[atom] = self.to_id[ref_atom]

                    if (
                        atom.status
                        and atom.parent.status is ExecutionStatus.Status.RUNNING
                    ):
                        node = None
                        for a_node in nodes:
                            if a_node["id"] == self.to_id[atom]:
                                node = a_node
                                break

                        if not node:
                            # TODO: add specific exception?
                            raise (
                                "Node " + self.to_id[atom] + " not found in " + nodes
                            )  # pragma: no cover

                        node["status"] = str(atom.status)

                    break

            if atom in self.to_id:
                continue

            self.to_id[atom] = "Dis" + str(atom_id)
            node = {"id": self.to_id[atom], "name": atom.process.name}

            # node type
            if isinstance(atom.process, BaseMDA):
                node["type"] = "mda"
            elif isinstance(atom.process, BaseScenario):
                node["type"] = "mdo"
                node["subxdsm"] = self.to_hashref[atom]
                node["name"] = self.to_hashref[atom]
            else:
                node["type"] = "analysis"

            if statuses[atom.uuid]:
                node["status"] = str(statuses[atom.uuid])

            nodes.append(node)

        return nodes

    def _create_edges(self) -> list[EdgeType]:
        """Create the edges of the XDSM from the dataflow of the scenario."""
        edges = []

        # convenient method to factorize code for creating and appending edges

        def add_edge(
            from_edge: Discipline,
            to_edge: Discipline,
            varnames: list[str],
        ) -> None:
            """Add an edge from a process to another with variables names as label.

            Args:
                from_edge: The starting process.
                to_edge: The end process.
                varnames: The names of the variables
                    going from the starting process to the end one.
            """
            edge = {"from": from_edge, "to": to_edge, "name": ", ".join(varnames)}
            edges.append(edge)

        # For User to/from optimization
        opt_pb = self.scenario.formulation.optimization_problem

        # fct names such as -y4
        function_name = opt_pb.function_names

        # output variables used by the fonction (eg y4)
        fct_varnames = [f.output_names for f in opt_pb.functions]
        function_varnames = []
        for fvars in fct_varnames:
            function_varnames.extend(fvars)

        to_user = function_name
        to_opt = self.scenario.get_optim_variable_names()

        if self._is_scenario:
            user_pattern = "L({})" if self.scenario.name == "Sampling" else "{}^(0)"
            opt_pattern = "{}^(1:N)" if self.scenario.name == "Sampling" else "{}^*"
            add_edge(USER_ID, OPT_ID, [user_pattern.format(x) for x in to_opt])
            add_edge(OPT_ID, USER_ID, [opt_pattern.format(x) for x in to_user])

        # Disciplines to/from optimization
        for atom in self.atoms:
            if atom is not self.root_atom:
                if isinstance(atom.process, BaseScenario):
                    continue
                varnames = sorted(
                    set(atom.process.io.input_grammar)
                    & set(self.scenario.get_optim_variable_names())
                )

                if varnames:
                    add_edge(OPT_ID, self.to_id[atom], varnames)

                varnames = sorted(
                    set(atom.process.io.output_grammar) & set(function_varnames)
                )
                if varnames:
                    add_edge(self.to_id[atom], OPT_ID, varnames)

        # Disciplines to User/Optimization (from User is already handled at
        # optimizer level)
        disc_to_opt = function_varnames
        for atom in self.atoms:
            if atom is not self.root_atom:
                # special case MDA : skipped
                if isinstance(atom.process, (BaseMDA, BaseScenario)):
                    continue
                out_to_user = [
                    o for o in atom.process.io.output_grammar if o not in disc_to_opt
                ]
                out_to_opt = [
                    o for o in atom.process.io.output_grammar if o in disc_to_opt
                ]
                if out_to_user:
                    add_edge(self.to_id[atom], USER_ID, [x + "^*" for x in out_to_user])
                if out_to_opt:
                    add_edge(self.to_id[atom], OPT_ID, out_to_opt)

        # Disciplines to/from disciplines
        for coupling in self.scenario.get_process_flow().get_data_flow():
            (disc1, disc2, varnames) = coupling
            add_edge(
                self.to_id[self._find_atom(disc1)],
                self.to_id[self._find_atom(disc2)],
                sorted(varnames),
            )

        return edges

    @classmethod
    def _get_single_level_atoms(
        cls,
        workflow: BaseCompositeExecSequence,
    ) -> list[ExecutionSequence]:
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
        for sequence in workflow.sequences:
            if isinstance(sequence, LoopExecSequence):
                atoms.append(sequence.atom_controller)
                if not isinstance(sequence.atom_controller.process, BaseScenario):
                    atoms += cls._get_single_level_atoms(sequence.iteration_sequence)
            elif isinstance(sequence, ExecutionSequence):
                atoms.append(sequence)
            else:
                atoms += cls._get_single_level_atoms(sequence)
        return atoms

    def _find_atom(
        self,
        process: Discipline,
    ) -> ExecutionSequence:
        """Find the atomic sequence corresponding to a given process.

        Args:
            process: A process.

        Returns:
            The atomic sequence corresponding to the given process.

        Raises:
            ValueError: If the atomic sequence is not found.
        """
        atom = None
        if isinstance(process, MDOScenarioAdapter):
            atom = self._find_atom(process.scenario)
        else:
            for atom_i in self.atoms:
                if process == atom_i.process:
                    atom = atom_i
        if atom is None:
            disciplines = [a.process for a in self.atoms]
            msg = f"Discipline {process} not found in {disciplines}"
            raise ValueError(msg)
        return atom

    @classmethod
    def _find_sub_workflow(
        cls,
        workflow: BaseCompositeExecSequence,
        atom_controller: ExecutionSequence,
    ) -> LoopExecSequence | None:
        """Find the sub-workflow from a workflow and controller atom in it.

        Args:
            workflow: The workflow from which to find a sub-workflow.
            atom_controller: The atomic execution sequence that controls
                the loop execution sequence to find.

        Returns:
            The sub-workflow.
            None if the list of execution sequences of the original workflow is empty.
        """
        for sequence in workflow.sequences:
            if not isinstance(sequence, ExecutionSequence):
                sub = cls._find_sub_workflow(sequence, atom_controller)
                if sub is not None:
                    return sub
        return None

    def _create_workflow(self) -> list[str, IdsType]:
        """Manage the creation of the XDSM workflow creation from a formulation one."""
        return [USER_ID, expand(self.workflow, self.to_id)]


def expand(
    wks: BaseCompositeExecSequence,
    to_id: Mapping[str, str],
) -> IdsType:
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
    if isinstance(wks, SequentialExecSequence):
        res = []
        for sequence in wks.sequences:
            res += expand(sequence, to_id)
        ids = res
    elif isinstance(wks, ParallelExecSequence):
        res = []
        for sequence in wks.sequences:
            if isinstance(sequence, ExecutionSequence):
                res += expand(sequence, to_id)
            else:
                res.append(expand(sequence, to_id))
        ids = [{"parallel": res}]
    elif isinstance(wks, LoopExecSequence):
        if (
            isinstance(wks.atom_controller.process, BaseScenario)
            and to_id[wks.atom_controller] != OPT_ID
        ):
            # sub-scnario consider only the controller
            ids = [to_id[wks.atom_controller]]
        else:
            ids = [to_id[wks.atom_controller], expand(wks.iteration_sequence, to_id)]
    elif isinstance(wks, ExecutionSequence):
        ids = [to_id[wks]]
    else:
        msg = f"Bad execution sequence: found {wks}"
        raise TypeError(msg)
    return ids
