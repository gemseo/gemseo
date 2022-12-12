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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Graph algorithms to solve the Jacobian accumulation problem for MDOChain and MDA."""
from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.chain_rule import traverse_add_diff_io
from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from typing import Iterable
    from gemseo.core.derivatives.chain_rule import DisciplineIOMapping


def _replace_strongly_coupled(
    coupling_structure: MDOCouplingStructure,
) -> tuple[list[MDODiscipline], list[MDODiscipline]]:
    """Replace the strongly coupled disciplines by a single one.

    The replacing discipline has for inputs the merged inputs of all coupled
    discipline except for the strong couplings, while its outputs are all combined
    outputs.

    Args:
        coupling_structure: The input coupling structure containing the graph and all
            the disciplines

    Returns:
        All the disciplines with the strongly coupled replaced by the
        merged one, and the replacing disciplines.
    """
    disciplines_with_group = list(coupling_structure.disciplines)
    reduced_disciplines = []
    all_disc_with_red = []
    strong_c = set(coupling_structure.strong_couplings)
    for parallel_tasks in coupling_structure.sequence:
        for group in parallel_tasks:
            # The strong coupling cycles are treated here
            # And also the self coupled disciplines
            if len(group) > 1 or (
                len(group) == 1 and coupling_structure.is_self_coupled(group[0])
            ):
                disc_merged = MDODiscipline(str(uuid.uuid4()))
                for disc in group:
                    disciplines_with_group.remove(disc)
                    # The strong couplings are not real dependencies of the MDA for
                    # derivatives computation.
                    disc_merged.input_grammar.update(
                        set(disc.input_grammar.names) - strong_c
                    )
                    disc_merged.output_grammar.update(disc.output_grammar.names)

                all_disc_with_red.append(disc_merged)
                reduced_disciplines.append(disc_merged)
            else:
                disc = group[0]
                # The weakly coupled disciplines are treated like in the chain.
                all_disc_with_red.append(disc)

    return all_disc_with_red, reduced_disciplines


def traverse_add_diff_io_mda(
    coupling_structure: MDOCouplingStructure,
    inputs: Iterable[str],
    outputs: Iterable[str],
) -> DisciplineIOMapping:
    """Set the required differentiated IOs for the disciplines in a chain.

    Add the differentiated inputs and outputs to the disciplines in a chain of of
    executions, given the list of inputs with respect to which the derivation is made
    and the list of outputs to be derived. The inputs and outputs are a subset of all
    the inputs and outputs of the chain. This allows to minimize the number of
    linearizations to perform in the disciplines.

    Uses a two ways (from inputs to outputs and then from outputs to inputs)
     breadth first search graph traversal algorithm.
    The graph is constructed by :class:`.DependencyGraph`, which represents the data
    connexions (edges) between the disciplines (nodes).

    Args:
        coupling_structure: The coupling structure of the MDA.
        inputs: The inputs with respect to which the chain chain is differentiated.
        outputs: The chain outputs to be differentiated.

    Returns:
        The merged differentiated inputs and outputs.
    """
    strong_groups = coupling_structure.get_strongly_coupled_disciplines(
        by_group=True, add_self_coupled=True
    )

    all_disc_with_red, reduced_disciplines = _replace_strongly_coupled(
        coupling_structure
    )

    reduced_coupling_structure = MDOCouplingStructure(all_disc_with_red)
    diff_ios_merged = traverse_add_diff_io(
        reduced_coupling_structure.graph.graph,
        inputs,
        outputs,
        add_differentiated_ios=True,
    )

    # The sub MDAs where the strong couplings are handled here.
    strong_couplings = coupling_structure.strong_couplings
    for group, disc_reduced in zip(strong_groups, reduced_disciplines):
        if disc_reduced in diff_ios_merged:
            diff_red_in = set(diff_ios_merged[disc_reduced][0])
            diff_red_out = set(diff_ios_merged[disc_reduced][1])

            for disc in group:
                # There is a need to differentiate with respect to all the inputs of
                # the MDA that are also inputs of the discipline
                # And we add all strong input couplings
                # Finally we keep only the discipline inputs.
                diff_in = diff_red_in.union(strong_couplings).intersection(
                    disc.input_grammar.names
                )
                disc.add_differentiated_inputs(diff_in)

                # It is simpler for the outputs because the outputs to be differentiated
                # are the ones from the MDA.

                diff_out = diff_red_out.union(strong_couplings).intersection(
                    disc.output_grammar.names
                )
                disc.add_differentiated_outputs(diff_out)

                diff_ios_merged[disc] = (list(diff_in), list(diff_out))

    # The state variables and residuals must be differentiated too, only for the
    # disciplines involved in the computations.
    for disc, diffio_disc in diff_ios_merged.items():
        if disc.residual_variables:
            residuals = disc.residual_variables.keys()
            states = disc.residual_variables.values()
            diffio_disc[0].extend(states)
            diffio_disc[1].extend(residuals)
            disc.add_differentiated_inputs(states)
            disc.add_differentiated_outputs(residuals)

    return diff_ios_merged
