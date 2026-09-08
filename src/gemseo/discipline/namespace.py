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
#        :author: Gilberto Ruiz Jimenez
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Propagate namespaces along the discipline coupling graph.

Given one or more seed variable names to namespace, this module walks the
discipline coupling graph forward from the disciplines owning those variables
and namespaces every input and output affected by the propagation.

The forward walk relies on `compute_reachable_nodes()`
over the [DependencyGraph][gemseo.core.dependency_graph.DependencyGraph].

Notes:
    The set of disciplines passed to
    [propagate_namespace()][gemseo.discipline.namespace.propagate_namespace]
    must be self-contained. Namespacing an output that is also consumed by a
    discipline **outside** the passed group renames that output only for the
    group, which desynchronizes the external consumer (it keeps expecting the
    bare name).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.discipline.namespace import namespaces_separator
from gemseo.core.discipline.namespace import split_namespace
from gemseo.core.util._graph_traversal import DisciplineIOs
from gemseo.core.util._graph_traversal import compute_reachable_nodes
from gemseo.util.string import pretty_repr
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline.base_discipline import BaseDiscipline


def _compute_affected_ios(
    disciplines: Sequence[BaseDiscipline],
    variable_names: set[str],
) -> tuple[dict[BaseDiscipline, DisciplineIOs], frozenset[str]]:
    """Compute the inputs and outputs affected by namespacing the seed variables.

    The seed disciplines are those owning at least one seed variable as an input
    or an output. Walking the coupling graph forward from these seeds yields the
    reached disciplines. For each reached discipline, all its outputs are affected,
    while its affected inputs are the seeds and the variables produced within the
    reached set. Consequently, design variables that are neither seeds nor
    produced inside the reached set remain unaffected.

    This function does not mutate the disciplines; it only reports what
    [propagate_namespace()][gemseo.discipline.namespace.propagate_namespace]
    would namespace.

    Args:
        disciplines: The disciplines forming the coupling graph.
        variable_names: The set of names of the seed variables to namespace.

    Returns:
        The affected inputs and all outputs of each reached discipline,
        and the names of the variables produced within the reached set.
        Both are empty when `variable_names` is empty.

    Raises:
        ValueError: If a seed name is neither an input nor an output
            of any of the disciplines.
    """
    graph = DependencyGraph(disciplines).graph

    sources = set()
    unmatched_seeds = set(variable_names)
    for discipline in graph.nodes:
        io = discipline.io
        matched_seeds = variable_names.intersection(
            io.input_grammar
        ) | variable_names.intersection(io.output_grammar)
        if matched_seeds:
            sources.add(discipline)
            unmatched_seeds -= matched_seeds

    if unmatched_seeds:
        msg = (
            "The following seed variables are neither inputs nor outputs of the "
            f"disciplines: {pretty_repr(unmatched_seeds)}."
        )
        raise ValueError(msg)

    reached = compute_reachable_nodes(graph, sources)

    produced = set()
    for discipline in reached:
        produced.update(discipline.io.output_grammar)

    seeds_or_produced = variable_names | produced
    # Iterate over graph.nodes rather than over reached: the latter is a set of
    # disciplines hashed by identity, so its iteration order varies between runs,
    # whereas graph.nodes preserves the order of the disciplines. Callers rely on
    # the mapping and the renamed grammar elements being reproducibly ordered.
    discipline_to_ios = {
        discipline: DisciplineIOs(
            inputs=frozenset(
                seeds_or_produced.intersection(discipline.io.input_grammar)
            ),
            outputs=frozenset(discipline.io.output_grammar),
        )
        for discipline in graph.nodes
        if discipline in reached
    }

    return discipline_to_ios, frozenset(produced)


def propagate_namespace(
    disciplines: Sequence[BaseDiscipline],
    namespace: str,
    variable_names: Iterable[str],
) -> dict[BaseDiscipline, DisciplineIOs]:
    """Namespace the inputs and outputs affected by the seed variables.

    The affected inputs and outputs are computed with
    `_compute_affected_ios()` and then renamed in place with
    [add_namespace_to_input()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_input]
    and
    [add_namespace_to_output()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_output].

    Notes:
        The passed disciplines must be self-contained. Namespacing an output that
        is also consumed by a discipline **outside** the passed group renames that
        output only for the group, which desynchronizes the external consumer.
        A variable produced both inside and outside the set of disciplines reached
        from the seeds is rejected outright instead.

        Every output of every reached discipline is namespaced, not only the ones
        carrying the propagation. The renaming therefore also breaks the references
        held by the objects that are not disciplines and so cannot be checked here:

        - the variable names of a
          [DesignSpace][gemseo.space.design.DesignSpace],
        - the objective, constraint and observable names passed to a formulation
          or to a scenario,
        - the coupling names passed to
          [create_mda()][gemseo.create_mda] and any variable name stored in
          settings.

        None of those are reachable from `disciplines`, so no diagnostic can be emitted
        for them: the coupling those references stood for is silently gone. Use the
        returned mapping, which pairs each reached discipline with the original names of
        its affected inputs and outputs, to rewrite such external references by
        prepending `namespace` to them.

    Args:
        disciplines: The disciplines forming the coupling graph.
        namespace: The name of the namespace to prepend to the affected variables.
        variable_names: The names of the seed variables to namespace.

    Returns:
        The affected inputs and all outputs of each reached discipline,
        with their original (un-namespaced) names.
        An empty mapping when `variable_names` is empty.

    Raises:
        ValueError: If a seed name is neither an input nor an output of any of the
            disciplines, if an affected input or output already has a namespace,
            or if an affected output is also produced by a discipline outside
            the set of disciplines reached from the seeds.
    """
    discipline_to_ios, produced = _compute_affected_ios(
        disciplines, set(variable_names)
    )

    already_namespaced = set()
    for discipline, ios in discipline_to_ios.items():
        for name in ios.inputs | ios.outputs:
            if namespaces_separator in name:
                existing_namespace, bare_name = split_namespace(name)
                already_namespaced.add(
                    f"{bare_name!r} of discipline {discipline.name!r} "
                    f"(namespace {existing_namespace!r})"
                )
    if already_namespaced:
        msg = (
            "The following affected variables already have a namespace and "
            f"cannot be namespaced again: {pretty_str(already_namespaced)}."
        )
        raise ValueError(msg)

    disciplines_outside_reached_set = set(disciplines).difference(discipline_to_ios)

    clashing_names = {
        f"{name!r} of discipline {discipline.name!r}"
        for discipline in disciplines_outside_reached_set
        for name in produced.intersection(discipline.io.output_grammar)
    }
    if clashing_names:
        msg = (
            "The following variables are produced both inside and outside "
            "the set of disciplines reached from the seeds, so namespacing "
            f"them would break those couplings: {pretty_str(clashing_names)}."
        )
        raise ValueError(msg)

    for discipline, ios in discipline_to_ios.items():
        for name in sorted(ios.inputs):
            discipline.add_namespace_to_input(name, namespace)
        for name in sorted(ios.outputs):
            discipline.add_namespace_to_output(name, namespace)

    return discipline_to_ios
