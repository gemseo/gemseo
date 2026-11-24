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
"""Graph visualization.

[draw_coupling_graph()][gemseo.post.graph.draw_coupling_graph]
draws the graph of the couplings between disciplines
with [Graphviz](https://graphviz.org/)
as a [graphviz.Digraph](https://graphviz.readthedocs.io/en/stable/api.html#graphviz.Digraph).
[save_graph()][gemseo.post.graph.save_graph]
saves it, or any other graph, to a file,
whose extension sets the format;
the `render()` method of the graph can also save it or show it.

This module requires the optional dependency
[graphviz](https://pypi.org/project/graphviz/),
and saving the graph requires
the `dot` executable of [Graphviz](https://graphviz.org/).
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from contextlib import nullcontext
from contextlib import suppress
from functools import cache
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from typing import Any
from typing import Final
from typing import NoReturn
from typing import TypeAlias
from warnings import catch_warnings
from warnings import simplefilter

from graphviz import FORMATS
from graphviz import Digraph as GraphvizDigraph
from graphviz import UnknownSuffixWarning
from graphviz import nohtml

from gemseo.core.discipline.base_discipline import BaseDiscipline
from gemseo.util.discipline import get_discipline_variable_properties
from gemseo.util.string import pretty_str

if TYPE_CHECKING:
    from collections.abc import Sequence

    from graphviz import Graph as GraphvizGraph
    from networkx import DiGraph as NetworkxDiGraph

    from gemseo.core.dependency_graph import DependencyGraph
    from gemseo.util.discipline import DisciplineVariableProperties
    from gemseo.util.typing import StrPath

    _PropertyGetter: TypeAlias = Callable[
        [BaseDiscipline],
        tuple[
            Mapping[str, DisciplineVariableProperties],
            Mapping[str, DisciplineVariableProperties],
        ],
    ]
    """A function returning the input and output properties of a discipline."""


EdgeGetter: TypeAlias = Callable[
    [BaseDiscipline | None, BaseDiscipline | None, str], Iterable[Mapping[str, str]]
]
"""A function returning the Graphviz attributes of the edges of a coupling variable.

It is called with the tail discipline of the edges,
or `None` for the node of an input of the problem,
their head discipline,
or `None` for a hidden node or the node of an output of the problem,
and the name of the variable.
It returns one mapping of Graphviz attributes per edge to draw.
"""

_device_formats: Final[frozenset[str]] = frozenset({"gtk", "x11", "xlib"})
"""The Graphviz formats that open a window instead of writing a file."""

NodeAttributeGetter: TypeAlias = Callable[[str, bool], Mapping[str, str]]
"""A function returning the Graphviz attributes of the node of a variable.

It is called with the name of the variable
and whether it is an input (`True`) or an output (`False`) of the problem.
"""


def draw_coupling_graph(
    dependency_graph: DependencyGraph,
    full: bool = True,
    show_edge_labels: bool = True,
    get_edges: EdgeGetter | None = None,
    input_names: Iterable[str] = (),
    output_names: Iterable[str] = (),
    get_node_attributes: NodeAttributeGetter | None = None,
) -> GraphvizDigraph:
    """Draw the coupling graph of a dependency graph.

    The graph is neither saved nor shown;
    use [save_graph()][gemseo.post.graph.save_graph]
    or the `render()` method of the returned graph,
    e.g. `graph.render(outfile="graph.pdf")`,
    to save it.

    This function requires the optional dependency
    [graphviz](https://pypi.org/project/graphviz/),
    without which its module cannot be imported,
    and saving the graph requires
    the `dot` executable of [Graphviz](https://graphviz.org/).

    The graph shows how the disciplines are coupled.
    Use `input_names` and `output_names`
    to also show where the inputs of the problem enter the disciplines
    and which disciplines compute its outputs of interest.
    Each of these variables is drawn as a node with the variable name as label.
    The input nodes are placed on the first rank
    and the output nodes on the last rank,
    i.e. above and below the disciplines with the default top-down layout.

    A coupling variable can be passed as input,
    e.g. the initial guess of a coupling variable in an IDF formulation:
    its input node is drawn
    and the coupling edges between the disciplines are kept.

    The names of the disciplines and variables are only used as node labels;
    the node names, to use to post-process the returned graph, are:

    - `"d<i>"` for the discipline at position `i` (from 0)
      in `dependency_graph.graph.nodes`,
      in both the full and the condensed graphs,
    - `"scc<i>"` for the node `i` of the condensed graph,
      as numbered by
      [create_condensed_graph()][gemseo.core.dependency_graph.DependencyGraph.create_condensed_graph],
      when it groups several disciplines,
      labelled `"MDA of <discipline names>"`,
    - `"in<j>"` for the input name at position `j` (from 0)
      and `"out<k>"` for the output name at position `k` (from 0),
      the names being sorted in alphabetical order without duplicates,
    - `"h<i>"` for the invisible head node of the edges
      of the outputs of the leaf at position `i` (from 0)
      in `dependency_graph.graph.nodes` for the full graph
      or of the leaf `i` of the condensed graph.

    Args:
        dependency_graph: The dependency graph whose coupling graph to draw.
        full: Whether to draw the full graph;
            otherwise, draw the condensed graph.
        show_edge_labels: Whether to display the coupling variable names on edges.
            The edges of the input and output nodes have no label by default,
            as these nodes already name the variables.
        get_edges: A function to draw and style one edge per coupling variable,
            or even several edges per coupling variable,
            e.g. one per component of a vector variable.
            It is called for each coupling variable,
            in alphabetical order of the variable names,
            with the tail discipline of the edge,
            its head discipline
            and the name of the variable.
            The head discipline is the same as the tail discipline
            for a variable that is both an input and an output.
            When `output_names` is empty,
            the function is also called
            for each output of a discipline without successors,
            self-coupled outputs included,
            with `None` as head discipline.
            The function is also called for the edges of the input and output nodes:
            the tail discipline is `None` for an edge from an input node
            and the head discipline is `None` for an edge to an output node.
            It returns the attributes of the edges to draw,
            one mapping of Graphviz attributes per edge,
            e.g. `[{"label": "y[0]", "penwidth": "2"}, {"label": "y[1]"}]`;
            an empty list draws no edge for this variable.
            These attributes override the default ones,
            namely the label (the name of the variable if `show_edge_labels`
            and the edge is not that of an input or output node)
            and the tooltip (if available),
            so a `label` returned by this function is displayed
            even when `show_edge_labels` is `False`.
            If `None`, draw one edge per pair of disciplines,
            labelled with the names of all their coupling variables.
            This function is only supported by the full graph.
        input_names: The names of the inputs of the problem to draw.
            Each one is drawn as a node
            with an edge to each discipline that has this name as input.
            Each name must be an input of at least one discipline.
            These names are only supported by the full graph.
        output_names: The names of the outputs of the problem to draw.
            Each one is drawn as a node
            with an edge from each discipline that has this name as output,
            whether or not this discipline is a leaf of the graph.
            Each name must be an output of at least one discipline.
            When not empty,
            the outputs of the leaf disciplines are no longer drawn
            as edges to hidden nodes.
            These names are only supported by the full graph.
        get_node_attributes: A function to style the input and output nodes.
            It is called with the name of the variable
            and whether it is an input (`True`) or an output (`False`),
            and returns a mapping of Graphviz attributes,
            e.g. `{"color": "blue", "label": "x (m)"}`.
            These attributes override the default ones,
            namely the label (the name of the variable)
            and the shape (`"plaintext"`).
            If `None`, use the default attributes.
            This function is only supported by the full graph.

    Returns:
        The coupling graph.

    Raises:
        ValueError: When `get_edges`, `input_names`, `output_names`
            or `get_node_attributes` is passed with `full=False`;
            when a name of `input_names` is not an input of any discipline;
            when a name of `output_names` is not an output of any discipline.
        TypeError: When `get_edges` returns something else
            than a sequence of mappings.
    """
    input_names = sorted(set(input_names))
    output_names = sorted(set(output_names))
    _check_arguments(full, get_edges, input_names, output_names, get_node_attributes)
    disciplines = dependency_graph.graph.nodes
    input_to_disciplines = _get_name_to_disciplines(input_names, disciplines, True)
    output_to_disciplines = _get_name_to_disciplines(output_names, disciplines, False)
    graph = (
        dependency_graph.graph if full else dependency_graph.create_condensed_graph()
    )
    digraph = GraphvizDigraph(node_attr={"style": "filled", "fillcolor": "white"})
    # The tooltips need a full graph;
    # the properties are computed once per discipline and only when needed.
    get_properties = cache(get_discipline_variable_properties) if full else None
    add_edges = partial(_add_edges, digraph, get_edges, get_properties)

    # Declare all the nodes, including those without any edge.
    node_names = _add_discipline_nodes(digraph, graph, disciplines)
    _add_variable_nodes(digraph, input_names, True, get_node_attributes)
    _add_variable_nodes(digraph, output_names, False, get_node_attributes)

    # 1. Add the edges with different head and tail nodes
    #    (case: some outputs of a discipline are inputs of another one)
    for tail_node, head_node, coupling_names in graph.edges(data=dependency_graph.io):
        add_edges(
            coupling_names,
            tail_node,
            head_node,
            node_names[tail_node],
            node_names[head_node],
            show_edge_labels,
        )

    # 2. Add the edges with same head and tail nodes
    #    (case: some outputs of a discipline are inputs of itself)
    if full:
        for discipline in disciplines:
            coupling_names = set(discipline.io.input_grammar).intersection(
                discipline.io.output_grammar
            )
            if coupling_names:
                name = node_names[discipline]
                add_edges(
                    coupling_names, discipline, discipline, name, name, show_edge_labels
                )

    # 3. Add the edges from the nodes of the inputs of the problem.
    for j, (name, input_disciplines) in enumerate(input_to_disciplines.items()):
        for discipline in input_disciplines:
            add_edges([name], None, discipline, f"in{j}", node_names[discipline], False)

    # 4. Add the edges to the nodes of the outputs of the problem if any,
    #    otherwise the edges from the leaves to hidden nodes
    #    (case: some output variables of discipline are not coupling variables).
    if output_names:
        for k, (name, output_disciplines) in enumerate(output_to_disciplines.items()):
            for discipline in output_disciplines:
                add_edges(
                    [name], discipline, None, node_names[discipline], f"out{k}", False
                )
    else:
        for i, node in enumerate(graph.nodes):
            if graph.out_degree(node):
                continue

            leaf_output_names = _get_variable_names(graph, node, False)
            if not leaf_output_names:
                continue

            head_name = f"h{i}"
            if add_edges(
                leaf_output_names,
                node,
                None,
                node_names[node],
                head_name,
                show_edge_labels,
            ):
                digraph.node(head_name, style="invis")

    return digraph


def save_graph(
    graph: GraphvizGraph | GraphvizDigraph,
    file_path: StrPath,
    clean_up: bool = True,
) -> None:
    """Save a Graphviz graph to a file.

    The extension of the file sets the format, e.g. `".pdf"` or `".svg"`;
    it must be one of
    [graphviz.FORMATS](https://graphviz.readthedocs.io/en/stable/api.html#graphviz.FORMATS)
    except `".gtk"`, `".x11"` and `".xlib"`,
    which open a window instead of writing a file.
    The graph is laid out from its DOT source file,
    which is either removed or kept next to the file with the extension `.gv`,
    or `.dot` when the extension of the file is `.gv`.

    The graph is unchanged, in particular its `filename` attribute.

    This function requires
    the `dot` executable of [Graphviz](https://graphviz.org/).

    Args:
        graph: The Graphviz graph, directed or not,
            e.g. drawn by
            [draw_coupling_graph()][gemseo.post.graph.draw_coupling_graph].
        file_path: The path of the file,
            relative to the current working directory if not absolute,
            whatever the `directory` attribute of the graph.
            Its missing parent directories are created.
            If the graph cannot be saved,
            the files created by this function are removed
            and so are these directories, unless they are no longer empty,
            e.g. because another process has written a file to them;
            the files that existed before are not removed,
            but may have been overwritten.
        clean_up: Whether to remove the DOT source file;
            otherwise, keep it at `file_path` with the extension `.gv`,
            e.g. `graph.gv` for `file_path="graph.pdf"`,
            or `.dot` when the extension of `file_path` is `.gv`,
            e.g. `graph.dot` for `file_path="graph.gv"`,
            overwriting an existing file at this path,
            even if the graph cannot be saved.

    Raises:
        ValueError: When `file_path` has no extension,
            an extension that is not a Graphviz format
            or that of a Graphviz format opening a window.
    """
    file_format = _check_graph_file_path(file_path)
    # Use an absolute path,
    # as render() resolves a relative one against the directory of the graph.
    path = Path(file_path).absolute()
    # Remember the missing parent directories of the file, from the deepest,
    # to remove those created if saving fails.
    missing_directories = [parent for parent in path.parents if not parent.exists()]
    # Render a copy, as render() sets the filename of the graph for good.
    # With clean_up,
    # write the DOT source file in a temporary directory
    # so that it neither overwrites a file next to the output file
    # nor is the output file itself when its extension is .gv.
    # Without clean_up,
    # replace the extension of the output file, which can contain dots,
    # e.g. ".xdot1.4",
    # by one that is not that of the output file.
    with TemporaryDirectory() if clean_up else nullcontext() as directory:
        if clean_up:
            source = Path(directory) / "graph.gv"
        else:
            source_extension = ".dot" if file_format == "gv" else ".gv"
            stem = path.name[: -len(file_format) - 1]
            source = path.with_name(f"{stem}{source_extension}")

        # Remember the files that did not exist before rendering
        # to remove them if render() fails, keeping those that existed.
        new_files = [file for file in (source, path) if not file.exists()]
        try:
            # Create the parent directory of the file,
            # as dot does not and render() only creates that of the DOT source file.
            path.parent.mkdir(parents=True, exist_ok=True)
            # Pass the format, as render() infers it from the last suffix only,
            # and ignore its warning about a format containing dots, e.g. xdot1.4.
            with catch_warnings():
                simplefilter("ignore", UnknownSuffixWarning)
                graph.copy().render(source, outfile=path, format=file_format)
        except BaseException:
            for file in new_files:
                file.unlink(missing_ok=True)

            # Only remove the empty directories,
            # as another process may have written files to them meanwhile;
            # the directories that could not be created do not exist.
            for missing_directory in missing_directories:
                with suppress(OSError):
                    missing_directory.rmdir()

            raise


def _check_graph_file_path(
    file_path: StrPath, example_file_name: str = "graph.pdf"
) -> str:
    """Check the path of a file to save a graph to.

    Use it to check the path before drawing a graph that takes long to draw;
    [save_graph()][gemseo.post.graph.save_graph]
    also checks it.

    Args:
        file_path: The path of the file,
            whose extension must be one of
            [graphviz.FORMATS](https://graphviz.readthedocs.io/en/stable/api.html#graphviz.FORMATS)
            except `".gtk"`, `".x11"` and `".xlib"`,
            which open a window instead of writing a file.
            This extension can contain dots, e.g. `".xdot1.4"`.
        example_file_name: The name of the file
            that the error message suggests adding to `file_path`
            when it is `"."` or `".."`.

    Returns:
        The Graphviz format set by the extension of the file path, in lower case.

    Raises:
        ValueError: When `file_path` has no extension,
            an extension that is not a Graphviz format
            or that of a Graphviz format opening a window.
    """
    path = Path(file_path)
    if not path.suffix:
        # Path(".").name and Path("..").name are not file names to extend.
        example = path / example_file_name if path.name in {"", ".."} else f"{path}.pdf"
        msg = (
            f"The file path '{path}' has no extension; "
            f"add one to set the format, e.g. '{example}'."
        )
        raise ValueError(msg)

    # The longest format ending the file name, as a format can contain dots.
    file_name = path.name.lower()
    file_format = max(
        (
            file_format
            for file_format in FORMATS
            if file_name.endswith(f".{file_format}")
        ),
        key=len,
        default="",
    )
    if not file_format:
        msg = (
            f"The extension of the file path '{path}' is not a Graphviz format; "
            "use one of graphviz.FORMATS, e.g. '.pdf'."
        )
        raise ValueError(msg)

    if file_format in _device_formats:
        msg = (
            f"The extension of the file path '{path}' is that of a Graphviz format "
            "opening a window instead of writing a file; "
            "use another one of graphviz.FORMATS, e.g. '.pdf'."
        )
        raise ValueError(msg)

    return file_format


def _check_arguments(
    full: bool,
    get_edges: EdgeGetter | None,
    input_names: Sequence[str],
    output_names: Sequence[str],
    get_node_attributes: NodeAttributeGetter | None,
) -> None:
    """Check the arguments of `draw_coupling_graph()`.

    Args:
        full: Whether to draw the full graph.
        get_edges: The function drawing the edges, if any.
        input_names: The names of the inputs of the problem to draw.
        output_names: The names of the outputs of the problem to draw.
        get_node_attributes: The function styling the variable nodes, if any.

    Raises:
        ValueError: When `get_edges`, `input_names`, `output_names`
            or `get_node_attributes` is passed with `full=False`.
    """
    if not full:
        for argument_name, argument in (
            ("get_edges", get_edges),
            ("input_names", input_names),
            ("output_names", output_names),
            ("get_node_attributes", get_node_attributes),
        ):
            if argument:
                msg = (
                    f"The argument {argument_name!r} is only supported "
                    "by the full graph "
                    "because the condensed graph merges disciplines; "
                    "use full=True."
                )
                raise ValueError(msg)


def _get_name_to_disciplines(
    names: Sequence[str], disciplines: Iterable[BaseDiscipline], is_input: bool
) -> dict[str, list[BaseDiscipline]]:
    """Return the disciplines having each name as input or output.

    Args:
        names: The names of the variables.
        disciplines: The disciplines.
        is_input: Whether the names are inputs;
            otherwise, they are outputs.

    Returns:
        The disciplines having each name as input or output,
        in the order of `disciplines`,
        for each name in the order of `names`.

    Raises:
        ValueError: When a name is not an input or output of any discipline.
    """
    if not names:
        return {}

    name_to_disciplines = {name: [] for name in names}
    for discipline in disciplines:
        io = discipline.io
        for name in io.input_grammar if is_input else io.output_grammar:
            name_disciplines = name_to_disciplines.get(name)
            if name_disciplines is not None:
                name_disciplines.append(discipline)

    unknown_names = [name for name, value in name_to_disciplines.items() if not value]
    if unknown_names:
        kind = "input" if is_input else "output"
        msg = (
            f"The following names are not {kind}s of any discipline: "
            f"{pretty_str(unknown_names)}."
        )
        raise ValueError(msg)

    return name_to_disciplines


def _get_default_label(text: str) -> str:
    """Return a default label of a node or an edge.

    Args:
        text: The text of the label.

    Returns:
        The label displaying the text as is.
    """
    # Use the text as is and not as an HTML-like label, e.g. "<x>".
    return nohtml(text)


def _add_variable_nodes(
    digraph: GraphvizDigraph,
    names: Sequence[str],
    is_input: bool,
    get_node_attributes: NodeAttributeGetter | None,
) -> None:
    """Add the nodes of the inputs or outputs of the problem to a graph.

    The input nodes are in a subgraph of rank `"source"`
    and the output nodes in a subgraph of rank `"sink"`.

    Args:
        digraph: The [Digraph][graphviz.Digraph].
        names: The names of the variables,
            whose nodes are named `"in<j>"` for the input at position `j`
            and `"out<k>"` for the output at position `k`.
        is_input: Whether the variables are inputs;
            otherwise, they are outputs.
        get_node_attributes: The function returning the attributes of a node
            from the name of its variable and `is_input`;
            if `None`, use the default attributes.
    """
    if not names:
        return

    rank = "source" if is_input else "sink"
    prefix = "in" if is_input else "out"
    with digraph.subgraph(name=f"{rank}_nodes") as subgraph:
        subgraph.attr(rank=rank)
        for i, name in enumerate(names):
            attributes = {"shape": "plaintext"}
            if get_node_attributes is not None:
                attributes.update(get_node_attributes(name, is_input))

            label = attributes.pop("label", _get_default_label(name))
            subgraph.node(f"{prefix}{i}", label, **attributes)


def _get_members(
    graph: NetworkxDiGraph, node: int | BaseDiscipline
) -> tuple[BaseDiscipline, ...]:
    """Return the disciplines of a node.

    Args:
        graph: A full or condensed graph.
        node: A node of the graph.

    Returns:
        The discipline of a node of a full graph
        or the disciplines of a node of a condensed graph,
        in the order of the dependency graph.
    """
    if isinstance(node, BaseDiscipline):
        return (node,)

    # networkx stores the disciplines under the members node attribute,
    # which DependencyGraph.create_condensed_graph() orders like the disciplines.
    return tuple(graph.nodes[node]["members"])


def _get_variable_names(
    graph: NetworkxDiGraph, node: int | BaseDiscipline, is_input: bool
) -> frozenset[str]:
    """Return the names of the inputs or outputs of the disciplines of a node.

    Args:
        graph: A full or condensed graph.
        node: A node of the graph.
        is_input: Whether to return the names of the inputs;
            otherwise, return the names of the outputs.

    Returns:
        The names of the inputs or outputs of the disciplines of the node.
    """
    return frozenset(
        name
        for discipline in _get_members(graph, node)
        for name in (
            discipline.io.input_grammar if is_input else discipline.io.output_grammar
        )
    )


def _add_discipline_nodes(
    digraph: GraphvizDigraph,
    graph: NetworkxDiGraph,
    disciplines: Iterable[BaseDiscipline],
) -> dict[int | BaseDiscipline, str]:
    """Add the nodes of the disciplines to a graph.

    A node is labelled with the name of its discipline,
    or `"MDA of <names>"` for a node of a condensed graph
    grouping several disciplines.

    Args:
        digraph: The [Digraph][graphviz.Digraph].
        graph: A full or condensed graph.
        disciplines: The disciplines of the dependency graph.

    Returns:
        The names of the nodes of the graph, in the order of its nodes,
        namely `"d<i>"` for the discipline at position `i` in `disciplines`
        and `"scc<i>"` for the node `i` of a condensed graph
        grouping several disciplines.
    """
    discipline_to_index = {discipline: i for i, discipline in enumerate(disciplines)}
    node_names = {}
    for node in graph.nodes:
        members = _get_members(graph, node)
        if len(members) == 1:
            discipline = members[0]
            node_name = f"d{discipline_to_index[discipline]}"
            label = discipline.name
        else:
            node_name = f"scc{node}"
            label = f"MDA of {', '.join(discipline.name for discipline in members)}"

        node_names[node] = node_name
        digraph.node(node_name, _get_default_label(label))

    return node_names


def _add_edges(
    digraph: GraphvizDigraph,
    get_edges: EdgeGetter | None,
    get_properties: _PropertyGetter | None,
    coupling_names: Iterable[str],
    tail_node: int | BaseDiscipline | None,
    head_node: int | BaseDiscipline | None,
    tail_name: str,
    head_name: str,
    labelled: bool,
) -> bool:
    """Add edges to a graph.

    Args:
        digraph: The [Digraph][graphviz.Digraph].
        get_edges: The function returning the attributes of the edges to draw
            from the tail discipline, the head discipline and the coupling name;
            if `None`, draw one edge for all the coupling variables.
        get_properties: The function returning the properties of the variables
            of a discipline;
            if `None`, the edges have no tooltip.
        coupling_names: The names of the coupling variables.
        tail_node: The tail node,
            or `None` for the node of an input of the problem.
        head_node: The head node,
            or `None` for a hidden node or the node of an output of the problem.
        tail_name: The name of the tail node.
        head_name: The name of the head node.
        labelled: Whether to display the coupling variable names on the edges
            by default.

    Returns:
        Whether at least one edge has been drawn.

    Raises:
        TypeError: When `get_edges` returns something else
            than a sequence of mappings.
    """
    # The names of the coupling variables of a group of edges
    # with the Graphviz attributes overriding the default ones of each edge.
    if get_edges is None:
        names_to_edges = [(sorted(coupling_names), [{}])]
    else:
        names_to_edges = (
            ([name], get_edges(tail_node, head_node, name))
            for name in sorted(coupling_names)
        )

    has_edge = False
    for names, edges in names_to_edges:
        if isinstance(edges, (Mapping, str)) or not isinstance(edges, Iterable):
            _raise_wrong_edges_type(edges)

        edges = list(edges)
        for edge in edges:
            if not isinstance(edge, Mapping):
                _raise_wrong_edges_type(edges, edge)

        if not edges:
            # Do not build the tooltip of a variable without edges.
            continue

        tooltip = (
            None
            if get_properties is None
            else _get_tooltip(names, get_properties, tail_node, head_node)
        )
        for edge_attributes in edges:
            attributes = dict(edge_attributes)
            label = attributes.pop(
                "label",
                _get_default_label(pretty_str(names, use_and=False))
                if labelled
                else None,
            )
            kwargs = {}
            if tooltip is not None:
                # An empty label cannot be hovered.
                kwargs["labeltooltip" if label else "edgetooltip"] = tooltip

            kwargs.update(attributes)
            digraph.edge(tail_name, head_name, label, **kwargs)
            has_edge = True

    return has_edge


def _raise_wrong_edges_type(edges: Any, *edge: Any) -> NoReturn:
    """Raise an error for a wrong value returned by an edge getter.

    Args:
        edges: The value returned by the edge getter.
        *edge: The item of `edges` that is not a mapping, if any.

    Raises:
        TypeError: Always.
    """
    if not edge:
        wrong_value = f"an object of type {type(edges).__name__!r}"
    else:
        wrong_value = (
            f"an object of type {type(edges).__name__!r} "
            f"containing an object of type {type(edge[0]).__name__!r}"
        )

    msg = (
        "The function 'get_edges' must return a sequence of mappings, "
        f"one per edge, not {wrong_value}; "
        "return [{...}] to draw one edge or [] to draw no edge."
    )
    raise TypeError(msg)


def _get_tooltip(
    names: Iterable[str],
    get_properties: _PropertyGetter,
    tail: BaseDiscipline | None,
    head: BaseDiscipline | None,
) -> str:
    """Return the tooltip of an edge.

    Args:
        names: The names of the coupling variables, in alphabetical order.
        get_properties: The function returning the properties of the variables
            of a discipline.
        tail: The tail discipline,
            or `None` for the node of an input of the problem.
        head: The head discipline,
            or `None` for a hidden node or the node of an output of the problem.

    Returns:
        The original and current names of the coupling variables,
        one line per variable in alphabetical order.
        For a self-coupling,
        the headers of the columns of the discipline read "Output name"
        and "Input name" instead of "Name".
        For two different disciplines with the same name,
        they read "Name in source discipline" and "Name in target discipline"
        instead of "Name in discipline".
    """
    tail_kind = head_kind = "Name"
    tail_qualifier = head_qualifier = ""
    if tail is not None and head is not None:
        if tail is head:
            tail_kind = "Output name"
            head_kind = "Input name"
        elif tail.name == head.name:
            tail_qualifier = "source "
            head_qualifier = "target "

    # The disciplines of the edge with the index of the relevant properties
    # and the header of their column:
    # the outputs of the tail discipline and the inputs of the head discipline.
    disciplines = [
        (discipline, index, f"{kind} in {qualifier}discipline {discipline.name!r}")
        for discipline, index, kind, qualifier in (
            (tail, 1, tail_kind, tail_qualifier),
            (head, 0, head_kind, head_qualifier),
        )
        if discipline is not None
    ]

    sep = ", "
    header = sep.join(["Global name", *(h for _, _, h in disciplines)])
    lines = [f"{header}\n"]
    for name in names:
        original_names = (
            get_properties(d)[index][name].original_name for d, index, _ in disciplines
        )
        lines.append(sep.join([name, *original_names]))

    return "\n".join(lines)
