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
"""Graph visualization."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

from docstring_inheritance import GoogleDocstringInheritanceMeta
from graphviz import Digraph

from gemseo.util.file_path_manager import FilePathManager

if TYPE_CHECKING:
    from gemseo.util.typing import StrPath


class GraphView(Digraph, metaclass=GoogleDocstringInheritanceMeta):
    """A tool for graph visualization."""

    @dataclass
    class DefaultNodeAttributeValues:
        """The default values of the attributes of a node."""

        color: str = "black"
        """The default foreground color."""

        fillcolor: str = "white"
        """The default background color."""

        fontcolor: str = "black"
        """The default font color."""

        penwidth: str = "1.0"
        """The default line thickness color."""

        style: str = "filled"
        """The default shape."""

    @dataclass
    class DefaultEdgeAttributeValues:
        """The default values of the attributes of an edge."""

        color: str = "black"
        """The default foreground color."""

        fontcolor: str = "black"
        """The default font color."""

        penwidth: str = "1.0"
        """The default line thickness color."""

    __hidden_node_style: Final[str] = "invis"
    """The style of the hidden nodes."""

    __dot_suffix: Final[str] = ".dot"
    """The suffix of the DOT source file written by `visualize()`."""

    __dir_attr_name: Final[str] = "dir"
    """The name of the attribute setting the direction of an edge."""

    __forward: Final[str] = "forward"
    """The direction of a directed edge."""

    __none: Final[str] = "none"
    """The direction of an undirected edge."""

    __default_dir: str
    """The default direction of the edges, either `"forward"` or `"none"`.

    The `dir` attribute passed to `edge()` takes precedence over it.
    """

    def __init__(self, is_directed: bool = True, **kwargs: Any) -> None:
        """
        Args:
            is_directed: Whether to use directed edges by default.
                This direction is written on each edge added with `edge()`,
                so a graph-level edge attribute `dir`,
                set at instantiation or afterwards, is ignored for these edges.
            **kwargs: The arguments of `graphviz.Digraph`.
        """  # noqa: D205, D212, D415
        super().__init__(**kwargs)
        self.__default_dir = self.__forward if is_directed else self.__none

    def _copy_kwargs(self, **kwargs: Any) -> dict[str, Any]:
        """Return the arguments to create a copy or a subgraph of this graph.

        A subclass must accept, by keyword,
        the arguments of `graphviz.Digraph` and `is_directed`,
        or remove from the returned arguments those that its `__init__` does not take.

        Args:
            **kwargs: The arguments that subclasses add to create their copy;
                they override those of `GraphView`.

        Returns:
            The arguments to create a copy or a subgraph of this graph.
        """
        # Pass kwargs on, then merge them again after is_directed,
        # so that they override it and the result
        # does not depend on whether graphviz forwards them.
        copy_kwargs = super()._copy_kwargs(**kwargs)
        copy_kwargs["is_directed"] = self.__default_dir == self.__forward
        copy_kwargs.update(kwargs)
        return copy_kwargs

    def node(  # noqa:D102
        self, name: str, label: str | None = None, attributes=None, **attrs: str
    ) -> None:
        new_attrs = asdict(self.DefaultNodeAttributeValues())
        new_attrs.update(attrs)
        super().node(name, label=label, _attributes=attributes, **new_attrs)

    def edge(  # noqa:D102
        self,
        tail_name: str,
        head_name: str,
        label: str | None = None,
        attributes=None,
        **attrs: str,
    ) -> None:
        new_attrs = asdict(self.DefaultEdgeAttributeValues())
        new_attrs.update(attrs)
        new_attrs.setdefault(self.__dir_attr_name, self.__default_dir)
        super().edge(
            tail_name, head_name, label=label, _attributes=attributes, **new_attrs
        )

    def hide_node(self, name: str) -> None:
        """Hide a node.

        Args:
            name: The name of the node.
        """
        super().node(name, style=self.__hidden_node_style)

    def visualize(
        self,
        show: bool = False,
        file_path: StrPath = "",
        clean_up: bool = True,
    ) -> None:
        """Create the visualization and save it on the disk.

        Args:
            show: Whether to display the graph
                with the default application associated to the file extension.
            file_path: The file path with extension to save the graph,
                whatever the `filename` passed at instantiation.
                A relative path is relative to the `directory` passed at instantiation.
                The suffix sets the format,
                whatever the `format` passed at instantiation.
                If `""`, use the class name with PNG format.
            clean_up: Whether to remove the source files.
        """
        file_path = Path(
            file_path or f"{FilePathManager.to_snake_case(self.__class__.__name__)}.png"
        )
        filename = self.filename
        try:
            self.render(
                file_path.with_suffix(self.__dot_suffix),
                format=file_path.suffix[1:],
                outfile=file_path,
                view=show,
                cleanup=clean_up,
            )
        finally:
            # render() sets the filename for good, which would leak into the copies.
            self.filename = filename

    def _repr_html_(self) -> str:
        return self._repr_image_svg_xml()
