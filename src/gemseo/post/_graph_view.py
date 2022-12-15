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

from docstring_inheritance import GoogleDocstringInheritanceMeta
from graphviz import Digraph

from gemseo.utils.file_path_manager import FilePathManager
from gemseo.utils.python_compatibility import Final


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

    __HIDDEN_NODE_STYLE: Final[str] = "invis"
    """The style of the hidden nodes."""

    __FORWARD: Final[str] = "forward"
    __NONE: Final[str] = "none"
    __DOT_SUFFIX: Final[str] = ".dot"
    __DIR: Final[str] = "dir"

    def __init__(self, is_directed: bool = True) -> None:
        """
        Args:
            is_directed: Whether to use directed edges by default.
        """  # noqa: D205, D212, D415
        super().__init__()
        self.__dir = self.__FORWARD if is_directed else self.__NONE

    def node(  # noqa:D102
        self, name: str, label: str | None = None, _attributes=None, **attrs: str
    ) -> None:
        new_attrs = asdict(self.DefaultNodeAttributeValues())
        new_attrs.update(attrs)
        super().node(name, label=label, _attributes=_attributes, **new_attrs)

    def edge(  # noqa:D102
        self,
        tail_name: str,
        head_name: str,
        label: str | None = None,
        _attributes=None,
        **attrs: str,
    ) -> None:
        new_attrs = asdict(self.DefaultEdgeAttributeValues())
        new_attrs.update(attrs)
        new_attrs.setdefault(self.__DIR, self.__dir)
        super().edge(
            tail_name, head_name, label=label, _attributes=_attributes, **new_attrs
        )

    def hide_node(self, name: str) -> None:
        """Hide a node.

        Args:
            name: The name of the node.
        """
        super().node(name, style=self.__HIDDEN_NODE_STYLE)

    def visualize(
        self,
        show: bool = True,
        file_path: str | Path = "",
        clean_up: bool = True,
    ):
        """Create the visualization and save it on the disk.

        Args:
            show: Whether to display the graph
                with the default application associated to the file extension.
            file_path: The file path with extension to save the graph.
                If ``""``, use the class name with PNG format.
            clean_up: Whether to remove the source files.
        """
        file_path = Path(
            file_path or f"{FilePathManager.to_snake_case(self.__class__.__name__)}.png"
        )
        self.render(
            file_path.with_suffix(self.__DOT_SUFFIX),
            format=file_path.suffix[1:],
            outfile=file_path,
            view=show,
            cleanup=clean_up,
        )
