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
"""Generator of the HTML file containing a D3.js version of the N2 chart."""
from __future__ import annotations

import json
import webbrowser
from typing import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gemseo.core.coupling_structure import DependencyGraph

from gemseo.utils.n2d3.n2_json import N2JSON
from pathlib import Path


class N2HTML:
    """Generate an HTML file to visualize a dynamic and interactive N2 chart."""

    def __init__(
        self,
        file_path: str | Path = "n2.html",
        open_browser: bool = False,
    ) -> None:
        """
        Args:
            file_path: The file path of the HTML file.
            open_browser: If True, open the browser and display the HTML file.
        """
        self.__file_path = Path(file_path)
        self.__open_browser = open_browser

    def __create_html_file(self, json_structure: str) -> None:
        """Build the HTML file from the JSON structure of the N2 chart.

        Args:
            json_structure: The JSON structure of the N2 chart.
        """
        with Path(self.__file_path).open("w", encoding="utf-8", newline="") as stream:
            stream.write(self.__create_html_contents(json_structure))

        if self.__open_browser:
            webbrowser.open_new_tab(str(self.__file_path))

    def from_graph(
        self,
        graph: DependencyGraph,
        self_coupled_disciplines: Sequence[str] | None = None,
    ) -> None:
        """Create the HTML file from a dependency graph.

        Args:
            graph: The dependency graph.
            self_coupled_disciplines: The names of the self-coupled disciplines, if any.
        """
        self.__create_html_file(str(N2JSON(graph, self_coupled_disciplines)))

    def from_json(
        self,
        file_path: str | Path,
    ) -> None:
        """Create the HTML file from a JSON file.

        Args:
            file_path: The JSON file containing the JSON structure of the N2 chart.
        """
        with Path(file_path).open(encoding="utf-8") as input_file:
            self.__create_html_file(json.dumps(json.load(input_file)))

    def __create_html_contents(
        self,
        json_data: str,
    ) -> str:
        """Create the HTML content related to the N2 chart.

        Args:
            json_data: The JSON structure of the N2 chart.

        Returns:
            The HTML content.
        """

        css_files = ["style.css", "modal.css", "button.css", "materialize.min.css"]
        js_files = [
            "d3.v3.js",
            "d3.parcoords.js",
            "science.v1.js",
            "tiny-queue.js",
            "reorder.v1.js",
            "matrix.js",
            "expand_groups.js",
            "editable_span.js",
            "canvas_toBlob.js",
            "FileSave.js",
            "save_json.js",
            "save_png.js",
            "materialize.min.js",
        ]
        template = (Path(__file__).parent / "n2_html.tmpl").read_text()
        data = [
            self.__get_file_contents(Path("css") / css_file) for css_file in css_files
        ]
        data += [self.__get_file_contents(Path("js") / js_file) for js_file in js_files]
        data += [self.__get_file_contents("gemseo_logo.svg")]
        data += [json_data]
        return template.format(*data)

    @staticmethod
    def __get_file_contents(
        file_name: Path | str,
    ) -> str:
        """Read the content of a file located in the directory `n2d3`.

        Args:
            file_name: The name of the file.

        Returns:
            The content of the file.
        """
        return (Path(__file__).parent / file_name).read_text(encoding="utf-8")
