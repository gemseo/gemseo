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
"""A XDSM diagram (eXtended Design Structure Matrix)."""

from __future__ import annotations

import webbrowser
from typing import TYPE_CHECKING
from typing import Any
from typing import Final

if TYPE_CHECKING:
    from pathlib import Path


class XDSM:
    """A XDSM diagram (eXtended Design Structure Matrix)."""

    __XDSM_TEMPLATE: Final[str] = """
    <script type="text/javascript">
        document.addEventListener('DOMContentLoaded', () => {{
          const mdo = {};
          const config = {{
            labelizer: {{
                ellipsis: 5,
                subSupScript: false,
                showLinkNbOnly: false,
            }},
            layout: {{
                origin: {{ x: 50, y: 20 }},
                cellsize: {{ w: 150, h: 50 }},
                padding: 10,
            }},
            withDefaultDriver: true,
            withTitleTooltip: true,
          }};
          xdsmjs.XDSMjs(config).createXdsm(mdo);
        }});
    </script>
    """

    def __init__(self, json_schema: dict[str, Any], html_file_path: Path | None):
        """
        Args:
            json_schema: The JSON schema of the XDSM.
            html_file_path: The path to the HTML representation of the XDSM if any.
        """  # noqa: D205 D212 D415
        self.__json_schema = json_schema
        self.__html_file_path = html_file_path
        if html_file_path is not None:
            self.__html_file_url = f"file://{html_file_path}"
        else:
            self.__html_file_url = ""

    @property
    def html_file_path(self) -> Path | None:
        """The path to the HTML file if any."""
        return self.__html_file_path

    @property
    def json_schema(self) -> dict[str, Any]:
        """The JSON schema for `XDSMjs <https://github.com/OneraHub/XDSMjs>`__."""
        return self.__json_schema

    def visualize(self) -> None:
        """Open a web browser and display the XDSM."""
        if not self.__html_file_url:
            raise ValueError(
                "A HTML file is required to visualize the XDSM in a web browser."
            )

        webbrowser.open(self.__html_file_url, new=2)

    def _repr_html_(self) -> str:
        return (
            f"{self.__XDSM_TEMPLATE.format(self.__json_schema)}"
            "<div class='xdsm-toolbar'></div>"
            "<div class='xdsm2'></div>"
        )
