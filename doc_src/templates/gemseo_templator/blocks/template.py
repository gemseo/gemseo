# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
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
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class WebLink:
    """A weblink defined by a text, an URL and an anchor."""

    text: str
    url: str | None = None
    anchor: str | None = None


@dataclass
class Block:
    """A text block with at least a title and a description.

    A block can also provide features, algorithms and dependencies as weblinks,
    with a default URL to be used when the URL of a weblink is None.

    Finally,
    a block can contain additional URLs,
    to get more information, to discover examples or to read algorithms options.
    """

    title: str
    description: str
    features: list[WebLink] | None = None
    algorithms: list[WebLink] | None = None
    dependencies: list[WebLink] | None = None
    url: str | None = None
    info: str | None = None
    examples: str | None = None
    options: str | None = None

    def __post_init__(self):
        """Update the unset URLs of algorithms, dependencies and features."""
        for weblinks in [self.algorithms, self.dependencies, self.features]:
            if weblinks is not None:
                self.__update_weblinks(weblinks)

    def __update_weblinks(self, weblinks: Iterable[WebLink]) -> None:
        """Set the unset URLs with the URL of the block.

        Args:
            weblinks: The weblinks to be updated if the URL is unset.
        """
        for weblink in weblinks:
            if weblink.url is None:
                weblink.url = f"{self.url}#{weblink.anchor}"
