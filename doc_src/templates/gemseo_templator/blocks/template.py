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


@dataclass
class WebLink:
    """A weblink defined by a text, an URL and an anchor."""

    text: str
    url: str = ""
    anchor: str = ""


@dataclass
class Block:
    """A text block with at least a title and a description.

    A block can also provide features and dependencies as weblinks, with a default URL
    to be used when the URL of a weblink is None.

    Finally, a block can contain additional URLs, to get more information, to discover
    examples or to read algorithms options.
    """

    title: str
    description: str
    features: tuple[WebLink] = ()
    dependencies: tuple[WebLink] = ()
    url: str = ""
    button_info_url: str = ""
    button_examples_url: str = ""
    button_types_url: str = ""
    button_types_name: str = "Algorithms"

    def __post_init__(self):
        """Update the unset URLs of algorithms, dependencies and features."""
        for weblinks in [self.dependencies, self.features]:
            for weblink in weblinks:
                weblink.url = f"{self.url}#{weblink.anchor}"
