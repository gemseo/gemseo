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
"""Reset of the global configuration between the gallery examples.

The examples run in the same process,
so an example changing `gemseo.configuration` would change the next ones.
The directory manager is left as is, as it cannot be disabled once enabled;
the examples enabling it run last (see `docs/gallery_conf.py`).
"""

from __future__ import annotations

from logging import root
from typing import TYPE_CHECKING
from typing import Any

from gemseo import configuration
from gemseo.util.global_configuration import GlobalConfiguration
from gemseo.util.global_configuration import _apply
from gemseo.util.logging import LoggingConfiguration

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

_root_handlers = tuple(root.handlers)
"""The handlers of the root logger captured at module import."""


def reset_configuration(gallery_conf: Mapping[str, Any], fname: Path) -> None:
    """Reset the global configuration to its default values.

    Args:
        gallery_conf: The gallery configuration, unused.
        fname: The example about to run, unused.
    """
    # Re-assigning every option re-applies the class attribute that it drives.
    for name in _apply:
        setattr(configuration, name, GlobalConfiguration.model_fields[name].default)

    # configure_root_logger cannot be set back to False
    # and the handlers it added stay on the root logger:
    # replace the logging settings and restore the root handlers.
    configuration.logging = LoggingConfiguration()
    root.handlers = list(_root_handlers)
