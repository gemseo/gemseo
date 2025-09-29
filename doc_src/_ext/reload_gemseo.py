"""Reload all gemseo modules to reset any changes done by sphinx-gallery."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING
from typing import Any

from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.util.logging import getLogger

if TYPE_CHECKING:
    from sphinx.application import Sphinx

logger = getLogger("reload-gemseo")


def reload_gemseo(app: Sphinx, *args, **kwargs) -> None:
    """Reload all gemseo modules"""
    logger.info("reloading gemseo modules", color="white")
    for name in tuple(sys.modules):
        if name.startswith("gemseo."):
            del sys.modules[name]
            with mock(app.config.autodoc_mock_imports):
                import_module(name)


def setup(app: Sphinx) -> dict[str, Any]:
    # Priority must be above the one of sphinx-gallery.
    app.connect("builder-inited", reload_gemseo, priority=1000)
    return {
        "version": "builtin",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
