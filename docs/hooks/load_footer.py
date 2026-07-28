"""Load the footer link columns from footer.yml into the template context."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import yaml

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig


def on_config(config: MkDocsConfig, **kwargs: Any) -> MkDocsConfig:
    """Load footer column definitions from footer.yml into config.extra.

    The footer partial (``docs/overrides/partials/footer.html``) is a theme
    template and can only read ``config.extra``, so the standalone data file is
    injected here at build time.

    Args:
        config: The MkDocs configuration object.
        **kwargs: Additional keyword arguments.

    Returns:
        The configuration object with ``extra.footer`` populated.
    """
    footer_file = Path(config["config_file_path"]).parent / "docs" / "footer.yml"
    config["extra"]["footer"] = yaml.safe_load(footer_file.read_text(encoding="utf-8"))
    return config
