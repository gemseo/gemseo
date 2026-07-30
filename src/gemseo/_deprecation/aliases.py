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
"""Old-to-new name tables for deprecated imports.

The tables are computed at import time from `bump-version.yml` (shipped in this
package; the rename map that also drives the external codemod):

- `MODULE_RENAMES`: old fully-qualified module (or package) name -> new one, fully
  resolved (ancestor package renames applied), from the `modules:` section.
- `ATTRIBUTE_RENAMES`: old module fully-qualified name -> {old attribute: new
  attribute}, from the `attributes:` section.
- `LIVE_ALIASED_MODULES`: the modules of `ATTRIBUTE_RENAMES` that were not renamed.
- `DISSOLVED_PACKAGES`: old package name -> the ordered new locations of its former
  submodules, for the packages listed in the `dissolved:` section.

Accumulation across releases is achieved by keeping the old entries in
`bump-version.yml` until their scheduled removal.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

_CONFIG_PATH: Final[Path] = Path(__file__).parent / "bump-version.yml"

_PACKAGE_PREFIX: Final[str] = "gemseo."


def _parse_section(text: str, name: str) -> dict[str, str]:
    """Parse a section of the configuration file.

    The sections read here are either a flat mapping of plain scalars
    (`old.qualified.name: new_name` lines indented by two spaces) or a flat list of
    plain scalars (`- item` lines, yielding an empty value), so a full YAML parser is
    not needed.

    Args:
        text: The content of the configuration file.
        name: The name of the section.

    Returns:
        The mapping from old fully-qualified name to new name, in file order.
    """
    entries: dict[str, str] = {}
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_section:
            in_section = stripped == f"{name}:"
            continue
        # Comments are skipped whatever their indentation, so that an unindented one
        # does not silently truncate the section.
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith("  "):
            # An unindented entry starts the next section.
            break
        old_name, _, new_name = stripped.removeprefix("- ").partition(":")
        entries[old_name.strip()] = new_name.strip()
    return entries


def _parent(name: str) -> str:
    """Return the dotted name without its last segment.

    Args:
        name: A dotted qualified name.

    Returns:
        The qualified name of the parent.
    """
    return name.rsplit(".", 1)[0]


def _last(name: str) -> str:
    """Return the last segment of a dotted qualified name.

    Args:
        name: A dotted qualified name.

    Returns:
        The last segment.
    """
    return name.rsplit(".", 1)[1]


def _raw_new_module(old_name: str, new_value: str) -> str:
    """Apply a single module-rename rule to its own key.

    Args:
        old_name: The old fully-qualified module name.
        new_value: The rule value: an absolute `gemseo.*` path replaces the whole
            name, otherwise it replaces the last segment (keeping the parent).

    Returns:
        The new fully-qualified module name, before ancestor renames are applied.
    """
    if new_value.startswith(_PACKAGE_PREFIX):
        return new_value
    return f"{_parent(old_name)}.{new_value}"


def _apply_longest_prefix(name: str, mapping: dict[str, str]) -> str:
    """Rewrite `name` using the mapping entry whose key is its longest prefix.

    Args:
        name: The fully-qualified name to rewrite.
        mapping: A mapping from old prefix to new prefix.

    Returns:
        The rewritten name, or `name` unchanged if no key is a prefix.
    """
    best = None
    for key in mapping:
        if (name == key or name.startswith(f"{key}.")) and (
            best is None or len(key) > len(best)
        ):
            best = key
    if best is None:
        return name
    return mapping[best] + name[len(best) :]


def _resolve(name: str, mapping: dict[str, str]) -> str:
    """Rewrite `name` repeatedly until it stops changing.

    Applies ancestor package renames on top of the module's own rename.

    Args:
        name: The fully-qualified name to resolve.
        mapping: A mapping from old prefix to new prefix.

    Returns:
        The fully-resolved new name.
    """
    seen = set()
    while name not in seen:
        seen.add(name)
        new_name = _apply_longest_prefix(name, mapping)
        if new_name == name:
            break
        name = new_name
    return name


def _group_by_module(entries: dict[str, str]) -> dict[str, dict[str, str]]:
    """Group attribute-rename entries by the module defining the attribute.

    The new name is kept as written: a bare name when the attribute stayed in the
    module, a fully-qualified one when it moved to another module.

    Args:
        entries: The mapping from old fully-qualified attribute name to new name.

    Returns:
        The mapping from old module name to {old attribute name: new name}.
    """
    renames: dict[str, dict[str, str]] = {}
    for old, value in entries.items():
        renames.setdefault(_parent(old), {})[_last(old)] = value
    return renames


def _build() -> tuple[dict[str, str], dict[str, dict[str, str]], tuple[str, ...]]:
    """Build the alias tables from the configuration.

    Returns:
        The resolved module renames, the attribute renames grouped by old module and
        the names of the dissolved packages.
    """
    text = _CONFIG_PATH.read_text(encoding="utf-8")

    raw = {
        old: _raw_new_module(old, value)
        for old, value in _parse_section(text, "modules").items()
    }
    module_renames: dict[str, str] = {}
    for old in raw:
        new = _resolve(old, raw)
        if new != old:
            module_renames[old] = new

    return (
        module_renames,
        _group_by_module(_parse_section(text, "attributes")),
        tuple(_parse_section(text, "dissolved")),
    )


_TABLES: Final[tuple[dict[str, str], dict[str, dict[str, str]], tuple[str, ...]]] = (
    _build()
)

# Old fully-qualified module name -> new one (fully resolved).
MODULE_RENAMES: Final[dict[str, str]] = _TABLES[0]

# Old module name -> {old attribute name: new attribute name}.
ATTRIBUTE_RENAMES: Final[dict[str, dict[str, str]]] = _TABLES[1]

# The old modules that kept their name: they are loaded by the normal import machinery,
# so their old attribute names have to be aliased in their own namespace instead of
# being resolved through a stand-in module.
LIVE_ALIASED_MODULES: Final[frozenset[str]] = frozenset(
    module
    for module in ATTRIBUTE_RENAMES
    if _apply_longest_prefix(module, MODULE_RENAMES) == module
)

# Old packages dissolved into several new packages: old package name -> the
# ordered new locations of its former submodules, used to resolve attribute
# access on the old package itself.
DISSOLVED_PACKAGES: Final[dict[str, tuple[str, ...]]] = {
    old: tuple(
        dict.fromkeys(
            new
            for old_child, new in MODULE_RENAMES.items()
            if old_child.startswith(f"{old}.")
        )
    )
    for old in _TABLES[2]
}
