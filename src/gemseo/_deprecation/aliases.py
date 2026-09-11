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

- `module_renames`: old fully-qualified module (or package) name -> new one, fully
  resolved (ancestor package renames applied), from the `modules:` section.
- `attribute_renames`: old module fully-qualified name -> {old attribute: new
  attribute}, from the `attributes:` section.
- `manual_migrations`: old module fully-qualified name -> {old attribute: how to
  migrate}, from the `manual:` section, for the names whose migration cannot be
  automated.
- `live_aliased_modules`: the modules of `attribute_renames` that were not renamed.
- `dissolved_packages`: old package name -> the ordered new locations of its former
  submodules, for the packages listed in the `dissolved:` section.
- `class_attribute_renames`: class name -> {old attribute: new attribute}, from the
  `classes:` section, keyed by the class's current name (resolved through
  `attribute_renames` when the class itself was also renamed).

Accumulation across releases is achieved by keeping the old entries in
`bump-version.yml` until their scheduled removal.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

if TYPE_CHECKING:
    from collections.abc import Mapping

_config_path: Final[Path] = Path(__file__).parent / "bump-version.yml"

_package_prefix: Final[str] = "gemseo."


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
    if new_value.startswith(_package_prefix):
        return new_value
    return f"{_parent(old_name)}.{new_value}"


def _apply_longest_prefix(name: str, mapping: Mapping[str, str]) -> str:
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


def _group_by_module(entries: dict[str, str]) -> Mapping[str, Mapping[str, str]]:
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
    return MappingProxyType({
        module: MappingProxyType(entry) for module, entry in renames.items()
    })


def _parse_classes_section(text: str) -> dict[str, dict[str, str]]:
    """Parse the nested `classes:` section of the configuration file.

    Unlike `_parse_section`, entries here are nested under the class they rename an
    attribute of, and a method may itself nest codemod-only parameter renames, some
    of which carry a codemod-only value-conversion note (e.g. `new_name={old}_Settings`
    or `convert from dict to _Settings(**arg_value)`) instead of a plain new name;
    only the class-level attribute (and method name) renames are read here:

    - The section starts after a line that is exactly `classes:` and ends at the
      first non-blank line not starting with two spaces.
    - A line indented exactly 2 spaces of the form `ClassName:` (empty value, or a
      value that only tags the mapping with a YAML anchor, e.g. `Foo: &anchor`)
      opens that class; any other 2-space line (a YAML alias reference this simple
      parser does not resolve, e.g. `Bar: *anchor`) closes the current class
      instead, so that the entries following it are not misattributed.
    - A line indented exactly 4 spaces of the form `old_name: new_name` is an
      attribute rename for the current class, unless there is none, and `new_name`
      is a plain identifier (a YAML anchor or alias reference, and a codemod-only
      value-conversion note, are not, and are dropped like a removal is).
    - A 4-space line with an empty (or anchor-only) value (e.g. `__init__:`) opens a
      nested, codemod-only block; it and every line indented 6 spaces or more are
      ignored.
    - An entry whose value is `null` is dropped, as a removal cannot be aliased.
    - A class left with no entry is dropped.

    Args:
        text: The content of the configuration file.

    Returns:
        The mapping from class name to {old attribute name: new attribute name}.
    """
    classes: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    in_section = False
    for line in text.splitlines():
        stripped = line.strip()
        if not in_section:
            in_section = stripped == "classes:"
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith("  "):
            break
        indent = len(line) - len(line.lstrip(" "))
        if indent == 2:
            name, _, value = stripped.partition(":")
            if value.strip() and not value.strip().startswith("&"):
                current = None
            else:
                current = classes.setdefault(name.strip(), {})
            continue
        if indent != 4 or current is None:
            continue
        old_name, _, value = stripped.partition(":")
        value = value.strip()
        if value == "null" or not value.isidentifier():
            continue
        current[old_name.strip()] = value
    return {name: entries for name, entries in classes.items() if entries}


def _build() -> tuple[
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
    tuple[str, ...],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, str]],
]:
    """Build the alias tables from the configuration.

    Returns:
        The resolved module renames, the attribute renames grouped by old module,
        the names of the dissolved packages, the manual migrations grouped by old
        module and the class-attribute renames grouped by the class's current name.
    """
    text = _config_path.read_text(encoding="utf-8")

    raw = {
        old: _raw_new_module(old, value)
        for old, value in _parse_section(text, "modules").items()
    }
    module_renames: dict[str, str] = {}
    for old in raw:
        new = _resolve(old, raw)
        if new != old:
            module_renames[old] = new

    attribute_entries = _parse_section(text, "attributes")

    # Old bare class name -> new bare class name. The `classes:` section keys its
    # blocks by the class's OLD name (the codemod matches it against user code
    # written before the rename), which may differ from the class's current name
    # when the class itself was also renamed; that case is recorded among the
    # attribute renames as a plain rename of the class's own (possibly
    # fully-qualified) old name, so it is resolved here before exposure.
    class_renames = {
        _last(old): _last(new) if "." in new else new
        for old, new in attribute_entries.items()
        if _last(old) != (_last(new) if "." in new else new)
    }
    classes: dict[str, dict[str, str]] = {}
    for name, entries in _parse_classes_section(text).items():
        classes.setdefault(class_renames.get(name, name), {}).update(entries)

    return (
        MappingProxyType(module_renames),
        _group_by_module(attribute_entries),
        tuple(_parse_section(text, "dissolved")),
        _group_by_module(_parse_section(text, "manual")),
        MappingProxyType({
            name: MappingProxyType(entries) for name, entries in classes.items()
        }),
    )


_tables: Final[
    tuple[
        Mapping[str, str],
        Mapping[str, Mapping[str, str]],
        tuple[str, ...],
        Mapping[str, Mapping[str, str]],
        Mapping[str, Mapping[str, str]],
    ]
] = _build()

# Old fully-qualified module name -> new one (fully resolved).
module_renames: Final[Mapping[str, str]] = _tables[0]

# Old module name -> {old attribute name: new attribute name}.
attribute_renames: Final[Mapping[str, Mapping[str, str]]] = _tables[1]

# Old module name -> {old attribute name: how to migrate}, for the names that cannot be
# aliased to a new one, as the latter does not behave as the old one on its own.
manual_migrations: Final[Mapping[str, Mapping[str, str]]] = _tables[3]

# The old modules that kept their name: they are loaded by the normal import machinery,
# so their old attribute names have to be aliased in their own namespace instead of
# being resolved through a stand-in module.
live_aliased_modules: Final[frozenset[str]] = frozenset(
    module
    for module in attribute_renames
    if _apply_longest_prefix(module, module_renames) == module
)

# Old packages dissolved into several new packages: old package name -> the
# ordered new locations of its former submodules, used to resolve attribute
# access on the old package itself.
dissolved_packages: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    old: tuple(
        dict.fromkeys(
            new
            for old_child, new in module_renames.items()
            if old_child.startswith(f"{old}.")
        )
    )
    for old in _tables[2]
})

# Class name -> {old attribute name: new attribute name}, keyed by the class's
# current name (resolved when the class itself was also renamed).
class_attribute_renames: Final[Mapping[str, Mapping[str, str]]] = _tables[4]
