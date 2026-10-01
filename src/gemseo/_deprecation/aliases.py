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
  attribute}, from the `attributes:` entries that rename a name.
- `manual_migrations`: old module fully-qualified name -> {old attribute: how to
  migrate}, from the `attributes:` entries whose value is `TODO <text>`, for the names
  whose migration cannot be automated. The text is an instruction telling how to
  migrate by hand, e.g. `use gemseo.x.New instead`, which ends the message of the error
  raised for the old name.
- `removed_modules`: the old fully-qualified names of the modules (and packages)
  removed with no replacement, the `modules:` entries whose value is `null`.
- `removed_attributes`: old module fully-qualified name -> its attributes removed with
  no replacement, the `attributes:` entries whose value is `null`.
- `unmigrated_attributes`: old module fully-qualified name -> its attributes that raise
  instead of resolving, those of `removed_attributes` and `manual_migrations`.
- `live_aliased_modules`: the modules of `attribute_renames`, `removed_attributes` and
  `manual_migrations` that were not renamed, whose old attribute names are resolved in
  their own namespace.
- `dissolved_packages`: old package name -> the ordered new locations of its former
  submodules, for the old packages that have no rename of their own and no longer exist
  in the tree, but whose submodules were renamed. They are derived from
  `module_renames`, not listed in the configuration.
- `class_attribute_renames`: class name -> {old attribute: new attribute}, from the
  `classes:` section, keyed by the class's current name (resolved through
  `attribute_renames` when the class itself was also renamed). An old attribute name
  may be dotted, e.g. `DifferentiationMethod.USER_GRAD`, for the renamed attribute of
  a class nested in the class, e.g. a member of an enumeration.

Accumulation across releases is achieved by keeping the old entries in
`bump-version.yml` until their scheduled removal.
"""

from __future__ import annotations

import sys
from pathlib import Path
from pkgutil import get_importer
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

if TYPE_CHECKING:
    from collections.abc import Container
    from collections.abc import Iterable
    from collections.abc import Mapping
    from importlib.machinery import ModuleSpec

_config_path: Final[Path] = Path(__file__).parent / "bump-version.yml"

_package_prefix: Final[str] = "gemseo."

_plugin_prefix: Final[str] = "gemseo_"
"""The prefix of the name of the package of a GEMSEO plugin, e.g. `gemseo_excel`."""

_removed: Final[str] = "null"
"""The value of an entry whose old name was removed with no replacement."""

_todo_prefix: Final[str] = "TODO "
"""The prefix of the value of an entry whose old name was removed, to migrate by hand.

The rest of the value is an instruction telling how to migrate, e.g.
`TODO use gemseo.x.New instead`.
"""


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


def _get_todo_text(value: str) -> str:
    """Return the instruction of a `TODO <text>` value of a rule.

    Such a value tells that the old name was removed, and how to migrate by hand the
    code using it: `<text>` is what follows `TODO `, stripped, as the external codemod
    reads it. A `TODO` with no text, and a `todo` in lower case, are not such a value.

    Args:
        value: The value of a rule.

    Returns:
        The text, or an empty string if `value` is not a `TODO <text>` value.
    """
    if value.startswith(_todo_prefix):
        return value.removeprefix(_todo_prefix).strip()
    return ""


def _check_rule_value(section: str, old_name: str, new_value: str) -> None:
    """Check that the value of a rename rule is a complete dotted path.

    Every rule value is the complete new dotted path, never a bare last segment, so
    that a relative value added by mistake fails loudly instead of silently resolving
    to a name nobody wrote. A `modules:` rule value is in GEMSEO (`gemseo.*`) or, for a
    module moved to a plugin, in the package of that plugin (`gemseo_*.*`, e.g.
    `gemseo_excel.xls_discipline`). An `attributes:` rule value may be in any package,
    e.g. `scipy.sparse.sparray` or, for an old name outside GEMSEO, which only the
    codemod reads, `enum.StrEnum`.

    Args:
        section: The name of the section of the rule, `modules` or `attributes`.
        old_name: The old fully-qualified name, for the error message only.
        new_value: The rule value.

    Raises:
        ValueError: If `new_value` is not an absolute `gemseo.*` or `gemseo_*.*` path
            for a `modules:` rule, or not a dotted path of identifiers for an
            `attributes:` rule.
    """
    if section == "modules":
        if _is_absolute(new_value):
            return
        location = ", in GEMSEO or in a plugin"
    elif _is_dotted(new_value):
        return
    else:
        location = ""
    msg = (
        f"The {section}: rule for {old_name!r} has the value {new_value!r}; "
        f"it must be the complete new dotted path{location}."
    )
    raise ValueError(msg)


def _is_absolute(name: str) -> bool:
    """Return whether a rule value is a complete dotted path, in GEMSEO or in a plugin.

    Args:
        name: The rule value.

    Returns:
        Whether `name` starts with `gemseo.`, or with the package of a GEMSEO plugin
        (`gemseo_*`) followed by a dot.
    """
    package, dot, _ = name.partition(".")
    return name.startswith(_package_prefix) or bool(
        dot and package.startswith(_plugin_prefix) and package.isidentifier()
    )


def _is_dotted(name: str) -> bool:
    """Return whether a rule value is a dotted path of identifiers, in any package.

    Args:
        name: The rule value.

    Returns:
        Whether `name` has at least two segments, all of them identifiers.
    """
    segments = name.split(".")
    return len(segments) > 1 and all(segment.isidentifier() for segment in segments)


def _apply_longest_prefix(name: str, mapping: Mapping[str, str]) -> str:
    """Rewrite `name` using the mapping entry whose key is its longest prefix.

    The dotted prefixes of `name` are looked up in `mapping`, the longest first, so
    that the cost does not depend on the size of `mapping`.

    Args:
        name: The fully-qualified name to rewrite.
        mapping: A mapping from old prefix to new prefix, with no empty key.

    Returns:
        The rewritten name, or `name` unchanged if no key is a prefix.
    """
    prefix = name
    while prefix:
        new_prefix = mapping.get(prefix)
        if new_prefix is not None:
            return new_prefix + name[len(prefix) :]
        prefix = prefix.rpartition(".")[0]
    return name


def _resolve(name: str, mapping: Mapping[str, str]) -> str:
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
    """Group attribute entries by the module defining the attribute.

    The value is kept as written, e.g. the new name of a rename, fully qualified.

    Args:
        entries: The mapping from old fully-qualified attribute name to its value.

    Returns:
        The mapping from old module name to {old attribute name: value}.
    """
    renames: dict[str, dict[str, str]] = {}
    for old, value in entries.items():
        renames.setdefault(_parent(old), {})[_last(old)] = value
    return MappingProxyType({
        module: MappingProxyType(entry) for module, entry in renames.items()
    })


def _group_removed_attributes(
    entries: Mapping[str, str],
) -> MappingProxyType[str, frozenset[str]]:
    """Group the removed attributes by the module defining them.

    Args:
        entries: The mapping from old fully-qualified attribute name to new name.

    Returns:
        The mapping from old module name to the names of its attributes removed with no
        replacement.
    """
    removed: dict[str, set[str]] = {}
    for old, value in entries.items():
        if value == _removed:
            removed.setdefault(_parent(old), set()).add(_last(old))
    return MappingProxyType({
        module: frozenset(names) for module, names in removed.items()
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
      attribute rename for the current class, unless there is none, and the entry is
      aliasable (see
      [_is_aliasable_class_entry][gemseo._deprecation.aliases._is_aliasable_class_entry];
      a YAML anchor or alias reference, and a codemod-only value-conversion note, are
      not, and are dropped like a removal is).
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
        old_name = old_name.strip()
        value = value.strip()
        if value == _removed or not _is_aliasable_class_entry(old_name, value):
            continue
        current[old_name] = value
    return {name: entries for name, entries in classes.items() if entries}


def _is_aliasable_class_entry(old_name: str, new_name: str) -> bool:
    """Return whether an entry of the `classes:` section is an aliasable rename.

    Either both names are plain identifiers, for an attribute of the class itself, or
    both are dotted paths of identifiers differing only by their last segment, for an
    attribute of a class nested in the class, e.g. the member of an enumeration:
    `DifferentiationMethod.USER_GRAD: DifferentiationMethod.USER`.

    Args:
        old_name: The old name of the attribute.
        new_name: The new name of the attribute.

    Returns:
        Whether the entry is an aliasable rename.
    """
    old_path = old_name.split(".")
    new_path = new_name.split(".")
    return old_path[:-1] == new_path[:-1] and all(
        segment.isidentifier() for segment in (*old_path, new_path[-1])
    )


def _find_dissolved_packages(
    module_renames: Mapping[str, str],
) -> MappingProxyType[str, tuple[str, ...]]:
    """Find the old packages dissolved into other packages.

    An old package is dissolved when it has no rename of its own but some of its
    submodules were renamed, and it no longer exists. Precisely, a package `P` is
    dissolved when:

    - `P` is a proper dotted ancestor of a key of `module_renames`,
    - neither `P` nor one of its ancestors is a key,
    - `P` is not the root package,
    - no module or regular package `P` is found (see
      [_find_module_spec][gemseo._deprecation.aliases._find_module_spec]).

    Args:
        module_renames: The old module name -> new module name mapping, fully resolved.

    Returns:
        The old package name -> the ordered new locations of its former submodules.
    """
    new_locations: dict[str, dict[str, None]] = {}
    for old, new in module_renames.items():
        parts = old.split(".")
        for i in range(2, len(parts)):
            new_locations.setdefault(".".join(parts[:i]), {})[new] = None
    return MappingProxyType({
        package: tuple(locations)
        for package, locations in new_locations.items()
        if not _is_renamed(package, module_renames)
        and _find_module_spec(package) is None
    })


def _is_renamed(name: str, module_renames: Mapping[str, str]) -> bool:
    """Return whether a module is renamed, by a rule of its own or of an ancestor.

    Args:
        name: The fully-qualified module name.
        module_renames: The old module name -> new module name mapping.

    Returns:
        Whether `name` or one of its ancestors is a key of `module_renames`.
    """
    return find_ancestor(name, module_renames) is not None


def find_ancestor(name: str, names: Container[str]) -> str | None:
    """Return the outermost of a module and its ancestor packages found among names.

    Args:
        name: The fully-qualified module name.
        names: The fully-qualified module names to look for `name` and its ancestors
            in.

    Returns:
        The outermost of `name` and its ancestor packages that is in `names`, or `None`
        if there is none.
    """
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        ancestor = ".".join(parts[:i])
        if ancestor in names:
            return ancestor
    return None


def _find_module_spec(name: str) -> ModuleSpec | None:
    """Return the spec of a module or regular package, without importing anything.

    Unlike [importlib.util.find_spec][], which imports the parent packages, each
    segment of `name` is looked up by the path entry finders of the search locations
    of the previous one, as [pkgutil.iter_modules][] does, so that the lookup neither
    runs code nor depends on a source tree, e.g. for modules in a zip archive. A
    namespace package portion, e.g. a leftover directory holding no `__init__.py` but a
    `__pycache__`, is not a regular package and is not found.

    Args:
        name: The fully-qualified module name.

    Returns:
        The spec of the module, or `None` if it does not exist.
    """
    parts = name.split(".")
    spec = getattr(sys.modules.get(parts[0]), "__spec__", None)
    if spec is None:
        spec = _find_spec_in_locations(parts[0], sys.path)
    for i in range(2, len(parts) + 1):
        if spec is None or spec.submodule_search_locations is None:
            return None
        spec = _find_spec_in_locations(
            ".".join(parts[:i]), spec.submodule_search_locations
        )
    return spec


def _find_spec_in_locations(name: str, locations: Iterable[str]) -> ModuleSpec | None:
    """Return the spec of a module or regular package found in search locations.

    Args:
        name: The fully-qualified module name.
        locations: The search locations, e.g. the directories of the parent package.

    Returns:
        The spec of the module, or `None` if no path entry finder of `locations` finds
        it as a module or a regular package.
    """
    for location in locations:
        find_spec = getattr(get_importer(location), "find_spec", None)
        if find_spec is not None:
            spec = find_spec(name)
            # A spec with no loader is that of a namespace package portion.
            if spec is not None and spec.loader is not None:
                return spec
    return None


def _find_live_modules(
    module_renames: Mapping[str, str], *module_names: Iterable[str]
) -> frozenset[str]:
    """Find the modules that were not renamed.

    Args:
        module_renames: The old module name -> new module name mapping.
        *module_names: The fully-qualified module names.

    Returns:
        The modules of `module_names` that are not renamed, by a rule of their own or
        of an ancestor.
    """
    return frozenset(
        module
        for names in module_names
        for module in names
        if not _is_renamed(module, module_renames)
    )


def _build() -> tuple[
    Mapping[str, str],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, str]],
    Mapping[str, Mapping[str, str]],
    frozenset[str],
    Mapping[str, frozenset[str]],
]:
    """Build the alias tables from the configuration.

    Returns:
        The resolved module renames, the attribute renames grouped by old module,
        the manual migrations grouped by old module, the class-attribute renames
        grouped by the class's current name, the removed modules and the removed
        attributes grouped by old module.
    """
    text = _config_path.read_text(encoding="utf-8")

    module_entries = _parse_section(text, "modules")
    removed_modules = frozenset(
        old for old, value in module_entries.items() if value == _removed
    )
    raw = {old: value for old, value in module_entries.items() if value != _removed}
    for old, value in raw.items():
        _check_rule_value("modules", old, value)
    module_renames: dict[str, str] = {}
    for old in raw:
        new = _resolve(old, raw)
        if new != old:
            module_renames[old] = new

    all_attribute_entries = _parse_section(text, "attributes")
    manual_entries = {
        old: todo_text
        for old, value in all_attribute_entries.items()
        if (todo_text := _get_todo_text(value))
    }
    # Neither a removal nor a manual migration is a rename.
    attribute_entries = {
        old: value
        for old, value in all_attribute_entries.items()
        if value != _removed and old not in manual_entries
    }
    for old, value in attribute_entries.items():
        _check_rule_value("attributes", old, value)

    # Old bare class name -> new bare class name. The `classes:` section keys its
    # blocks by the class's OLD name (the codemod matches it against user code
    # written before the rename), which may differ from the class's current name
    # when the class itself was also renamed; that case is recorded among the
    # attribute renames as a plain rename of the class's own (possibly
    # fully-qualified) old name, so it is resolved here before exposure.
    class_renames = {
        _last(old): _last(new)
        for old, new in attribute_entries.items()
        if _last(old) != _last(new)
    }
    classes: dict[str, dict[str, str]] = {}
    for name, entries in _parse_classes_section(text).items():
        classes.setdefault(class_renames.get(name, name), {}).update(entries)

    return (
        MappingProxyType(module_renames),
        _group_by_module(attribute_entries),
        _group_by_module(manual_entries),
        MappingProxyType({
            name: MappingProxyType(entries) for name, entries in classes.items()
        }),
        removed_modules,
        _group_removed_attributes(all_attribute_entries),
    )


_tables: Final[
    tuple[
        Mapping[str, str],
        Mapping[str, Mapping[str, str]],
        Mapping[str, Mapping[str, str]],
        Mapping[str, Mapping[str, str]],
        frozenset[str],
        Mapping[str, frozenset[str]],
    ]
] = _build()

module_renames: Final[Mapping[str, str]] = _tables[0]
"""Old fully-qualified module name -> new one (fully resolved)."""

attribute_renames: Final[Mapping[str, Mapping[str, str]]] = _tables[1]
"""Old module name -> {old attribute name: new attribute name}."""

manual_migrations: Final[Mapping[str, Mapping[str, str]]] = _tables[2]
"""Old module name -> {old attribute name: how to migrate}.

For the names that cannot be aliased to a new one, as the latter does not behave as the
old one on its own. The text is the one of the `TODO <text>` value of the `attributes:`
entry, an instruction that reads after "was removed;" and before a period.
"""

removed_modules: Final[frozenset[str]] = _tables[4]
"""The old modules and packages removed with no replacement; their submodules too."""

removed_attributes: Final[Mapping[str, frozenset[str]]] = _tables[5]
"""Old module name -> the names of its attributes removed with no replacement."""

unmigrated_attributes: Final[Mapping[str, frozenset[str]]] = MappingProxyType({
    module: removed_attributes.get(module, frozenset()).union(
        manual_migrations.get(module, {})
    )
    for module in (*removed_attributes, *manual_migrations)
})
"""Old module name -> the names of its attributes that raise instead of resolving.

They are the union of its attributes of `removed_attributes` and `manual_migrations`.
"""

live_aliased_modules: Final[frozenset[str]] = _find_live_modules(
    module_renames, attribute_renames, removed_attributes, manual_migrations
)
"""The old modules with renamed, removed or manual attributes that kept their name.

They are loaded by the normal import machinery, so their old attribute names have to be
aliased in their own namespace instead of being resolved through a stand-in module.
"""

dissolved_packages: Final[Mapping[str, tuple[str, ...]]] = _find_dissolved_packages(
    module_renames
)
"""Old packages dissolved into several new packages.

Old package name -> the ordered new locations of its former submodules. Derived from
`module_renames`, used to import the old package itself as a stand-in through which its
submodules are redirected.
"""

class_attribute_renames: Final[Mapping[str, Mapping[str, str]]] = _tables[3]
"""Class name -> {old attribute name: new attribute name}.

Keyed by the class's current name (resolved when the class itself was also renamed).
"""
