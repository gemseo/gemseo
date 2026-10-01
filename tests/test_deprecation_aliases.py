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
"""Unit tests for the alias-table builder behind the deprecated-import machinery."""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import pytest
import yaml

import gemseo
from gemseo._deprecation import aliases
from gemseo.util.testing.helper import assert_exception


def test_parse_section_skips_comments_and_blank_lines():
    """Comment and blank lines inside a section are ignored."""
    text = "modules:\n  # a comment\n\n  old.a: new_a\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_skips_unindented_comments():
    """An unindented comment does not truncate a section."""
    text = "modules:\n  old.a: new_a\n# an unindented comment\n  old.b: new_b\n"
    assert aliases._parse_section(text, "modules") == {
        "old.a": "new_a",
        "old.b": "new_b",
    }


def test_parse_section_stops_before_next_section():
    """Parsing stops at the first unindented line following the section name."""
    text = "modules:\n  old.a: new_a\nclasses:\n  Foo: bar\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_reaches_end_of_text():
    """Parsing consumes the whole text when the section is the last one."""
    text = "modules:\n  old.a: new_a\n"
    assert aliases._parse_section(text, "modules") == {"old.a": "new_a"}


def test_parse_section_skips_other_sections():
    """Only the requested section is parsed."""
    text = "modules:\n  old.a: new_a\nattributes:\n  old.b.Old: New\n"
    assert aliases._parse_section(text, "attributes") == {"old.b.Old": "New"}


def test_parse_section_list_items():
    """A list section yields its items with an empty value."""
    text = "listed:\n  - old.a\n  - old.b\n"
    assert aliases._parse_section(text, "listed") == {"old.a": "", "old.b": ""}


def test_parent():
    """The parent of a dotted name drops its last segment."""
    assert aliases._parent("a.b.c") == "a.b"


def test_last():
    """The last segment of a dotted name is returned."""
    assert aliases._last("a.b.c") == "c"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("TODO use gemseo.x.New instead", "use gemseo.x.New instead"),
        # The text is kept as written, stripped.
        ("TODO  use x.New, which has y, instead ", "use x.New, which has y, instead"),
        # A `TODO` with no text is no instruction.
        ("TODO", ""),
        ("TODO ", ""),
        # Neither is a `todo` in lower case, a word starting with `TODO`, a new name
        # or a removal.
        ("todo use gemseo.x.New instead", ""),
        ("TODOS use gemseo.x.New instead", ""),
        ("gemseo.x.New", ""),
        ("null", ""),
    ],
)
def test_get_todo_text(value, expected):
    """The instruction of a `TODO <text>` value is the text following `TODO `.

    Args:
        value: The value of a rule.
        expected: The expected instruction, empty for a value that is not a
            `TODO <text>` one.
    """
    assert aliases._get_todo_text(value) == expected


@pytest.mark.parametrize("section", ["modules", "attributes"])
@pytest.mark.parametrize("value", ["gemseo.x.y", "gemseo_excel.xls_discipline"])
def test_check_rule_value_accepts_an_absolute_path(section, value):
    """An absolute value, in GEMSEO or in a plugin, is accepted.

    Args:
        section: The name of the section of the rule.
        value: The rule value.
    """
    aliases._check_rule_value(section, "gemseo.a.b", value)


@pytest.mark.parametrize(
    "old_name",
    [
        # An attribute of GEMSEO replaced by a third-party one.
        "gemseo.a.Old",
        # An attribute outside GEMSEO and its plugins.
        "strenum.StrEnum",
    ],
)
def test_check_rule_value_accepts_a_third_party_attribute_value(old_name):
    """The value of an attribute may be a dotted path in any package.

    Args:
        old_name: The old name of the attribute.
    """
    aliases._check_rule_value("attributes", old_name, "scipy.sparse.sparray")


def test_check_module_value_rejects_a_third_party_value(snapshot):
    """The value of a module is in GEMSEO or in a plugin.

    Args:
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with assert_exception(ValueError, snapshot):
        aliases._check_rule_value("modules", "gemseo.a.b", "scipy.sparse")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scipy.sparse.sparray", True),
        ("gemseo.a", True),
        # A bare name, or a dotted name with a segment that is not an identifier.
        ("sparray", False),
        (".sparray", False),
        ("scipy..sparray", False),
        ("scipy.sparse.not an identifier", False),
    ],
)
def test_is_dotted(name, expected):
    """A value is dotted when it has several segments, all identifiers.

    Args:
        name: The rule value.
        expected: Whether the value is dotted.
    """
    assert aliases._is_dotted(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gemseo.a.b", True),
        ("gemseo_excel.xls_discipline", True),
        # A plugin package alone, or a name merely starting like one, is not a path.
        ("gemseo_excel", False),
        ("gemseo_", False),
        ("xls_discipline", False),
        ("chain.b", False),
    ],
)
def test_is_absolute(name, expected):
    """A value is complete when it is in GEMSEO or in the package of a plugin."""
    assert aliases._is_absolute(name) is expected


def test_check_module_value_rejects_a_relative_value(snapshot):
    """A bare (non-absolute) value is rejected: every rule value is absolute."""
    with assert_exception(ValueError, snapshot):
        aliases._check_rule_value("modules", "gemseo.a.b", "c")


@pytest.mark.parametrize(
    "old_name",
    [
        "gemseo.a.Old",
        # Even outside GEMSEO and its plugins, a value is a dotted path.
        "strenum.StrEnum",
    ],
)
def test_check_attribute_value_rejects_a_relative_value(old_name, snapshot):
    """A bare value of the `attributes:` section is rejected.

    Args:
        old_name: The old name of the attribute.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    with assert_exception(ValueError, snapshot):
        aliases._check_rule_value("attributes", old_name, "New")


def test_build_rejects_a_relative_attribute_value(tmp_path, monkeypatch, snapshot):
    """`_build` rejects a bare `attributes:` value instead of crashing.

    Args:
        tmp_path: Fixture giving a temporary directory for the configuration.
        monkeypatch: Fixture to patch the path of the configuration.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    config = tmp_path / "bump-version.yml"
    config.write_text("attributes:\n  gemseo.x.Old: New\n", encoding="utf-8")
    monkeypatch.setattr(aliases, "_config_path", config)

    with assert_exception(ValueError, snapshot):
        aliases._build()


@pytest.mark.parametrize(
    ("section", "value"),
    [
        # A `TODO` with no text is no instruction.
        ("attributes", "TODO"),
        # A `TODO <text>` value is read in `attributes:` only.
        ("modules", "TODO use gemseo.y instead"),
    ],
    ids=["no-text", "module"],
)
def test_build_rejects_a_value_that_is_no_manual_migration(
    section, value, tmp_path, monkeypatch, snapshot
):
    """`_build` rejects a `TODO` value that is not a manual migration of `attributes:`.

    Such a value is not the complete new dotted path of a rename either.

    Args:
        section: The name of the section of the rule.
        value: The rule value.
        tmp_path: Fixture giving a temporary directory for the configuration.
        monkeypatch: Fixture to patch the path of the configuration.
        snapshot: Fixture to compare the error message with a snapshot.
    """
    config = tmp_path / "bump-version.yml"
    config.write_text(f"{section}:\n  gemseo.x: {value}\n", encoding="utf-8")
    monkeypatch.setattr(aliases, "_config_path", config)

    with assert_exception(ValueError, snapshot):
        aliases._build()


def test_build_does_not_rename_a_todo_value(tmp_path, monkeypatch):
    """`_build` reads a `TODO <text>` value of `attributes:` as a manual migration.

    It is neither a rename, nor checked as the dotted path of a new name, nor the
    rename of the class which the `classes:` section is keyed on.

    Args:
        tmp_path: Fixture giving a temporary directory for the configuration.
        monkeypatch: Fixture to patch the path of the configuration.
    """
    config = tmp_path / "bump-version.yml"
    config.write_text(
        "attributes:\n"
        "  gemseo.old_pkg.Old: TODO use gemseo.new_pkg.New instead\n"
        "classes:\n"
        "  Old:\n"
        "    OLD_ATTRIBUTE: new_attribute\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_config_path", config)

    (
        module_renames,
        attribute_renames,
        manual_migrations,
        class_attribute_renames,
        removed_modules,
        removed_attributes,
    ) = aliases._build()

    assert manual_migrations == {
        "gemseo.old_pkg": {"Old": "use gemseo.new_pkg.New instead"}
    }
    assert not attribute_renames
    assert not removed_attributes
    assert not module_renames
    assert not removed_modules
    # The class is not renamed, so its entries stay keyed on its name.
    assert class_attribute_renames == {"Old": {"OLD_ATTRIBUTE": "new_attribute"}}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("gemseo.a.b.c", "gemseo.a"),
        ("gemseo.a", "gemseo.a"),
        ("gemseo.a_b", None),
        ("gemseo", None),
    ],
)
def test_find_ancestor(name, expected):
    """The outermost of a module and its ancestors among names is found.

    Args:
        name: The fully-qualified module name.
        expected: The expected ancestor.
    """
    assert aliases.find_ancestor(name, frozenset({"gemseo.a", "gemseo.a.b"})) == (
        expected
    )


def test_manual_migrations_are_the_todo_values_of_the_configuration():
    """The manual migrations are the `TODO <text>` values of the `attributes:` section.

    The configuration is read as YAML, as the external codemod does, to check that the
    runtime, which reads it line by line, reads the same values from it. No `manual:`
    section is left, which nothing would read.
    """
    config = yaml.safe_load(aliases._config_path.read_text(encoding="utf-8"))

    todo_values = {
        old: value
        for old, value in config["attributes"].items()
        if isinstance(value, str) and value.startswith("TODO")
    }
    manual_migrations = {
        f"{module}.{name}": f"TODO {text}"
        for module, migrations in aliases.manual_migrations.items()
        for name, text in migrations.items()
    }
    assert manual_migrations
    assert manual_migrations == todo_values
    assert "manual" not in config


def test_manual_migrations_of_the_configuration_end_without_a_period():
    """The instruction of a manual migration ends the message of the error.

    The message adds the final period.
    """
    texts = [
        text
        for migrations in aliases.manual_migrations.values()
        for text in migrations.values()
    ]
    assert texts
    assert not [text for text in texts if text.endswith(".")]


def test_unmigrated_attributes_of_the_configuration():
    """The unmigrated attributes are the removed ones and the manual migrations."""
    modules = {*aliases.removed_attributes, *aliases.manual_migrations}
    assert set(aliases.unmigrated_attributes) == modules
    for module in modules:
        assert aliases.unmigrated_attributes[module] == {
            *aliases.removed_attributes.get(module, ()),
            *aliases.manual_migrations.get(module, {}),
        }


def test_apply_longest_prefix_picks_longest_match():
    """The mapping entry whose key is the longest prefix wins."""
    mapping = {"a": "z", "a.b": "y"}
    assert aliases._apply_longest_prefix("a.b.c", mapping) == "y.c"


def test_apply_longest_prefix_no_match():
    """The name is returned unchanged when no key is a prefix of it."""
    mapping = {"a": "z"}
    assert aliases._apply_longest_prefix("other", mapping) == "other"


def test_apply_longest_prefix_matches_whole_segments():
    """A key is a prefix of a name only when it is made of whole segments of it."""
    mapping = {"a": "z", "a.b": "y"}
    assert aliases._apply_longest_prefix("ab.c", mapping) == "ab.c"
    assert aliases._apply_longest_prefix("a.bc", mapping) == "z.bc"


def test_find_live_modules():
    """The live modules are those not renamed, by a rule of their own or an ancestor.

    The modules of every table are considered, e.g. one with manual migrations only.
    """
    module_renames = {"gemseo.old": "gemseo.new", "gemseo.a.old": "gemseo.a.new"}

    live_modules = aliases._find_live_modules(
        module_renames,
        ["gemseo.a", "gemseo.old.b"],
        ["gemseo.a.old", "gemseo.c"],
        {"gemseo.manual_only": {"Old": "use New with an extra argument instead"}},
    )

    assert live_modules == {"gemseo.a", "gemseo.c", "gemseo.manual_only"}


def test_resolve_applies_ancestor_renames_repeatedly():
    """Resolution keeps rewriting until an ancestor rename also applies."""
    mapping = {"a": "b", "b.c": "b.d"}
    assert aliases._resolve("a.c", mapping) == "b.d"


def test_resolve_cycle_guard_terminates():
    """A rename cycle terminates via the `seen` guard instead of looping forever."""
    mapping = {"a": "b", "b": "a"}
    assert aliases._resolve("a", mapping) == "a"


def test_build_from_synthetic_config(tmp_path, monkeypatch):
    """`_build` resolves the renames of modules, attributes and class attributes.

    It also reads the removals, the entries whose value is `null`, and the manual
    migrations, the entries of `attributes:` whose value is `TODO <text>`.
    """
    config = tmp_path / "bump-version.yml"
    config.write_text(
        "modules:\n"
        "  gemseo.old_pkg.old_mod: gemseo.new_pkg.new_mod\n"
        "  gemseo.old_pkg: gemseo.new_pkg\n"
        "  gemseo.same.name: gemseo.same.name\n"
        "  gemseo.old_pkg.gone_mod: null\n"
        "attributes:\n"
        "  gemseo.old_pkg.Old: gemseo.new_pkg.New\n"
        "  gemseo.old_pkg.old_mod.old_func: gemseo.new_pkg.new_mod.new_func\n"
        "  gemseo.old_pkg.old_mod.OldClass: gemseo.new_pkg.new_mod.NewClass\n"
        "  gemseo.old_pkg.Gone: null\n"
        "  gemseo.old_pkg.Manual: TODO use New with an extra argument instead\n"
        "classes:\n"
        "  Foo:\n"
        "    OLD: new\n"
        "  OldClass:\n"
        "    OTHER_OLD: other_new\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(aliases, "_config_path", config)

    (
        module_renames,
        attribute_renames,
        manual,
        class_attribute_renames,
        removed_modules,
        removed_attributes,
    ) = aliases._build()

    assert module_renames["gemseo.old_pkg"] == "gemseo.new_pkg"
    assert module_renames["gemseo.old_pkg.old_mod"] == "gemseo.new_pkg.new_mod"
    # A no-op entry is dropped.
    assert "gemseo.same.name" not in module_renames
    # A null entry is a removal, not a rename.
    assert "gemseo.old_pkg.gone_mod" not in module_renames
    assert removed_modules == {"gemseo.old_pkg.gone_mod"}
    assert removed_attributes == {"gemseo.old_pkg": {"Gone"}}
    assert attribute_renames == {
        "gemseo.old_pkg": {"Old": "gemseo.new_pkg.New"},
        "gemseo.old_pkg.old_mod": {
            "old_func": "gemseo.new_pkg.new_mod.new_func",
            "OldClass": "gemseo.new_pkg.new_mod.NewClass",
        },
    }
    assert manual == {
        "gemseo.old_pkg": {"Manual": "use New with an extra argument instead"}
    }
    assert class_attribute_renames == {
        "Foo": {"OLD": "new"},
        # Rekeyed from `OldClass` to `NewClass` via the attribute-rename entry above.
        "NewClass": {"OTHER_OLD": "other_new"},
    }


def test_find_dissolved_packages_from_synthetic_renames():
    """A package is dissolved only when nothing else accounts for its name."""
    module_renames = {
        # `gemseo.gone` has no rule of its own and does not exist: dissolved.
        "gemseo.gone.a": "gemseo.new_a",
        "gemseo.gone.b": "gemseo.new_b",
        "gemseo.gone.c": "gemseo.new_a",
        # `gemseo.gone.deep` is dissolved as well, and only its own children count.
        "gemseo.gone.deep.d": "gemseo.new_d",
        # `gemseo.renamed` is renamed itself, and so is its child `gemseo.renamed.x`.
        "gemseo.renamed": "gemseo.other",
        "gemseo.renamed.x": "gemseo.elsewhere.x",
        # `gemseo.renamed.y.z` is covered by the rename of its ancestor.
        "gemseo.renamed.y.z": "gemseo.other.z",
        # `gemseo.core` exists in the tree: not dissolved.
        "gemseo.core.old_module": "gemseo.new_core",
        # A top-level module whose name is the root is never dissolved.
        "gemseo.top": "gemseo.new_top",
    }
    assert aliases._find_dissolved_packages(module_renames) == {
        "gemseo.gone": ("gemseo.new_a", "gemseo.new_b", "gemseo.new_d"),
        "gemseo.gone.deep": ("gemseo.new_d",),
    }


def test_find_dissolved_packages_covered_by_ancestor_rename():
    """A package covered by the rename of an ancestor is not dissolved."""
    module_renames = {"gemseo.old": "gemseo.new", "gemseo.old.sub.mod": "gemseo.n.mod"}
    assert aliases._find_dissolved_packages(module_renames) == {}


def test_find_dissolved_packages_checks_regular_packages(tmp_path, monkeypatch):
    """A package exists when found as a regular one, whether on disk or not.

    A directory holding no `__init__.py`, e.g. one left with only a `__pycache__`, is
    not a package, while a package that is not in a file tree, here in a zip archive
    as in a frozen application, exists. Nothing is imported to check it.

    Args:
        tmp_path: Fixture giving a temporary directory for the packages.
        monkeypatch: Fixture to patch the module search path.
    """
    root = tmp_path / "disk_root"
    (root / "stale" / "__pycache__").mkdir(parents=True)
    (root / "live").mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    (root / "live" / "__init__.py").write_text("", encoding="utf-8")
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zip_file:
        zip_file.writestr("zip_root/__init__.py", "")
        zip_file.writestr("zip_root/live/__init__.py", "")
        zip_file.writestr("zip_root/module.py", "")
    monkeypatch.syspath_prepend(tmp_path)
    monkeypatch.syspath_prepend(archive)
    # A search location with no path entry finder is skipped.
    monkeypatch.syspath_prepend(tmp_path / "missing")
    module_renames = {
        "disk_root.stale.a": "disk_root.new_a",
        "disk_root.live.b": "disk_root.new_b",
        "disk_root.gone.c": "disk_root.new_c",
        "zip_root.live.d": "zip_root.new_d",
        "zip_root.module.e": "zip_root.new_e",
        "zip_root.gone.f": "zip_root.new_f",
        # A module has no submodule.
        "zip_root.module.sub.g": "zip_root.new_g",
    }

    assert aliases._find_dissolved_packages(module_renames) == {
        "disk_root.stale": ("disk_root.new_a",),
        "disk_root.gone": ("disk_root.new_c",),
        "zip_root.gone": ("zip_root.new_f",),
        "zip_root.module.sub": ("zip_root.new_g",),
    }
    assert "disk_root" not in sys.modules
    assert "zip_root" not in sys.modules


def test_dissolved_packages_of_the_configuration():
    """Only `gemseo.settings` is dissolved in the real configuration."""
    assert set(aliases.dissolved_packages) == {"gemseo.settings"}
    assert "gemseo.optimization" in aliases.dissolved_packages["gemseo.settings"]


def test_module_renames_target_existing_modules():
    """Every module rename within GEMSEO points at a module of the package.

    A rename whose target does not exist would shadow the normal import error on the old
    name with a confusing one on the new name. The check is done on the file tree rather
    than by importing, so that the optional dependencies are not needed. A module moved
    to a plugin is not checked, the plugin not being part of the package.
    """
    root = Path(gemseo.__file__).parent
    missing = [
        f"{old} -> {new}"
        for old, new in aliases.module_renames.items()
        if new.startswith("gemseo.")
        and not (root / Path(*new.split(".")[1:])).is_dir()
        and not (root / Path(*new.split(".")[1:])).with_suffix(".py").is_file()
    ]
    assert not missing


def test_attribute_renames_do_not_shadow_modules():
    """No attribute rename is also tabulated as a module rename.

    The `modules:` and `attributes:` sections must be disjoint: an entry in both would
    make an old name resolve either as a submodule or as an attribute depending on how
    it is accessed.
    """
    overlap = [
        f"{module}.{name}"
        for module, renames in aliases.attribute_renames.items()
        for name in renames
        if f"{module}.{name}" in aliases.module_renames
    ]
    assert not overlap


def test_module_and_attribute_values_are_absolute():
    """Every `modules:` and `attributes:` value is a complete path, or `null`.

    A complete `modules:` value is in GEMSEO (`gemseo.*`) or in the package of a plugin
    (`gemseo_*.*`), while an `attributes:` value may be a dotted path in any package,
    or, for a name whose migration cannot be automated, a `TODO <text>` instruction.

    Values are never relative (a bare last segment): each rule is applied once, so
    the runtime and the external codemod never need to chain rules to reach the
    final target.
    """
    text = aliases._config_path.read_text(encoding="utf-8")
    relative = [
        f"{old} -> {new}"
        for section, is_complete in (
            ("modules", aliases._is_absolute),
            ("attributes", aliases._is_dotted),
        )
        for old, new in aliases._parse_section(text, section).items()
        if new != aliases._removed
        and not is_complete(new)
        and not (section == "attributes" and aliases._get_todo_text(new))
    ]
    assert not relative


def test_modules_section_does_not_chain():
    """No `modules:` rule value is itself matched by a LATER rule.

    The external codemod applies `modules:` rules once, top to bottom; if an
    earlier rule's (already final) value were matched by a later key, the codemod
    would rewrite that value a second time.
    """
    text = aliases._config_path.read_text(encoding="utf-8")
    entries = list(aliases._parse_section(text, "modules").items())
    chained = [
        f"{old!r} -> {new!r} matched by the later key {later_key!r}"
        for i, (old, new) in enumerate(entries)
        for later_key, _ in entries[i + 1 :]
        if new == later_key or new.startswith(f"{later_key}.")
    ]
    assert not chained


def test_modules_section_is_nested_most_first():
    """Every `modules:` key appears before every key that is a prefix of it.

    The external codemod applies the rules top to bottom and stops at the first
    match, so a nested (more specific) name's rule must precede its package's.
    """
    text = aliases._config_path.read_text(encoding="utf-8")
    keys = list(aliases._parse_section(text, "modules"))
    position = {key: index for index, key in enumerate(keys)}
    out_of_order = [
        f"{key!r} appears at {position[key]}, after its prefix {ancestor!r} at "
        f"{position[ancestor]}"
        for key in keys
        for cut in range(1, key.count(".") + 1)
        for ancestor in (".".join(key.split(".")[:cut]),)
        if ancestor in position and position[ancestor] < position[key]
    ]
    assert not out_of_order


def test_parse_classes_section_groups_entries_by_class():
    """A class-level rename is grouped under the class it belongs to."""
    text = "classes:\n  Foo:\n    OLD: new\n  Bar:\n    other_old: other_new\n"
    assert aliases._parse_classes_section(text) == {
        "Foo": {"OLD": "new"},
        "Bar": {"other_old": "other_new"},
    }


def test_parse_classes_section_skips_comments_and_blank_lines():
    """Comment and blank lines, at any indentation, are ignored."""
    text = "classes:\n  # a comment\n\n  Foo:\n    # another comment\n\n    OLD: new\n"
    assert aliases._parse_classes_section(text) == {"Foo": {"OLD": "new"}}


def test_parse_classes_section_ignores_nested_blocks():
    """A 4-space empty-value line opens a nested, codemod-only block that is ignored."""
    text = (
        "classes:\n"
        "  Foo:\n"
        "    OLD: new\n"
        "    __init__:\n"
        "      some_param: other_param\n"
        "      other_param: null\n"
    )
    assert aliases._parse_classes_section(text) == {"Foo": {"OLD": "new"}}


def test_parse_classes_section_keeps_renames_of_nested_classes():
    """A dotted entry renaming an attribute of a nested class is kept.

    Only the last segment may differ, and every segment must be an identifier.
    """
    text = (
        "classes:\n"
        "  Foo:\n"
        "    Enum.OLD: Enum.NEW\n"
        "    Enum.MOVED: Other.MOVED\n"
        "    Enum.OTHER: NEW\n"
        "    OTHER: Enum.NEW\n"
        "    Enum.BAD: Enum.not an identifier\n"
    )
    assert aliases._parse_classes_section(text) == {"Foo": {"Enum.OLD": "Enum.NEW"}}


def test_parse_classes_section_drops_null_targets():
    """An entry whose value is `null` is dropped, as a removal cannot be aliased."""
    text = "classes:\n  Foo:\n    OLD: new\n    REMOVED: null\n"
    assert aliases._parse_classes_section(text) == {"Foo": {"OLD": "new"}}


def test_parse_classes_section_drops_classes_left_with_no_entry():
    """A class whose only content is a nested block is dropped."""
    text = "classes:\n  Foo:\n    __init__:\n      old: new\n"
    assert aliases._parse_classes_section(text) == {}


def test_parse_classes_section_stops_at_the_next_section():
    """Parsing stops at the first unindented line following the section name."""
    text = "classes:\n  Foo:\n    OLD: new\nfunctions:\n  Bar:\n    x: y\n"
    assert aliases._parse_classes_section(text) == {"Foo": {"OLD": "new"}}


def test_parse_classes_section_closes_the_class_on_an_unrecognized_line():
    """A YAML alias reference this simple parser does not resolve closes the class.

    Otherwise the entries following it (here `Bar`'s) would be misattributed to
    whichever class was open before it (here `Foo`).
    """
    text = "classes:\n  Foo: *anchor\n    OLD: new\n  Bar:\n    other_old: other_new\n"
    assert aliases._parse_classes_section(text) == {"Bar": {"other_old": "other_new"}}


def test_parse_classes_section_treats_an_anchor_tag_as_an_empty_value():
    """A YAML anchor tag (`&name`) opens the class or the nested block it tags."""
    text = (
        "classes:\n"
        "  Foo: &anchor\n"
        "    OLD: new\n"
        "    __init__: &other-anchor\n"
        "      some_param: other_param\n"
    )
    assert aliases._parse_classes_section(text) == {"Foo": {"OLD": "new"}}


def test_renamed_class_attribute_raises_when_new_name_is_instance_only():
    """Reading a renamed class attribute raises when the new name is instance-only.

    `BaseFormulation.problem` (the new name of the `optimization_problem` alias) is
    only ever set on an instance (in `__init__`), so looking it up on the class
    object itself fails and `_RenamedClassAttribute.__get__` must turn that failure
    into a message naming both the class and the old name.
    """
    from gemseo.formulation.core.base import BaseFormulation

    with (
        pytest.warns(DeprecationWarning, match="'optimization_problem'"),
        pytest.raises(AttributeError) as exc_info,
    ):
        BaseFormulation.optimization_problem  # noqa: B018

    assert str(exc_info.value) == (
        "'BaseFormulation' has no attribute 'problem' "
        "(renamed from 'optimization_problem')"
    )


def test_renamed_class_attribute_message_is_lost_on_a_pydantic_settings_class():
    """A pydantic settings class swallows the crafted rename-hint message.

    `BiLevel_Settings.apply_constraints_to_sub_scenarios` (the new name of the
    `apply_cstr_tosub_scenarios` alias) is a model field, not a live class attribute,
    so the nested `getattr` in `_RenamedClassAttribute.__get__` fails too; the
    `AttributeError` it crafts then propagates out of `type.__getattribute__`, which
    makes Python retry the *original* attribute access through
    `ModelMetaclass.__getattr__` (pydantic's fallback for unknown attributes). That
    fallback raises its own bare `AttributeError`, built from the old name alone, so
    the rename hint never reaches the caller.
    """
    from gemseo.formulation.bilevel_settings import BiLevel_Settings

    with (
        pytest.warns(DeprecationWarning, match="'apply_cstr_tosub_scenarios'"),
        pytest.raises(AttributeError) as exc_info,
    ):
        BiLevel_Settings.apply_cstr_tosub_scenarios  # noqa: B018

    # Documents the current behaviour rather than asserting a desirable one: the
    # message below is pydantic's, not the one `_RenamedClassAttribute.__get__` built.
    assert str(exc_info.value) == "apply_cstr_tosub_scenarios"
